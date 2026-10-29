"""Reproducible diagnostic: noisy CE versus estimated-T forward correction.

This is *not* a substitute for the full Assignment 2 grid (3 datasets,
>=2 robust classifiers, >=10 independent splits). It probes whether a weak
anchor estimated T helps on CIFAR. No clean test arrays are used for model
selection, transition estimation, early stopping, or preprocessing statistics.

From the project root (with data/CIFAR.npz present):
  python -m algorithm.cifar_forward_study --seeds 0 1 2 \
      --epochs 20 --output-dir results/cifar_forward

Dependencies: numpy, scikit-learn, torch, and original algorithm.data/transition_estimation/
cifar_feature_baseline modules. Includes paired noisy-CE and Forward Loss Correction
with an identical MLP, split, initialization seed, validation criterion, and features.
"""
from __future__ import annotations
import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
from .data import ROOT, load_noisy_pool
from .cifar_feature_baseline import image_features, run as estimate_cifar_fast
from .transition_estimation import split_noisy_pool


def validate_transition(t: np.ndarray) -> np.ndarray:
    arr = np.asarray(t, dtype=np.float64)
    if arr.shape != (3, 3) or not np.isfinite(arr).all() or np.min(arr) < 0:
        raise ValueError("Transition must be non-negative finite 3x3")
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Transition rows must sum to 1")
    return arr


def noisy_nll_from_logits(logits, noisy_y, matrix=None):
    """True-label logits => observed-label loss, T[i,j]=P(S=j|Y=i)."""
    import torch
    import torch.nn.functional as F
    probs = torch.softmax(logits, dim=1)
    if matrix is None:
        return F.cross_entropy(logits, noisy_y)
    t = torch.as_tensor(validate_transition(matrix), device=logits.device, dtype=logits.dtype)
    observed = (probs @ t).clamp_min(1e-12)
    return -torch.log(observed.gather(1, noisy_y.reshape(-1, 1))).mean()


def prepare_data(data_dir: Path, seed: int):
    """Fit scaler only to Xtr[training partition]; return clean test only for final evaluation.

    Deliberately separate the clean test access from training loops below.
    """
    from .data import load_noisy_pool
    from sklearn.preprocessing import StandardScaler
    pool = load_noisy_pool('CIFAR', data_dir)
    tr, val = split_noisy_pool(len(pool.noisy_labels), seed)
    features_train = image_features(pool.images[tr])
    features_val = image_features(pool.images[val])
    scaler = StandardScaler().fit(features_train)
    return (scaler.transform(features_train).astype('float32'),
            pool.noisy_labels[tr].astype('int64'),
            scaler.transform(features_val).astype('float32'),
            pool.noisy_labels[val].astype('int64'), scaler)


def train_one(xtr, ytr, xval, yval, *, method, matrix, seed, epochs, batch_size, device='cpu'):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    if method not in ('noisy_ce', 'forward', 'gce'):
        raise ValueError(method)
    if method == 'forward':
        matrix = validate_transition(matrix)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(10_000 + seed); random.seed(10_000 + seed)
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    model = nn.Sequential(nn.Linear(xtr.shape[1], 128), nn.ReLU(),
                          nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    train_data = TensorDataset(torch.from_numpy(xtr),torch.from_numpy(ytr))
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(20_000 + seed), num_workers=0)
    vx = torch.from_numpy(xval).to(device); vy=torch.from_numpy(yval).to(device)
    train_matrix = (torch.tensor(matrix, device=device, dtype=torch.float32) if method=='forward' else None)
    def loss_with_matrix(logits, labels):
        if method == 'gce':
            # GCE(q=0.7) from the team's draft final.json: (1 - p_y**q) / q.
            # Unlike CE it bounds the influence of low-probability noisy labels.
            chosen = torch.softmax(logits, dim=1).gather(1, labels[:, None]).clamp_min(1e-12)
            return ((1 - chosen.pow(0.7)) / 0.7).mean()
        if train_matrix is None:
            return F.cross_entropy(logits,labels)
        return -torch.log((torch.softmax(logits, 1) @ train_matrix).clamp_min(1e-12).gather(1,labels[:,None])).mean()
    best = float('inf'); best_epoch=0; state=None; history=[]
    for epoch in range(1, epochs+1):
        model.train(); total=0.
        for xb,yb in loader:
            xb=xb.to(device);yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=loss_with_matrix(model(xb),yb)
            loss.backward(); optimizer.step()
            total+=loss.item()*len(xb)
        model.eval()
        with torch.no_grad():
            logits=model(vx)
            # For GCE, choose epoch by noisy-label CE, not the non-NLL GCE objective.
            val_loss=float((F.cross_entropy(logits,vy) if method=='gce' else loss_with_matrix(logits,vy)).item())
            noisy_acc=float((logits.argmax(1)==vy).float().mean().item())
        history.append({'epoch':epoch,'train_objective':total/len(xtr),
                        'validation_noisy_nll':val_loss, 'raw_prediction_noisy_accuracy':noisy_acc})
        if val_loss < best-1e-6:
            best=val_loss;best_epoch=epoch;state=copy.deepcopy(model.state_dict())
    model.load_state_dict(state);model.eval()
    return model, {'method':method,'split_seed':seed,'initialization_seed':10_000+seed,
                   'best_epoch':best_epoch,'best_validation_noisy_nll':best,
                   'epochs_max':epochs,'history':history,'model_selection':'noisy validation NLL',
                   'train_count':len(xtr),'validation_count':len(xval),
                   'gce_q':0.7 if method=='gce' else None}


def evaluate_clean_test(model, scaler, *, data_dir, device):
    """The ONLY function that opens clean Xts/Yts; invoked after training is frozen."""
    import torch
    with np.load(Path(data_dir)/'CIFAR.npz',allow_pickle=False) as z:
        x_test=z['Xts'];y_test=z['Yts'].copy()
    xt=scaler.transform(image_features(x_test)).astype(np.float32)
    model.eval()
    with torch.no_grad():
        pred=model(torch.from_numpy(xt).to(device)).argmax(1).cpu().numpy()
    return {'test_top1_accuracy':float((pred==y_test).mean()),
            'test_correct':int((pred==y_test).sum()),'test_count':len(y_test),
            'evaluation_only':'Xts/Yts read only after all training and selection'}


def run_one(seed:int, *, data_dir:Path, epochs:int=20, batch_size:int=256,
            device:str='cpu',top_k:int=25):
    transition_result = estimate_cifar_fast(dataset_dir=data_dir,seed=seed,top_k=top_k)
    t=validate_transition(transition_result['estimated_transition_matrix'])
    xtr,ytr,xval,yval,scaler=prepare_data(data_dir,seed)
    runs={}
    for method in ('noisy_ce','forward'):
        net, metadata = train_one(xtr,ytr,xval,yval,method=method,
                                  matrix=t if method=='forward' else None,
                                  seed=seed,epochs=epochs,batch_size=batch_size,device=device)
        scores=evaluate_clean_test(net,scaler,data_dir=data_dir,device=device)
        runs[method]={**metadata,**scores}
    return {'status':'preliminary_two_methods_one_dataset_not_assignment_final',
            'dataset':'CIFAR','seed':seed,'feature_pipeline':'8x8 block mean RGB + global colour/edges; train-only StandardScaler',
            'architecture':'MLP(204-128-64-3) identical for both loss functions',
            'transition_source':'noisy-only logistic top-k anchor; uses same 80/20 split',
            'transition_matrix':t.tolist(),
            'transition_diagnostics':{'information_gain_nats':transition_result['information_gain_nats'],
                                      'validation_noisy_accuracy':transition_result['validation_noisy_accuracy']},
            'results':runs, 'paired_forward_minus_ce_accuracy':runs['forward']['test_top1_accuracy']-runs['noisy_ce']['test_top1_accuracy'],
            'limitations':'Weak estimator; 3 seed pilot; noisy CE is diagnostic baseline, not one of two required robust methods.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--device',default='cpu')
    p.add_argument('--top-k',type=int,default=25)
    p.add_argument('--output-dir',type=Path,default=ROOT/'results/cifar_forward')
    a=p.parse_args()
    if a.epochs<=0 or a.batch_size<=0 or len(set(a.seeds))!=len(a.seeds):p.error('invalid arguments')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for seed in a.seeds:
        print(f'Run split {seed}...',flush=True)
        r=run_one(seed,data_dir=a.data_dir,epochs=a.epochs,batch_size=a.batch_size,
                  device=a.device,top_k=a.top_k)
        (a.output_dir/f'seed{seed}.json').write_text(json.dumps(r,indent=2))
        print(f"seed={seed}: CE={r['results']['noisy_ce']['test_top1_accuracy']:.4f}, "
              f"Forward={r['results']['forward']['test_top1_accuracy']:.4f}, "
              f"difference={r['paired_forward_minus_ce_accuracy']:+.4f}",flush=True)
        results.append(r)
    pairs=np.array([[x['results'][m]['test_top1_accuracy'] for m in ('noisy_ce','forward')]
                    for x in results])
    diff=pairs[:,1]-pairs[:,0]
    summary={'status':'preliminary_paired_CIFAR_comparison_not_final',
             'seeds':a.seeds, 'epochs':a.epochs,'method_names':['noisy_ce','forward'],
             'mean_test_accuracy':pairs.mean(axis=0).tolist(),
             'sample_std_test_accuracy':pairs.std(axis=0,ddof=1).tolist() if len(results)>1 else None,
             'paired_differences':diff.tolist(),'paired_mean_difference':float(diff.mean()),
             'paired_sample_std_difference':float(diff.std(ddof=1)) if len(results)>1 else None,
             'accuracy_unit':'fraction (multiply by 100 for percent)',
             'note':'Results computed on clean test labels once per finalized run. No test-based tuning. This does not satisfy >=10 seeds / >=2 robust classifiers / 3 datasets.'}
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2))
    print('SUMMARY',json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
