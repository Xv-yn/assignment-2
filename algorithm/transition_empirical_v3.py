"""Independent held-out noisy-label confusion estimator for class-conditional noise.

No clean test arrays are read. Pseudo-clean labels are argmax of a CE predictor
trained on the 80% noisy split. Validation noisy labels are used ONLY to tabulate
T[i,j] = P(S=j | pseudoY=i), optionally restricting to high-margin pseudo labels.

This estimator is *heuristic*. It is exact only when pseudo-clean labels match
latent clean classes or when their misclassification can be modeled accurately.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .data import ROOT, load_noisy_pool
from .transition_estimation import split_noisy_pool, calibrate_temperature, _softmax, estimate_transition_from_probabilities, known_matrix_diagnostics, KNOWN_TRANSITIONS
from .cifar_forward_study import train_one
from .cifar_feature_baseline import image_features
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def empirical_pseudo_confusion(noisy_probabilities, observed_labels, *, fraction=1.0, smoothing=0.0):
    """Held-out noisy label frequencies conditioned on predicted pseudo-clean class.

    Confidence selection based ONLY on predicted probabilities; raw noisy labels
    are not used for selection. For fraction <1, retain highest-confidence
    fraction WITHIN EACH predicted class. Fraction=1 is unbiased wrt selection
    conditional on pseudo-label prediction, but not wrt pseudo-label errors.
    """
    p = np.asarray(noisy_probabilities, np.float64)
    labels = np.asarray(observed_labels)
    if p.ndim != 2 or p.shape[1] != 3 or labels.shape != (len(p),):
        raise ValueError('expected (N,3) probabilities and (N,) observed labels')
    if len(p)<30 or not np.isfinite(p).all() or np.any(p<0) or not np.allclose(p.sum(1),1):
        raise ValueError('invalid probabilities')
    if not np.isin(labels,[0,1,2]).all() or not 0<fraction<=1 or smoothing<0:
        raise ValueError('invalid labels, fraction or smoothing')
    pseudo = np.argmax(p,axis=1)
    rows=[]; counts=[]; selected=[]
    for class_id in range(3):
        idx=np.flatnonzero(pseudo == class_id)
        if len(idx)<5:
            raise ValueError(f'insufficient validation pseudo examples for class {class_id}')
        n=max(5,int(np.ceil(len(idx)*fraction)))
        order=np.argsort(-p[idx,class_id],kind='stable')
        chosen=idx[order[:n]]
        count=np.bincount(labels[chosen],minlength=3).astype(np.float64)
        row=(count+smoothing)/(count.sum()+3*smoothing)
        rows.append(row);counts.append(len(idx));selected.append(len(chosen))
    return np.asarray(rows),{'pseudo_counts':counts,'selected_counts':selected,'fraction':fraction,'smoothing':smoothing}


def estimates_for_split(name, seed, data_dir=ROOT/'data', epochs=12, batch_size=512, device='cpu'):
    import torch
    pool=load_noisy_pool(name,data_dir)
    tr, val=split_noisy_pool(len(pool.noisy_labels),seed)
    ytr=pool.noisy_labels[tr].astype('int64');yval=pool.noisy_labels[val].astype('int64')
    if name=='CIFAR':
        feat=image_features(pool.images)
        scaler=StandardScaler().fit(feat[tr])
        xtr=scaler.transform(feat[tr]).astype('float32');xval=scaler.transform(feat[val]).astype('float32')
        clf=LogisticRegression(C=0.01,max_iter=200,random_state=seed)
        clf.fit(xtr,ytr)
        logits=clf.decision_function(xval)
        model='logistic_RGB_spatial_201'
    else:
        xtr=np.ascontiguousarray(pool.images[tr],dtype=np.float32)/255
        xval=np.ascontiguousarray(pool.images[val],dtype=np.float32)/255
        net,_=train_one(xtr,ytr,xval,yval,method='noisy_ce',matrix=None,seed=seed,epochs=epochs,batch_size=batch_size,device=device)
        net.eval()
        with torch.no_grad():
            logits=net(torch.from_numpy(xval).to(device)).cpu().numpy()
        model='MLP_784_128_64_3'
    temp=calibrate_temperature(logits,yval)
    p=_softmax(logits/temp)
    prior=np.bincount(yval,minlength=3)/len(yval)
    nll=-np.log(np.maximum(p[np.arange(len(yval)),yval],1e-12)).mean()
    information_gain=-np.sum(prior*np.log(prior))-nll
    results={'dataset':name,'seed':seed,'noisy_validation_accuracy':float((p.argmax(1)==yval).mean()),'information_gain_nats':float(information_gain),'temperature':temp,'model':model,'n_val':len(yval),'estimators':{}}
    for method in ('anchor25','hard_all','hard_top50','hard_top20','hard_top10'):
        if method=='anchor25':
            t,meta=estimate_transition_from_probabilities(p,top_k=25)
        else:
            t,meta=empirical_pseudo_confusion(p,yval,fraction={'hard_all':1.,'hard_top50':.5,'hard_top20':.2,'hard_top10':.1}[method])
        diag=known_matrix_diagnostics(name,t)
        results['estimators'][method]={'T_hat':t.tolist(),'metadata':meta,'validation_known_T':diag}
    return results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=[*KNOWN_TRANSITIONS,'CIFAR','all'], default='all')
    ap.add_argument('--data-dir',type=Path,default=ROOT/'data')
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    ap.add_argument('--output-dir',type=Path,default=ROOT/'results/transition_v3')
    ap.add_argument('--epochs',type=int,default=12)
    args=ap.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    names=[*KNOWN_TRANSITIONS,'CIFAR'] if args.dataset=='all' else [args.dataset]
    for name in names:
      for seed in args.seeds:
        out=estimates_for_split(name,seed,args.data_dir,epochs=args.epochs)
        dest=args.output_dir/name/f'seed{seed}.json'
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_text(json.dumps(out,indent=2)+'\n')
        print(name,seed,'noise_NLL_gain',round(out['information_gain_nats'],4),'MAE', {k:round(v['validation_known_T']['mae'],4) for k,v in out['estimators'].items()} if name in KNOWN_TRANSITIONS else {k:np.round(v['T_hat'],3).tolist() for k,v in out['estimators'].items() if k in ('anchor25','hard_all','hard_top20')},flush=True)
if __name__=='__main__':main()
