"""CIFAR CNN noisy-label predictor for transition diagnostics (GPU recommended).

Reads *only* Xtr and Str via load_noisy_pool. For each split:
  80% train, 10% noisy-calibration, 10% independent noisy-audit.
Early stopping and temperature calibration use calibration noisy labels.
Final independent audit measures noisy-label NLL and outputs a heuristic
pseudo-label confusion T_hat. No clean test access, no test-score tuning.

GPU full example:
  python -m algorithm.cifar_cnn_gpu_v4 --device cuda --seed 0 --epochs 35 \
     --output results/cifar_cnn_gpu_v4/seed0.json
CPU development smoke only:
  python -m algorithm.cifar_cnn_gpu_v4 --device cpu --smoke --seed 0 --epochs 1
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import random
import numpy as np
from .data import load_noisy_pool, preprocess_images, ROOT
from .transition_estimation import split_noisy_pool, calibrate_temperature, _softmax
from .transition_empirical_v3 import empirical_pseudo_confusion
from .cifar_transition_v4 import row_bootstrap_ci, split_cal_audit


def quality_guard(gain, cond, min_pseudo, *, gain_threshold=0.01, cond_limit=30, min_per_class=100):
    """Conservative *engineering* gate, not proof of transition correctness."""
    checks={'audit_information_gain_ge_0.01':bool(gain>=gain_threshold),
            'matrix_condition_number_le_30':bool(np.isfinite(cond) and cond<=cond_limit),
            'pseudo_examples_per_class_ge_100':bool(min_pseudo>=min_per_class)}
    return {'safe_to_try_in_forward':bool(all(checks.values())), 'checks':checks,
            'note':'A pass only permits downstream validation; it does not certify true T or guarantee improvement.'}


def make_model():
    import torch.nn as nn
    return nn.Sequential(
        nn.Conv2d(3,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),
        nn.Conv2d(32,32,3,padding=1),nn.BatchNorm2d(32),nn.ReLU(inplace=True),nn.MaxPool2d(2),
        nn.Conv2d(32,64,3,padding=1),nn.BatchNorm2d(64),nn.ReLU(inplace=True),nn.MaxPool2d(2),
        nn.Conv2d(64,128,3,padding=1),nn.BatchNorm2d(128),nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((2,2)),nn.Flatten(),
        nn.Linear(128*2*2,128),nn.ReLU(inplace=True),nn.Dropout(0.2),nn.Linear(128,3))


def run(*, data_dir=ROOT/'data',seed=0,epochs=35,batch_size=256,device='cuda',smoke=False):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available: use GPU machine, or --smoke --device cpu for wiring check')
    if epochs<1 or batch_size<1:raise ValueError('epochs and batch_size must be positive')
    if smoke and epochs>2:raise ValueError('smoke supports at most 2 epochs to avoid misleading results')
    if device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.manual_seed(2026+seed);np.random.seed(2026+seed);random.seed(2026+seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(2026+seed)
    pool=load_noisy_pool('CIFAR',data_dir)
    tr,val=split_noisy_pool(len(pool.noisy_labels),seed)
    cal,aud=split_cal_audit(val,seed)
    if smoke:
        tr=tr[:192];cal=cal[:96];aud=aud[:96]
    xtr=torch.from_numpy(preprocess_images(pool.images[tr],'CIFAR'))
    xcal=torch.from_numpy(preprocess_images(pool.images[cal],'CIFAR'))
    xaudit=torch.from_numpy(preprocess_images(pool.images[aud],'CIFAR'))
    ytr=torch.from_numpy(pool.noisy_labels[tr].astype(np.int64))
    ycal=torch.from_numpy(pool.noisy_labels[cal].astype(np.int64))
    yaud=pool.noisy_labels[aud].astype(np.int64)
    train_loader=DataLoader(TensorDataset(xtr,ytr),shuffle=True,batch_size=batch_size,
                            generator=torch.Generator().manual_seed(2026+seed))
    cal_loader=DataLoader(TensorDataset(xcal,ycal),shuffle=False,batch_size=batch_size)
    aud_loader=DataLoader(TensorDataset(xaudit,torch.from_numpy(yaud)),shuffle=False,batch_size=batch_size)
    net=make_model().to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=1e-4)
    best,epoch_best,state=float('inf'),0,None
    history=[]
    for epoch in range(1,epochs+1):
        net.train();total=0
        for xb,yb in train_loader:
            xb=xb.to(device);yb=yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=net(xb);loss=F.cross_entropy(logits,yb)
            loss.backward();optimizer.step();total+=float(loss.detach())*len(xb)
        net.eval();loss_sum=0
        with torch.inference_mode():
            for xb,yb in cal_loader:
                logits=net(xb.to(device));loss_sum+=float(F.cross_entropy(logits,yb.to(device)))*len(xb)
        cal_nll=loss_sum/len(cal)
        history.append({'epoch':epoch,'train_noisy_nll':total/len(tr),'calibration_noisy_nll':cal_nll})
        print(f'Epoch {epoch}/{epochs}: train noisy NLL {total/len(tr):.4f}; noisy cal NLL {cal_nll:.4f}',flush=True)
        if cal_nll < best-1e-6:
            best,epoch_best,state=cal_nll,epoch,copy.deepcopy({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
    net.load_state_dict(state);net.eval()
    def logits_for(loader):
        out=[]
        with torch.inference_mode():
            for xb,_ in loader:out.append(net(xb.to(device)).cpu().numpy())
        return np.concatenate(out)
    cal_logits=logits_for(cal_loader)
    audit_logits=logits_for(aud_loader)
    t=calibrate_temperature(cal_logits,pool.noisy_labels[cal])
    p=_softmax(audit_logits/t)
    prior=np.bincount(pool.noisy_labels[tr],minlength=3)/len(tr)
    prior_nll=float(-np.log(np.maximum(prior[yaud],1e-12)).mean())
    audit_nll=float(-np.log(np.maximum(p[np.arange(len(aud)),yaud],1e-12)).mean())
    gain=prior_nll-audit_nll
    pseudo_counts=np.bincount(p.argmax(axis=1),minlength=3).tolist()
    if min(pseudo_counts)>=5:
        mat,meta=empirical_pseudo_confusion(p,yaud,fraction=1.0)
        cond=float(np.linalg.cond(mat))
        ci=row_bootstrap_ci(p,yaud,seed=seed)
    else:
        # In particular, one-epoch smoke checks can collapse into one class.
        # Never invent missing transition rows or misrepresent them as estimates.
        mat,cond,ci=None,float('inf'),None
    status=('cpu_smoke_not_valid_for_reporting' if smoke else
            'full_noisy_validation_diagnostic' if mat is not None else 'invalid_missing_pseudo_classes')
    return {'status':status,
      'split_seed':seed,'device':device,'best_epoch':epoch_best,'epochs_run':epochs,
      'train_count':len(tr),'calibration_count':len(cal),'audit_count':len(aud),
      'temperature':float(t),'noisy_audit_accuracy':float((p.argmax(1)==yaud).mean()),
      'noisy_audit_nll':audit_nll,'constant_prior_audit_nll':prior_nll,
      'audit_information_gain_nats':gain,'estimated_T':None if mat is None else mat.tolist(),
      'estimated_T_condition_number':None if not np.isfinite(cond) else cond,'pseudo_counts':pseudo_counts,
      'bootstrap_ci':ci,
      'quality_guard':quality_guard(gain,cond,min(pseudo_counts)),
      'history':history,'has_clean_test_access':False,
      'warning':'T is a pseudo-label heuristic, and no CIFAR true T is supplied.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--seed',type=int,default=0);p.add_argument('--epochs',type=int,default=35)
    p.add_argument('--batch-size',type=int,default=256);p.add_argument('--device',default='cuda')
    p.add_argument('--smoke',action='store_true');p.add_argument('--output',type=Path)
    a=p.parse_args()
    res=run(data_dir=a.data_dir,seed=a.seed,epochs=a.epochs,batch_size=a.batch_size,device=a.device,smoke=a.smoke)
    print(json.dumps({k:v for k,v in res.items() if k!='history'},indent=2),flush=True)
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(res,indent=2)+'\n')

if __name__=='__main__':main()
