"""CIFAR noisy-label transition matrix GPU pilot, v5.

Training / calibration / independent audit = 80% / 10% / 10% of Xtr/Str.
NEVER reads clean Xts/Yts.  Output matrices are candidate pseudo-label estimates.
Run from group repository root: python -m algorithm.cifar_gpu_v5 --device cuda --seed 0
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np

from .data import ROOT, load_noisy_pool, preprocess_images
from .transition_estimation import split_noisy_pool, calibrate_temperature, _softmax, estimate_transition_from_probabilities
from .transition_empirical_v3 import empirical_pseudo_confusion
from .cifar_transition_v4 import split_cal_audit, row_bootstrap_ci


def seed_everything(seed: int) -> None:
    import torch
    torch.manual_seed(2026+seed)
    np.random.seed(2026+seed)
    random.seed(2026+seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2026+seed)
        torch.backends.cudnn.benchmark = False


def augment_batch(x):
    """Vectorized per-image reflect-crop and horizontal flip; training images only."""
    import torch
    import torch.nn.functional as F
    if x.ndim != 4 or x.shape[2:] != (32,32):
        raise ValueError('expected CIFAR tensor (B,C,32,32)')
    b,c,_,_=x.shape
    padded=F.pad(x,(4,4,4,4),mode='reflect').permute(0,2,3,1)
    ys=torch.randint(0,9,(b,1,1),device=x.device)+torch.arange(32,device=x.device)[None,:,None]
    xs=torch.randint(0,9,(b,1,1),device=x.device)+torch.arange(32,device=x.device)[None,None,:]
    crop=padded[torch.arange(b,device=x.device)[:,None,None],ys,xs].permute(0,3,1,2).contiguous()
    flip=torch.rand((b,),device=x.device)<0.5
    return torch.where(flip[:,None,None,None],crop.flip(-1),crop)


def make_model(width=32):
    import torch
    import torch.nn as nn
    class Residual(nn.Module):
        def __init__(self, c_in, c_out, stride=1):
            super().__init__()
            self.f=nn.Sequential(nn.Conv2d(c_in,c_out,3,stride,1,bias=False),nn.BatchNorm2d(c_out),nn.ReLU(inplace=True),
                                 nn.Conv2d(c_out,c_out,3,1,1,bias=False),nn.BatchNorm2d(c_out))
            self.skip=(nn.Identity() if c_in==c_out and stride==1 else
                       nn.Sequential(nn.Conv2d(c_in,c_out,1,stride,bias=False),nn.BatchNorm2d(c_out)))
            self.relu=nn.ReLU(inplace=True)
        def forward(self,x):return self.relu(self.f(x)+self.skip(x))
    return nn.Sequential(nn.Conv2d(3,width,3,1,1,bias=False),nn.BatchNorm2d(width),nn.ReLU(inplace=True),
                         Residual(width,width),Residual(width,width*2,stride=2),Residual(width*2,width*2),
                         Residual(width*2,width*4,stride=2),Residual(width*4,width*4),
                         nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(width*4,3))


def evidence_gate(gain, condition, pseudo_counts, *, gain_min=.01, condition_max=30., min_pseudo=100):
    checks={'information_gain_above_0.01': bool(np.isfinite(gain) and gain>=gain_min),
            'condition_number_at_most_30': bool(np.isfinite(condition) and condition<=condition_max),
            'each_pseudo_class_at_least_100': bool(min(pseudo_counts)>=min_pseudo)}
    return {'eligible_for_downstream_confirmation':all(checks.values()), 'checks':checks,
            'warning':'Gate is a screening rule only. Pseudo-label confusion is generally biased; passing is not proof of true T.'}


def _atomic_json(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def evaluate_candidate(audit_logits, calib_logits, y_audit, y_calib, train_prior, *, seed=0):
    """Strictly after checkpoint selection: calibrate on cal, report audit only."""
    p_cal=np.asarray(calib_logits,dtype=np.float64)
    p_aud=np.asarray(audit_logits,dtype=np.float64)
    if len(y_audit)<30 or len(y_calib)<30:raise ValueError('Insufficient held-out data')
    temperature=calibrate_temperature(p_cal,np.asarray(y_calib))
    p=_softmax(p_aud/temperature)
    y=np.asarray(y_audit,dtype=int)
    prior=np.asarray(train_prior,dtype=np.float64)
    prior_nll=float(-np.log(np.maximum(prior[y],1e-12)).mean())
    audit_nll=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
    gain=prior_nll-audit_nll
    counts=np.bincount(p.argmax(1),minlength=3).astype(int).tolist()
    result={'temperature':float(temperature),'noisy_audit_nll':audit_nll,'prior_audit_nll':prior_nll,
            'noisy_audit_information_gain_nats':float(gain),
            'noisy_audit_accuracy':float((p.argmax(1)==y).mean()),'pseudo_counts':counts}
    if min(counts)>=5:
        mat,info=empirical_pseudo_confusion(p,y,fraction=1.0)
        anchor,anchor_meta=estimate_transition_from_probabilities(p,top_k=25)
        cond=float(np.linalg.cond(mat))
        result.update({'estimated_T':mat.tolist(),'estimated_T_condition_number':cond if np.isfinite(cond) else None,
                       'anchor_T_diagnostic':anchor.tolist(),
                       'pseudo_vs_anchor_frobenius':float(np.linalg.norm(mat-anchor)),
                       'bootstrap_CI':row_bootstrap_ci(p,y,repeats=150,seed=1701+seed),
                       'estimation_counts':info,'anchor_info':anchor_meta})
    else:
        cond=float('inf')
        result.update({'estimated_T':None,'estimated_T_condition_number':None,
                       'reason':'One or more predicted pseudo-clean classes have fewer than 5 examples'})
    result['quality_gate']=evidence_gate(gain,cond,counts)
    return result


def run(*,data_dir=ROOT/'data', seed=0, epochs=35, batch_size=256, device='cuda',
        smoke=False, out_dir=ROOT/'results/cifar_gpu_v5', width=32, patience=12, resume=False):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader,TensorDataset
    if (seed<0 or epochs<1 or batch_size<1 or width<8 or patience<1):raise ValueError('bad hyperparameters')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('No CUDA detected. This machine can only do --device cpu --smoke checks.')
    if smoke and epochs>2:raise ValueError('Smoke is restricted to <= 2 epochs')
    if device=='cpu':torch.set_num_threads(min(4,torch.get_num_threads()))
    seed_everything(seed)
    data_dir=Path(data_dir); out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    target=out_dir/f'seed{seed}.json'
    checkpoint_path=out_dir/f'seed{seed}_checkpoint.pt'
    if target.exists() and not smoke:
        existing=json.loads(target.read_text())
        if existing.get('status','').startswith('completed_full_') and resume:
            settings=existing.get('training_config',{})
            if (existing.get('seed')!=seed or existing.get('epoch_budget')!=epochs or
                settings.get('batch_size')!=batch_size or settings.get('width')!=width or
                settings.get('patience')!=patience):
                raise ValueError(f'Existing result settings differ; use a new out-dir: {target}')
            print('Existing completed result, not re-running:',target,flush=True)
            return existing
        raise ValueError(f'Result already exists: {target}. Use --resume for matching completed runs or a new output folder.')
    pool=load_noisy_pool('CIFAR',data_dir)
    train_idx,val_idx=split_noisy_pool(len(pool.noisy_labels),seed)
    cal_idx,aud_idx=split_cal_audit(val_idx,seed)
    if len(set(train_idx)&set(cal_idx)) or len(set(train_idx)&set(aud_idx)) or len(set(cal_idx)&set(aud_idx)):
        raise RuntimeError('Leakage: train/cal/audit must be disjoint')
    if smoke:
        train_idx=train_idx[:96];cal_idx=cal_idx[:48];aud_idx=aud_idx[:48]
    xtr=torch.from_numpy(preprocess_images(pool.images[train_idx],'CIFAR'))
    xcal=torch.from_numpy(preprocess_images(pool.images[cal_idx],'CIFAR'))
    xaudit=torch.from_numpy(preprocess_images(pool.images[aud_idx],'CIFAR'))
    ytr=torch.from_numpy(pool.noisy_labels[train_idx].astype(np.int64))
    ycal=torch.from_numpy(pool.noisy_labels[cal_idx].astype(np.int64))
    yaud=pool.noisy_labels[aud_idx].astype(np.int64)
    train_gen=torch.Generator().manual_seed(5_000+seed)
    train=DataLoader(TensorDataset(xtr,ytr),batch_size=batch_size,shuffle=True,generator=train_gen,num_workers=0)
    cal=DataLoader(TensorDataset(xcal,ycal),batch_size=batch_size,shuffle=False,num_workers=0)
    audit=DataLoader(TensorDataset(xaudit,torch.from_numpy(yaud)),batch_size=batch_size,shuffle=False,num_workers=0)
    net=make_model(width).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=8e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs,eta_min=1e-5)
    best=float('inf');best_epoch=0;best_state=None;hist=[];bad=0
    device_label=str(device)
    start=time.monotonic()
    start_epoch=1
    checkpoint_identity={'seed':seed,'epochs':epochs,'batch_size':batch_size,'width':width,
                         'patience':patience,'data_path':str((data_dir/'CIFAR.npz').resolve())}
    if checkpoint_path.exists() and not smoke:
        if not resume:raise RuntimeError(f'Interrupted checkpoint exists at {checkpoint_path}; rerun with --resume')
        state=torch.load(checkpoint_path,map_location=device,weights_only=False)
        if state['identity']!=checkpoint_identity:raise ValueError('Checkpoint training config/data path differs')
        net.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        best=float(state['best']);best_epoch=int(state['best_epoch']);best_state=state['best_state']
        hist=state['history'];bad=int(state['bad']);start_epoch=int(state['epoch'])+1
        torch.set_rng_state(state['torch_rng'].cpu())
        train_gen.set_state(state['loader_rng'].cpu())
        if device.startswith('cuda') and state.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([t.cpu() for t in state['cuda_rng']])
        print(f'Resuming seed {seed} from epoch {start_epoch}',flush=True)
    elif resume and not smoke:
        print('No checkpoint found; starting a fresh seed',flush=True)
    for epoch in range(start_epoch,epochs+1):
        net.train();total=0
        for xb,yb in train:
            xb=xb.to(device,non_blocking=True);yb=yb.to(device,non_blocking=True)
            if not smoke:xb=augment_batch(xb)
            opt.zero_grad(set_to_none=True)
            loss=F.cross_entropy(net(xb),yb)
            loss.backward();opt.step();total+=float(loss.detach())*len(xb)
        net.eval();val_sum=0
        with torch.inference_mode():
            for xb,yb in cal:
                val_sum+=float(F.cross_entropy(net(xb.to(device)),yb.to(device)))*len(xb)
        cal_nll=val_sum/len(cal_idx)
        row={'epoch':epoch,'train_noisy_nll':total/len(train_idx),'calibration_noisy_nll':cal_nll,
             'lr':float(opt.param_groups[0]['lr'])}
        hist.append(row);print(f'V5 seed={seed} epoch={epoch}: train {row["train_noisy_nll"]:.5f}; cal {cal_nll:.5f}',flush=True)
        if cal_nll<best-1e-5:
            best=cal_nll;best_epoch=epoch;bad=0
            best_state=copy.deepcopy({k:v.detach().cpu().clone() for k,v in net.state_dict().items()})
        else: bad+=1
        sched.step()
        # Save lightweight progress JSON each epoch; best parameters are restored at completion.
        if not smoke:
            _atomic_json(out_dir/f'seed{seed}_progress.json',{'seed':seed,'epochs_completed':epoch,'best_epoch':best_epoch,'best_cal_nll':best,'history':hist,'status':'running'})
            ck={'identity':checkpoint_identity,'epoch':epoch,'model':net.state_dict(),
                'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),'best':best,
                'best_epoch':best_epoch,'best_state':best_state,'history':hist,'bad':bad,
                'torch_rng':torch.get_rng_state(),'loader_rng':train_gen.get_state(),
                'cuda_rng':torch.cuda.get_rng_state_all() if device.startswith('cuda') else None}
            temp_ck=checkpoint_path.with_suffix('.pt.tmp')
            torch.save(ck,temp_ck)
            os.replace(temp_ck,checkpoint_path)
        if bad>=patience:break
    if best_state is None:raise RuntimeError('No model checkpoint available')
    net.load_state_dict(best_state);net.eval()
    def logits(loader):
        pieces=[]
        with torch.inference_mode():
            for xb,_ in loader:pieces.append(net(xb.to(device)).cpu().numpy())
        return np.concatenate(pieces)
    prior=np.bincount(ytr.numpy(),minlength=3)/len(ytr)
    diagnostics=evaluate_candidate(logits(audit),logits(cal),yaud,ycal.numpy(),prior,seed=seed)
    file_hash=hashlib.sha256((data_dir/'CIFAR.npz').read_bytes()).hexdigest()
    status='cpu_smoke_not_for_reporting' if smoke else 'completed_full_'+('gpu' if device.startswith('cuda') else 'cpu')+'_diagnostic'
    result={'status':status,'model':'residual_cnn_v5','seed':seed,'best_epoch':best_epoch,
            'epochs_completed':len(hist),'epoch_budget':epochs,'device':device_label,
            'samples':{'train':len(train_idx),'calibration':len(cal_idx),'independent_audit':len(aud_idx)},
            'training_config':{'width':width,'batch_size':batch_size,'patience':patience,
                               'random_horizontal_flip_and_reflect_crop':not smoke,'optimizer':'AdamW','initial_lr':8e-4},
            'data_sha256':file_hash,'duration_seconds':float(time.monotonic()-start),
            'history':hist,'clean_test_access':False,**diagnostics,
            'warning':'The unknown true CIFAR noise transition cannot be verified from noisy labels alone.'}
    _atomic_json(target,result)
    if not smoke:
        # Only checkpoint frozen model; do not store training arrays / clean labels.
        torch.save({'state_dict':best_state,'architecture':'residual_cnn_v5','seed':seed,
                    'best_epoch':best_epoch,'data_sha256':file_hash},out_dir/f'seed{seed}_model.pt')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=ROOT/'data')
    p.add_argument('--out-dir',type=Path,default=ROOT/'results/cifar_gpu_v5')
    p.add_argument('--device',default='cuda');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=35);p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--width',type=int,default=32);p.add_argument('--patience',type=int,default=12)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    record=run(data_dir=a.data_dir,out_dir=a.out_dir,device=a.device,seed=a.seed,
               epochs=a.epochs,batch_size=a.batch_size,width=a.width,patience=a.patience,smoke=a.smoke,resume=a.resume)
    keys=('status','seed','best_epoch','noisy_audit_accuracy','noisy_audit_information_gain_nats','estimated_T_condition_number','quality_gate')
    print('FINAL',json.dumps({k:record.get(k) for k in keys},indent=2))

if __name__=='__main__':main()
