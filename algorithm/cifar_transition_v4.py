"""CIFAR label-noise diagnostic: texture HOG + colour features, held-out audit.

No access to Xts/Yts. Uses 80% noisy training, 10% noisy calibration,
10% noisy audit. Tests whether a richer SVM learns more than colour-only
logistic regression; it does not prove accuracy of the unknown transition T.

Run from repository root:
  python -m algorithm.cifar_transition_v4 --seeds 0 1 2
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
from skimage.color import rgb2gray
from skimage.feature import hog
from .data import ROOT, load_noisy_pool
from .cifar_feature_baseline import image_features
from .transition_estimation import split_noisy_pool, calibrate_temperature, _softmax
from .transition_empirical_v3 import empirical_pseudo_confusion


def texture_features(images: np.ndarray) -> np.ndarray:
    """Extract deterministic unsigned-gradient HOG per RGB image."""
    if images.ndim != 4 or images.shape[1:] != (32,32,3) or images.dtype != np.uint8:
        raise ValueError('expected CIFAR batch uint8 (N,32,32,3)')
    # HOG expected 3x3 overlapping blocks of 2x2 8-pixel cells and 9 bins = 324.
    output=np.empty((len(images),324), dtype=np.float32)
    for i,im in enumerate(images):
        gray=rgb2gray(im)
        output[i]=hog(gray, orientations=9,pixels_per_cell=(8,8),
                      cells_per_block=(2,2),block_norm='L2-Hys',feature_vector=True)
    return output


def split_cal_audit(validation_indices, seed: int):
    rng=np.random.default_rng(5_000+seed)
    idx=rng.permutation(validation_indices)
    k=len(idx)//2
    return idx[:k],idx[k:]


def row_bootstrap_ci(p, labels, *, repeats=300, seed=1701):
    """Conditional nonparametric bootstrap of pseudo-class/noisy-label counts.
    Bootstrap quantifies finite-sample noise, NOT latent-label bias or identifiability.
    """
    p=np.asarray(p);labels=np.asarray(labels)
    pseudo=p.argmax(1)
    rng=np.random.default_rng(seed)
    matrices=[]
    for _ in range(repeats):
        idx=rng.integers(0,len(labels),len(labels))
        try:
            t,_=empirical_pseudo_confusion(p[idx], labels[idx], fraction=1.0)
        except ValueError:
            continue
        matrices.append(t)
    if not matrices: raise ValueError('bootstrap produced no valid matrix')
    arr=np.asarray(matrices)
    return {'lo':np.quantile(arr,0.025,axis=0).tolist(),
            'hi':np.quantile(arr,0.975,axis=0).tolist(),
            'bootstrap_repeats':len(arr),
            'limitation':'CIs reflect only finite sample variation, not pseudo-label bias'}


def _validated_estimator(probabilities, y_audit, seed):
    est,meta=empirical_pseudo_confusion(probabilities,y_audit,fraction=1.0)
    return {'T_hat':est.tolist(), 'pseudo_metadata':meta,
            'condition_number':float(np.linalg.cond(est)),
            'mean_diagonal':float(np.trace(est)/3),
            'row_bootstrap_95pct':row_bootstrap_ci(probabilities,y_audit,seed=1701+seed)}


def evaluate_seed(features_color,features_texture, y,seed:int):
    train,validate=split_noisy_pool(len(y),seed,train_fraction=0.8)
    cal,audit=split_cal_audit(validate,seed)
    prior=np.bincount(y[train],minlength=3)/len(train)
    constant_nll=float(-np.log(np.maximum(prior[y[audit]],1e-12)).mean())
    prior_acc=float(np.mean(y[audit]==np.argmax(prior)))
    output={'seed':seed,'sizes':{'train':len(train),'calibration':len(cal),'audit':len(audit)},
            'constant_prior':{'audit_nll':constant_nll,'audit_accuracy':prior_acc},
            'models':{},'clean_test_access':False}
    # Fixed a priori models, no hyperparameter sweeps or clean test labels.
    for modelname,features in [('color_logistic',features_color),
                               ('hog_color_rff',np.concatenate((features_color,features_texture),axis=1))]:
        scaler=StandardScaler()
        xtrain=scaler.fit_transform(features[train]); xcal=scaler.transform(features[cal]); xaudit=scaler.transform(features[audit])
        if modelname=='color_logistic':
            model=LogisticRegression(C=0.01,max_iter=250,random_state=seed)
            model.fit(xtrain,y[train]); cal_logits=model.decision_function(xcal);audit_logits=model.decision_function(xaudit)
        else:
            # Approximate the RBF kernel with fixed random Fourier features; CPU bounded.
            rff=RBFSampler(gamma=1.0/features.shape[1], n_components=384,random_state=2026)
            ztrain=rff.fit_transform(xtrain)
            zcal=rff.transform(xcal); zaudit=rff.transform(xaudit)
            model=LogisticRegression(C=2,max_iter=140,random_state=seed)
            model.fit(ztrain,y[train]);cal_logits=model.decision_function(zcal);audit_logits=model.decision_function(zaudit)
        temperature=calibrate_temperature(cal_logits,y[cal])
        p=_softmax(audit_logits/temperature)
        nll=float(log_loss(y[audit],p,labels=[0,1,2]));acc=float(accuracy_score(y[audit],p.argmax(1)))
        output['models'][modelname]={
            'method':'RFF+LogisticRegression' if modelname.startswith('hog') else 'LogisticRegression',
            'features':features.shape[1],'temperature':float(temperature),
            'audit_noisy_nll':nll,'audit_noisy_accuracy':acc,
            'audit_information_gain_nats':constant_nll-nll,
            'estimator':_validated_estimator(p,y[audit],seed),
            'warning':'Transition matrix is heuristic; no true CIFAR T available'}
    return output


def run(data_dir=ROOT/'data', seeds=(0,1,2), out_dir=ROOT/'results/cifar_transition_v4'):
    pool=load_noisy_pool('CIFAR',data_dir)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    color=image_features(pool.images)
    texture=texture_features(pool.images)
    print('Computed features:',color.shape, texture.shape, flush=True)
    entries=[]
    for seed in seeds:
        path=out_dir/f'seed{seed}.json'
        if path.exists():
            record=json.loads(path.read_text())
            print(f'Loaded existing seed {seed}',flush=True)
        else:
            record=evaluate_seed(color,texture,pool.noisy_labels,seed)
            path.write_text(json.dumps(record,indent=2)+'\n')
        entries.append(record)
        print(f'Seed {seed}: '+ '; '.join(f'{k} NLL={v["audit_noisy_nll"]:.5f}, gain={v["audit_information_gain_nats"]:.5f}, acc={v["audit_noisy_accuracy"]:.4f}' for k,v in record['models'].items()),flush=True)
    summary={'status':'diagnostic_no_clean_test','seeds':list(seeds),'comparison':{}}
    for name in ('color_logistic','hog_color_rff'):
        def values(key):return np.asarray([r['models'][name][key] for r in entries])
        summary['comparison'][name]={'audit_information_gain_mean':float(values('audit_information_gain_nats').mean()),
          'audit_information_gain_sd':float(values('audit_information_gain_nats').std(ddof=1)) if len(entries)>1 else None,
          'audit_accuracy_mean':float(values('audit_noisy_accuracy').mean()),
          'audit_nll_mean':float(values('audit_noisy_nll').mean()),
          'estimated_T_mean':np.mean([r['models'][name]['estimator']['T_hat'] for r in entries],axis=0).tolist(),
          'estimated_T_condition_number_mean':float(np.mean([r['models'][name]['estimator']['condition_number'] for r in entries]))}
    summary['audit_gain_pairwise_delta']= [float(r['models']['hog_color_rff']['audit_information_gain_nats']-r['models']['color_logistic']['audit_information_gain_nats']) for r in entries]
    (out_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--out-dir',type=Path,default=ROOT/'results/cifar_transition_v4');p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    args=p.parse_args(); print(json.dumps(run(args.data_dir,args.seeds,args.out_dir),indent=2))
