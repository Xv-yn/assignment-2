"""Fast, clean-test-free CIFAR noisy-label signal diagnostic.

This is a baseline/probe, NOT a substitute for the assignment's two classifiers.
The model fits noisy training labels only, calibrates with noisy validation labels,
then returns an anchor-based transition estimate marked diagnostic-only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import load_noisy_pool, ROOT
from .transition_estimation import split_noisy_pool, estimate_transition_from_probabilities, calibrate_temperature, _softmax


def image_features(images):
    if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
        raise ValueError('Expected CIFAR images (N,32,32,3)')
    x = images.astype(np.float32) / 255
    low = x.reshape(-1, 8, 4, 8, 4, 3).mean(axis=(2,4)).reshape(len(x), -1)
    color_mean = x.mean(axis=(1,2))
    color_std = x.std(axis=(1,2))
    horiz = np.abs(x[:, :, 1:] - x[:, :, :-1]).mean(axis=(1,2))
    vert = np.abs(x[:, 1:, :] - x[:, :-1, :]).mean(axis=(1,2))
    return np.concatenate([low, color_mean, color_std, horiz, vert], axis=1)


def run(dataset_dir=ROOT/'data', seed=0, top_k=25):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import log_loss, accuracy_score
    pool = load_noisy_pool('CIFAR', dataset_dir)
    train, valid = split_noisy_pool(len(pool.noisy_labels), seed)
    X_train = image_features(pool.images[train]); X_valid = image_features(pool.images[valid])
    scaler=StandardScaler().fit(X_train)
    x=scaler.transform(X_train); y=scaler.transform(X_valid)
    clf=LogisticRegression(C=0.01,max_iter=200,random_state=seed)
    clf.fit(x,pool.noisy_labels[train]); logits=clf.decision_function(y)
    noisy_y=pool.noisy_labels[valid]
    temp=calibrate_temperature(logits,noisy_y)
    probs=_softmax(logits/temp)
    hist=np.bincount(noisy_y,minlength=3)/len(noisy_y)
    baseline=float(-(hist*np.log(hist)).sum()); ce=float(log_loss(noisy_y,probs,labels=[0,1,2]))
    T,anchor=estimate_transition_from_probabilities(probs,top_k=top_k)
    return {'dataset':'CIFAR','seed':seed,'method':'downsample_color_logistic_diagnostic','status':'diagnostic_not_validated_transition',
     'information_gain_nats':baseline-ce, 'validation_noisy_cross_entropy':ce,'constant_prior_ce':baseline,
     'validation_noisy_accuracy':float(accuracy_score(noisy_y,probs.argmax(axis=1))),
     'temperature':temp,'estimated_transition_matrix':T.tolist(),'anchor_selection':anchor,
     'data_access':'Xtr/Str only; no Xts/Yts','feature_count':int(X_train.shape[1]),
     'warning':'High noisy-label accuracy does not establish true class identifiability or transition estimation accuracy.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',type=Path,default=ROOT/'data');p.add_argument('--seed',type=int,default=0);p.add_argument('--output',type=Path)
    a=p.parse_args();result=run(a.data_dir,a.seed);print(json.dumps(result,indent=2))
    if a.output:
      a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
