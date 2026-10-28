"""Summarize held-out-noisy-label transition estimation diagnostics.

Noise matrices of the Fashion datasets may be consulted *only* after producing
estimates, for validation and choosing the estimator family to test on CIFAR.
All CIFAR truth/labels from test are off-limits in this script.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from .data import ROOT

METHODS = ('anchor25','hard_all','hard_top50','hard_top20','hard_top10')
THRESHOLD_GAIN = 0.05
LIMIT_NLL_GAIN = 0.01
MAX_CONDITION = 20.0


def select_method_by_signal(information_gain_nats:float)->str:
    if not np.isfinite(information_gain_nats):
        raise ValueError('information gain must be finite')
    return 'hard_top50' if information_gain_nats >= THRESHOLD_GAIN else 'hard_all'


def summarize(input_dir: Path, output_dir: Path):
    report = ['# COMP5328 Transition Matrix Estimation V3 — 10 splits', '',
       '**Evaluation policy:** all estimators are trained on noisy 80% and use the noisy 20% holdout; '
       'test labels or test images are never loaded. Known true Fashion matrices are used only to '
       'evaluate the resulting estimates. **Do not claim that CIFAR T_hat is accurate without known true T.**', '',
       '## Known-transition validation (mean ± sample SD of elementwise MAE)', '',
       '| Dataset | Original anchor 25 | Pseudo-confusion all | Pseudo-confusion top 50% | Chosen by noisy-label signal |',
       '|---|---:|---:|---:|---|']
    output={}
    for d in ('FashionMNIST0.3','FashionMNIST0.6','CIFAR'):
        records=[]
        for s in range(10):
            f=input_dir/d/f'seed{s}.json'
            if not f.exists():raise FileNotFoundError(f)
            v=json.loads(f.read_text())
            if v['dataset']!=d or v['seed']!=s:raise ValueError('mismatching run metadata')
            records.append(v)
        information_gain = np.array([r['information_gain_nats'] for r in records]);
        estimated={m:np.array([r['estimators'][m]['T_hat'] for r in records],np.float64) for m in METHODS}
        selected_methods=[select_method_by_signal(x) for x in information_gain]
        selected=np.array([estimated[m][i] for i,m in enumerate(selected_methods)])
        conds=np.array([np.linalg.cond(t) for t in selected])
        out={'n_splits':10,'seeds':list(range(10)),
          'selection_criterion':'hard_top50 if noisy NLL information gain >=0.05 nats; otherwise hard_all',
          'selected_per_seed':selected_methods,
          'mean_noisy_validation_information_gain_nats':float(information_gain.mean()),
          'sample_std_noisy_validation_information_gain_nats':float(information_gain.std(ddof=1)),
          'mean_noisy_validation_accuracy':float(np.mean([r['noisy_validation_accuracy'] for r in records])),
          'selected_T_mean':selected.mean(axis=0).tolist(),
          'selected_T_sample_std':selected.std(axis=0,ddof=1).tolist(),
          'selected_T_condition_number_mean':float(conds.mean()),
          'selected_T_condition_number_range':[float(conds.min()),float(conds.max())],
          'low_information_gain_splits':int((information_gain<LIMIT_NLL_GAIN).sum()),
          'ill_conditioned_splits':int((conds>MAX_CONDITION).sum()),
          'data_access':'Noisy Xtr/Str only, no test inputs; true Fashion T used only post-estimation.',
          'warning':'Pseudo clean labels can be systematically wrong; matrix identifiability not guaranteed.'}
        if d!='CIFAR':
            mae={m:np.array([r['estimators'][m]['validation_known_T']['mae'] for r in records]) for m in METHODS}
            chosen_mae=np.array([mae[m][i] for i,m in enumerate(selected_methods)])
            out['known_matrix_validation']={m:{'mae_mean':float(v.mean()),'mae_sample_sd':float(v.std(ddof=1))} for m,v in mae.items()}
            out['chosen_mae_mean']=float(chosen_mae.mean())
            out['chosen_mae_sample_sd']=float(chosen_mae.std(ddof=1))
            report.append(f'| {d} | {mae["anchor25"].mean():.4f} ± {mae["anchor25"].std(ddof=1):.4f} | '
                          f'{mae["hard_all"].mean():.4f} ± {mae["hard_all"].std(ddof=1):.4f} | '
                          f'{mae["hard_top50"].mean():.4f} ± {mae["hard_top50"].std(ddof=1):.4f} | '
                          f'{selected_methods[0]} ({chosen_mae.mean():.4f} ± {chosen_mae.std(ddof=1):.4f}) |')
        else:
            out['estimated_T_is_ground_truth']=False
            out['validated_for_forward_correction']=False
            out['reason_not_validated']='No clean CIFAR transition ground truth; weak noisy-label information gain; estimated matrix ill-conditioned.'
        output[d]=out
    report += ['', '## CIFAR: diagnostic only, not a validated transition matrix', '',
            f'Mean noisy-label information gain: {output["CIFAR"]["mean_noisy_validation_information_gain_nats"]:.4f} nats; '
            f'validation noisy-label accuracy: {output["CIFAR"]["mean_noisy_validation_accuracy"]:.2%}.', '',
            f'Proposed pseudo-confusion estimator condition number: '
            f'{output["CIFAR"]["selected_T_condition_number_mean"]:.1f} on average '
            f'({output["CIFAR"]["selected_T_condition_number_range"][0]:.1f}–'
            f'{output["CIFAR"]["selected_T_condition_number_range"][1]:.1f}).', '',
            'Provisional average CIFAR estimate (ROWS=latent Y, COLS=noisy S):', '', '```text',
            *[' '.join(f'{v:.4f}' for v in row) for row in output['CIFAR']['selected_T_mean']], '```', '',
            'The comparison with the earlier calibrated top-k estimator does **not** prove the new matrix '
            'is closer to the unknown truth. A near-singular T can amplify gradient instability in Forward '
            'Correction; do not silently substitute it into the final classifier. Stronger predictors, '
            'identifiability assumptions and held-out-only model selection are needed.', '',
            '## Additional predictor check (CIFAR, seeds 0 and 1)', '',
            'LightGBM (fixed compact 201-d spatial/color features) had noisy-label information gains '
            'of approximately 0.0044 and 0.0052 nats for a regularized 7-leaf model; this did not '
            'materially improve the very weak CIFAR signal. These are exploratory diagnostics, not '
            'pre-registered model-selection results.', '',
            '## Selection and caveats', '',
            '- The signal threshold 0.05 nats distinguishes the two known noise regimes here; it is '
            'not a universally justified rule and should not be presented as a proven noise classifier.',
            '- Selecting this rule by checking provided Fashion T can overfit these two known datasets. '
            'Treat the above as internal validation, not independent proof.',
            '- All matrices are row-stochastic. The empirical method conditions on pseudo-labels, '
            'not on actual unknown clean labels, so pseudo-label errors induce systematic bias.',
            '- CIFAR clean test labels have not been used to choose this estimator. The prior CIFAR '
            'classifier results in the team report use the *old* top-25 matrix, not this diagnostic.',
            '- The independent 80/20 split changes between seeds; a shared holdout is used for '
            'temperature calibration and pseudo-label extraction within each run.']
    output_dir.mkdir(parents=True,exist_ok=True)
    (output_dir/'transition_v3_summary.json').write_text(json.dumps(output,indent=2)+'\n')
    (output_dir/'TRANSITION_V3_REPORT.md').write_text('\n'.join(report)+'\n')
    return output


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-dir',type=Path,default=ROOT/'results/transition_v3');p.add_argument('--output-dir',type=Path,default=ROOT/'results/transition_v3/summary')
    a=p.parse_args();r=summarize(a.input_dir,a.output_dir)
    for dataset,stat in r.items():
        print(dataset, 'MAE' if 'chosen_mae_mean' in stat else 'CIFAR unvalidated',stat.get('chosen_mae_mean'), 'mean T=',np.round(stat['selected_T_mean'],3).tolist())
if __name__=='__main__':main()
