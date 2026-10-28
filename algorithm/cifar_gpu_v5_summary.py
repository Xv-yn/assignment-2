"""Summarize v5 GPU CIFAR noisy-only diagnostic; never claim unknown true-T accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def summarize(v5_dir, baseline_dir=None):
    folder=Path(v5_dir)
    rows=[]
    for p in sorted(folder.glob('seed*.json')):
        if '_progress' in p.stem or not p.stem[4:].isdigit():continue
        record=json.loads(p.read_text())
        if record.get('status')!='completed_full_gpu_diagnostic':continue
        s=int(record['seed']); gate=record['quality_gate']
        row={'seed':s,'noisy_accuracy':record['noisy_audit_accuracy'],
             'information_gain_nats':record['noisy_audit_information_gain_nats'],
             'condition_number':record['estimated_T_condition_number'],
             'eligible':gate['eligible_for_downstream_confirmation'],
             'pseudo_counts':record['pseudo_counts']}
        if baseline_dir is not None:
            base_file=Path(baseline_dir)/f'seed{s}.json'
            if base_file.is_file():
                old=json.loads(base_file.read_text())
                if 'models' in old and 'color_logistic' in old['models']:
                    m=old['models']['color_logistic']
                    row['baseline_color_information_gain_nats']=m['audit_information_gain_nats']
                    row['gain_minus_color']=row['information_gain_nats']-m['audit_information_gain_nats']
                    row['baseline_color_noisy_accuracy']=m['audit_noisy_accuracy']
        rows.append(row)
    if not rows:raise ValueError('No completed FULL CUDA records. CPU smoke is not official GPU evidence.')
    records=[json.loads((folder/f'seed{row["seed"]}.json').read_text()) for row in rows]
    values=np.asarray([r['noisy_accuracy'] for r in rows]); gain=np.asarray([r['information_gain_nats'] for r in rows])
    eligible=[record for record in records if record['quality_gate']['eligible_for_downstream_confirmation']]
    def stats(x):return {'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None}
    out={'scope':'CIFAR noisy-label validation ONLY; unknown true transition matrix not verified',
         'seeds':[r['seed'] for r in rows], 'count':len(rows),'accuracy':stats(values),
         'information_gain_nats':stats(gain),'eligible_count':len(eligible),
         'eligible_seeds':[r['seed'] for r in eligible],'per_seed':rows,
         'note':'No test labels were used. Eligibility does not certify T; downstream Forward-vs-baseline confirmation needed.'}
    if eligible:
        out['mean_candidate_T_from_eligible_runs']=np.mean([r['estimated_T'] for r in eligible],axis=0).tolist()
    improvements=[r['gain_minus_color'] for r in rows if 'gain_minus_color' in r]
    if improvements:out['paired_gain_minus_V4_color_baseline']=stats(np.asarray(improvements))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--v5-dir',type=Path,default=Path('results/cifar_gpu_v5'))
    p.add_argument('--v4-dir',type=Path,default=None)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args();out=summarize(args.v5_dir,args.v4_dir)
    print(json.dumps(out,indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
