"""Summarize v5 GPU CIFAR noisy-only diagnostic; never claim unknown true-T accuracy."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .cifar_gpu_v5 import _verify_completed
from .cifar_transition_v4 import _load_cached_seed
from .provenance import config_hash, write_json
from .transition import validate_transition


def _verified_gpu_record(path):
    record=json.loads(path.read_text(encoding='utf-8'))
    if record.get('status') != 'completed_full_gpu_diagnostic':
        return None
    identity=record.get('identity')
    settings=record.get('study_config')
    required={'schema_version','study','data_sha256','source_sha256','config_sha256','shared_config_sha256'}
    if not isinstance(identity,dict) or not required <= identity.keys() or not isinstance(settings,dict):
        raise ValueError(f'Unverified legacy GPU result: {path}; rerun into a new output directory')
    seed=record.get('seed')
    if type(seed) is not int or seed < 0 or path.stem != f'seed{seed}' or settings.get('seed') != seed:
        raise ValueError(f'GPU result seed/path mismatch: {path}')
    if (identity['study'] != 'cifar_gpu_v5' or identity['schema_version'] != 1
            or identity['config_sha256'] != config_hash(settings)
            or identity['shared_config_sha256'] != config_hash({k:v for k,v in settings.items() if k != 'seed'})
            or record.get('data_sha256') != identity['data_sha256']
            or settings.get('smoke') is not False or not str(settings.get('device','')).startswith('cuda')
            or record.get('device') != settings['device'] or record.get('epoch_budget') != settings.get('epochs')):
        raise ValueError(f'GPU result provenance/config mismatch: {path}')
    _verify_completed(record,identity,path.parent/f'seed{seed}_model.pt',path.parent/f'seed{seed}_checkpoint.pt')
    if (not np.isfinite(record['noisy_audit_accuracy']) or not 0 <= record['noisy_audit_accuracy'] <= 1
            or not np.isfinite(record['noisy_audit_information_gain_nats'])):
        raise ValueError(f'Nonfinite or invalid audit metrics: {path}')
    condition=record['estimated_T_condition_number']
    if condition is not None and (not np.isfinite(condition) or condition <= 0):
        raise ValueError(f'Invalid condition number: {path}; use null for nonfinite conditions')
    if record.get('estimated_T') is not None:
        validate_transition(record['estimated_T'])
    if record['quality_gate']['eligible_for_downstream_confirmation'] and (condition is None or record.get('estimated_T') is None):
        raise ValueError(f'Invalid eligibility for a missing/singular estimate: {path}')
    return record


def summarize(v5_dir, baseline_dir=None):
    folder=Path(v5_dir)
    rows=[]
    records=[]
    fingerprint=baseline_fingerprint=None
    for p in sorted(folder.glob('seed*.json')):
        if '_progress' in p.stem or not p.stem[4:].isdigit():continue
        record=_verified_gpu_record(p)
        if record is None:continue
        identity=record['identity']
        current=(identity['data_sha256'],identity['source_sha256'],identity['shared_config_sha256'])
        if fingerprint is not None and current != fingerprint:
            raise ValueError('Cannot combine GPU study results with different data/source/config identities')
        fingerprint=current
        s=int(record['seed']); gate=record['quality_gate']
        row={'seed':s,'noisy_accuracy':record['noisy_audit_accuracy'],
             'information_gain_nats':record['noisy_audit_information_gain_nats'],
             'condition_number':record['estimated_T_condition_number'],
             'eligible':gate['eligible_for_downstream_confirmation'],
             'pseudo_counts':record['pseudo_counts']}
        if baseline_dir is not None:
            base_file=Path(baseline_dir)/f'seed{s}.json'
            if base_file.is_file():
                old=json.loads(base_file.read_text(encoding='utf-8'))
                baseline_identity=old.get('identity',{})
                if (baseline_identity.get('study') != 'cifar_transition_v4'
                        or baseline_identity.get('data_sha256') != identity['data_sha256']
                        or baseline_identity.get('config_sha256') != config_hash({'seed':s})):
                    raise ValueError(f'Unverified/mismatched V4 baseline: {base_file}')
                old=_load_cached_seed(base_file,baseline_identity)
                baseline_current=(baseline_identity['data_sha256'],baseline_identity['source_sha256'])
                if baseline_fingerprint is not None and baseline_current != baseline_fingerprint:
                    raise ValueError('Cannot combine V4 baselines from different data/source identities')
                baseline_fingerprint=baseline_current
                if 'models' in old and 'color_logistic' in old['models']:
                    m=old['models']['color_logistic']
                    row['baseline_color_information_gain_nats']=m['audit_information_gain_nats']
                    row['gain_minus_color']=row['information_gain_nats']-m['audit_information_gain_nats']
                    row['baseline_color_noisy_accuracy']=m['audit_noisy_accuracy']
        rows.append(row)
        records.append(record)
    if not rows:raise ValueError('No completed FULL CUDA records. CPU smoke is not official GPU evidence.')
    values=np.asarray([r['noisy_accuracy'] for r in rows]); gain=np.asarray([r['information_gain_nats'] for r in rows])
    eligible=[record for record in records if record['quality_gate']['eligible_for_downstream_confirmation']]
    def stats(x):return {'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)) if len(x)>1 else None}
    out={'scope':'CIFAR noisy-label validation ONLY; unknown true transition matrix not verified',
         'seeds':[r['seed'] for r in rows], 'count':len(rows),'accuracy':stats(values),
         'information_gain_nats':stats(gain),'eligible_count':len(eligible),
         'eligible_seeds':[r['seed'] for r in eligible],'per_seed':rows,
         'verified_artifacts':True,'data_sha256':fingerprint[0],'source_sha256':fingerprint[1],
         'shared_config_sha256':fingerprint[2],
         'nonfinite_condition_count':sum(r['condition_number'] is None for r in rows),
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
        write_json(args.output,out)
if __name__=='__main__':main()
