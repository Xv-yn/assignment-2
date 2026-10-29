# Integration and cleanup — 29 October 2026

The merge includes `ced6eab` from main and `ce19d1a` from Hao Lu's transition-estimation branch. Both histories are retained. Main's removal of the long README is respected: the root README is a short index and the complete guide lives in [RUNNING.md](RUNNING.md).

## Authoritative implementations

| Responsibility | Implementation | Compatibility or research extensions |
| --- | --- | --- |
| CLI, common configuration, audits | `algorithm/run_experiments.py` | `run_pilots.py` forwards to the same implementation and retains CPU pilot restrictions |
| Pilot-specific configuration checks | `algorithm/pilot_config.py` | Imported by both entry points; no cyclic configuration reader |
| Forward probabilities and loss functions | `algorithm/losses.py` | `pilot_losses.py` re-exports the original pilot API; formal weighted GCE and stable unweighted pilot GCE retain their distinct formulas |
| Known Fashion transition matrices | `algorithm/transition.py` | Pilot and Hao anchor diagnostics derive/import the same constants |
| Formal training and checkpoint format | `training.py`, `experiments.py`, `models.py` | Existing model layers, final configuration, random splits, selection and checkpoint keys are retained |
| CPU pilot training | `train.py`, `pilot_models.py`, `splits.py` | Preserved: this CNN, shuffle policy, min-delta selection and checkpoint format differ from formal training |
| Calibrated anchor and empirical studies | `transition_estimation.py`, `transition_empirical_v3.py` | Separate noisy-only diagnostic protocols; not inserted into the archived formal classifier |
| CIFAR studies | `cifar_feature_baseline.py`, `cifar_forward_study.py`, `cifar_transition_v4.py`, `cifar_cnn_gpu_v4.py`, `cifar_gpu_v5.py` | Preserved independent feature/MLP/CNN/residual models; versioned names represent different experiments, not interchangeable duplicate files |

Two distinct training loops and CNNs are retained deliberately. Combining them by channel count would change the formal smoke model, optimization, early stopping and checkpoint semantics. The genuinely duplicated CLI/audit and Forward math have been consolidated instead.

## Correctness fixes found during review

- V4 caches validate data/config/source identity and a record checksum before reuse. Valid caches avoid recomputing HOG features.
- V5 checkpoints and completed results validate content hashes. Saving a final model precedes publishing completed status; a save failure cannot leave a reusable completed result.
- V5 smoke outputs have separate names so they cannot overwrite full GPU evidence.
- Restoring an already early-stopped V5 checkpoint proceeds to final audit without training extra epochs.
- Singular condition numbers are represented by JSON `null` with failed gates and explicit nonfinite counts; V4's all-run mean is not fabricated from only the finite subset.
- V5 summaries verify model/checkpoint manifests and common data/source/settings across seeds, including optional paired V4 baselines. Unverified legacy JSON cannot become new GPU evidence merely by declaring a completed status.
- Optional scientific imports are deferred until the relevant feature study runs. Core training and matrix utilities retain the original lightweight dependency set.
- The submission ZIP includes both requirements files and the documentation needed to run its optional studies.
- Feature metadata correctly identifies 204 dimensions.
- V3 reports derive signal/condition findings from their input records. They no longer inject hard-coded LightGBM scores, assert every CIFAR input is weak/ill-conditioned, or label a mixed per-seed selection as the first seed's method.

## Results and reproducibility

`experiment_results/final/` and its six ten-split result groups are unchanged. The completed 60 classifier/30 estimator jobs correspond to the previously recorded training version and exact local session archive. These cleanup changes do not imply the experiments were rerun on 29 October. Archived resumes require their original source and environment because source fingerprints include Python modules.

Hao Lu's aggregate handoff remains in `HaoLu_Transition_Summary.json` with attribution. Its 62.58%/58.17% CIFAR comparison uses a different experimental protocol from the formal 70.513%/64.440% table. Raw records for that handoff are not present in the branch; its statistics are retained as supplied, not relabeled as independently verified formal evidence.

Validation details are recorded in [MERGE_VERIFICATION.json](MERGE_VERIFICATION.json). No datasets, weights, private SSH information or local backups are added to Git by this merge. Existing remote branches are retained.
