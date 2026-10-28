# Hao Lu — Transition Matrix Estimation (COMP5328 Assignment 2)

This folder is **my contribution only**, not a full replacement for the group repository. Add these files to the matching paths inside `Xv-yn/assignment-2`.

## Core estimator code
- `algorithm/transition_estimation.py`: anchor-point baseline, known FashionMNIST transition-matrix diagnostics, 80/20 split and calibration helpers.
- `algorithm/transition_empirical_v3.py`: confidence-filtered / all-pseudo-label empirical confusion estimators; contains `empirical_pseudo_confusion()` and a Fashion benchmark entry point.
- `algorithm/summarize_transition_v3.py`: V3 result aggregation.
- `algorithm/cifar_gpu_v5.py`: residual CNN for CIFAR noisy-label transition estimation; training, calibration, audit and checkpoint logic.
- `algorithm/cifar_gpu_v5_summary.py`: V5 run summary.

## Supporting code required by imports / earlier comparisons
- `algorithm/cifar_feature_baseline.py`: CIFAR color-feature noisy-label baseline.
- `algorithm/cifar_forward_study.py`: shared MLP training helper used by V3 and diagnostic comparison. It is not a replacement for teammates' final classifier.
- `algorithm/cifar_transition_v4.py`: `split_cal_audit`, bootstrap and V4 diagnostics, imported by V5.

## Tests and reproducibility
- Four `algorithm/tests/test_*.py` files shipped in this archive.
- `requirements_additional.txt` supplements the team's `requirements.txt`.
- `docs/HaoLu_Transition_Summary.json` has compact verified aggregate results; it is not a substitute for raw experiment outputs.

From the **group repository root** after copying the files:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements_additional.txt
python -m unittest discover -s algorithm/tests -v
# After privately placing data/FashionMNIST0.3.npz etc. in data/
python -m algorithm.transition_empirical_v3 --help
python -m algorithm.cifar_gpu_v5 --help
python -m algorithm.cifar_gpu_v5_summary --help
```

## Findings and limitations
- Across ten seeds, FashionMNIST0.3 V3 top-50% pseudo-label estimate had MAE 0.0126, vs 0.0777 for prior anchor-25.
- For FashionMNIST0.6, V3 all-pseudo-label estimate had MAE 0.0148, vs 0.0196 for prior anchor-25.
- CIFAR V5 ten-seed noisy-label audit accuracy was 37.60% on average. The estimated matrix has **unknown true accuracy**, and the alternative V5 matrix did not improve the existing Forward classifier (58.17% vs 62.58% mean clean test accuracy in archived ten-split comparison).
- Do not tune on clean test labels. V5 quality-gate failures must remain visible.

## Do NOT upload
- `datasets.zip`, `.npz` files, `.pt/.pth/.ckpt` checkpoints, entire experiment tarballs, intermediate results, unrelated classification/report files.
- Do not overwrite group `algorithm/data.py`, `algorithm/run_experiments.py`, `requirements.txt`, or configs.

This is a contribution handoff. It is not a claim the complete group assignment is ready for submission.
