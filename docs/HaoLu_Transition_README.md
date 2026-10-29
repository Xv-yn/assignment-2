# Hao Lu — Transition Matrix Estimation (COMP5328 Assignment 2)

Hao Lu's contribution from `hao-lu-transition-estimation` is integrated into the group repository. These calibrated-anchor, pseudo-confusion and CIFAR studies complement the archived formal experiment; they use different models and selection protocols and do not replace `final.json`.

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
- `requirements_additional.txt` includes the core requirements and adds scikit-learn, scikit-image and SciPy for feature/HOG studies. Pure matrix helpers can be imported without these optional packages.
- `docs/HaoLu_Transition_Summary.json` preserves the author's aggregate handoff results. Raw experiment records were not included in this branch, so the merge does not independently reverify those historical scores.

From the **group repository root**:

```bash
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
- That 58.17%/62.58% comparison is a separate study, not the group's [formal CIFAR result](../experiment_results/final/summary.md), whose Forward mean is 70.513%. The matrix candidates and classifier protocols must not be combined into the same result table.
- Do not tune on clean test labels. V5 quality-gate failures must remain visible.

## Cache and completion integrity

V4 and V5 now bind cached records to data content, configuration and source hashes. V5 additionally verifies model/checkpoint checksums and writes completed status only after final model saving succeeds. Checkpoints have `*_checkpoint.pt.manifest.json` sidecars. CPU smoke files use a separate `seedN_smoke.json` name.

V5 summaries verify those artifacts and reject records from mixed data, source or shared settings. Early-stopped resumes do not add extra epochs. Singular estimates retain failed gates with a JSON `null` condition, rather than invalid infinity or a misleading finite-only average.

Historical caches lack this provenance and must be kept as historical evidence or rerun in a new output directory. They are not silently accepted as new completed runs. The default formal anchor estimator and its archived results are unchanged.

## Do NOT upload
- `datasets.zip`, `.npz` files, `.pt/.pth/.ckpt` checkpoints, entire experiment tarballs, intermediate results, unrelated classification/report files.
- Keep generated study outputs separate from `experiment_results/final/`; its original records are immutable evidence of the completed formal grid.

See [integration decisions](INTEGRATION.md) for shared code ownership and [running instructions](RUNNING.md) for the submission pipeline. The source ZIP includes these guides and the optional dependency file.
