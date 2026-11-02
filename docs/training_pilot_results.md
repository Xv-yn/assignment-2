# Training integration verification — 28 October 2026

All five pilots used split seed 0, initialization seed 1000, shuffle seed 2000,
two epochs, batch size 128 and two CPU threads. Full 80% training partitions
were used. These are development checks, not final experiments or convergence
claims. Accuracy below compares predictions with **noisy validation labels**;
it is not clean-test accuracy, and the objectives differ across methods.

| Dataset | Method | Train / validate | Selected epoch | Noisy validation loss | Noisy validation accuracy | Training + validation seconds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| CIFAR | gce | 12,000 / 3,000 | 2 | 0.763656 | 35.37% | 20.29 |
| FashionMNIST0.3 | gce | 14,400 / 3,600 | 2 | 0.464436 | 66.69% | 17.19 |
| FashionMNIST0.6 | gce | 14,400 / 3,600 | 2 | 0.753741 | 40.72% | 17.62 |
| FashionMNIST0.3 | forward | 14,400 / 3,600 | 2 | 0.666294 | 67.72% | 17.43 |
| FashionMNIST0.6 | forward | 14,400 / 3,600 | 2 | 1.088975 | 40.53% | 17.77 |

Verified evidence:

- 24 unit/regression tests pass, including partition coverage/disjointness, independent RNGs, matrix orientation, GCE limits, gradients, validation without weight updates, early stopping, checkpoint selection and corrupt-artifact handling.
- All 30 saved splits cover their complete noisy pool exactly once. Ten distinct training memberships exist per dataset.
- Each best checkpoint was loaded with `weights_only=True`; rerunning its noisy validation partition reproduced its recorded loss and accuracy.
- The five pilot manifests match the current source hash and saved artifact hashes. Each records no clean-test access and zero final classifier evaluations.
- Both pilot resume commands verified and skipped their completed runs. The current data smoke passed all three datasets and its resume checks.
- Training, validation and split generation use only Xtr/Str. The separate data-integrity smoke audits all four supplied arrays without computing test performance.

Artifacts live under `results/pilot-gce/`, `results/pilot-forward/`,
`results/splits/`, and `results/data-integration-smoke/`. They are local, ignored
results; the source and documentation remain the reproducible deliverable.

Still required: CIFAR transition estimation and forward integration, final
noisy-only tuning/selection freeze, clean-test evaluator, all 60 final runs,
accuracy aggregation and the report. Runtime here is an initial CPU timing,
not a promise for longer experiments or other hardware.
