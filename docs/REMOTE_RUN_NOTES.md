# Remote training notes

The formal experiment completed on 28 October 2026. All 60 classifier runs and 30 transition-estimator runs finished, and the complete grid passed aggregation checks. Training and aggregation took 1,121.135 seconds (approximately 18 minutes 41 seconds). The complete session was downloaded and verified locally before preparing the compact publication evidence.

## Isolation and file handling

The run used a newly created directory dedicated to this assignment session, with an ownership marker identifying the session. Uploaded source, the three supplied datasets, operational scripts, logs, checkpoints and experiment results were kept inside that directory. Existing server projects were outside the run directory.

After training, the complete result grid was verified and the session archive was downloaded. All 556 files in the downloaded manifest passed their SHA-256 checks, and the downloaded source and datasets matched their local originals. The verified extracted session is retained at `dist/server_session_verified/`. The full backup is `dist/COMP5328_A2_server_backup.tar.gz`, with SHA-256:

```text
4a93a5cd524ff66c9b92f145fbbbde54047dc099000e099ea162b09b37cb904d
```

Compact publishable evidence was prepared in `experiment_results/final/` (254 files). It excludes datasets, checkpoints, predictions, SSH information and uploaded payloads, while preserving every run's provenance and original artifact hashes. Full checkpoints, prediction arrays and operational logs remain available in the local backup.

After local verification and package checks, the dedicated session directory was deleted and confirmed absent. Cleanup checked its exact resolved path, ownership marker, completion state and backup hash. Existing projects and shared environments were not modified. The local verification and cleanup receipt is retained in [VERIFICATION.json](VERIFICATION.json).

## Experiment grid

The completed formal configuration contained 90 training tasks:

- 60 classifier runs: three datasets, two methods and ten distinct random 80/20 training/validation splits.
- 30 transition estimator runs: one per dataset and split, shared by the two classifiers on that split.

Each classifier trained for up to 30 epochs and each estimator for up to 20 epochs, with noisy-validation checkpoint selection and early stopping. The known FashionMNIST matrices were used for classifier training; estimates on those datasets measure estimator error. CIFAR classifiers used their own split's estimated matrix.

Clean test labels were accessed after checkpoint selection for final evaluation. The earlier CPU smoke runs were pipeline checks and are not formal experimental evidence.

## Final results

Clean test top-1 accuracy is reported as mean and sample standard deviation over ten random splits; values are percentages. The `gce` method is the documented importance-weighted GCE extension, including Forward warmup.

| Dataset | Method | Runs | Mean (%) | Sample SD (%) |
| --- | --- | ---: | ---: | ---: |
| FashionMNIST0.3 | Forward | 10 | 99.063 | 0.136 |
| FashionMNIST0.3 | Importance-weighted GCE | 10 | 98.997 | 0.119 |
| FashionMNIST0.6 | Forward | 10 | 93.087 | 1.548 |
| FashionMNIST0.6 | Importance-weighted GCE | 10 | 92.137 | 2.240 |
| CIFAR | Forward | 10 | 70.513 | 6.684 |
| CIFAR | Importance-weighted GCE | 10 | 64.440 | 6.714 |

The mean CIFAR transition estimate over the ten independently fitted estimators is:

```text
[[0.43552 0.31118 0.25330]
 [0.24515 0.44290 0.31196]
 [0.35662 0.22966 0.41372]]
```

The matrix convention is `T[i,j] = P(S=j | Y=i)`. Classifiers used their own split's estimate, rather than the matrix averaged across splits. The mean entrywise estimation MAE was 0.042624 on FashionMNIST0.3 and 0.017894 on FashionMNIST0.6; the corresponding mean Frobenius errors were 0.167779 and 0.068545. Per-entry matrix sample standard deviations and full-precision scores are retained in [the formal summary](../experiment_results/final/summary.json).

## Verified server environment

| Component | Version or hardware |
| --- | --- |
| GPU | NVIDIA A100-SXM4, 80 GB |
| Python | 3.12.3 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime reported by PyTorch | 12.8 |
| NumPy | 2.1.2 |
| Pillow | 11.0.0 |

All 38 unit tests passed on this environment before formal training. The preflight run also completed one full training-and-validation epoch for each dataset with the formal network configuration. The recorded duration was approximately 0.8–1.3 seconds per epoch; this measurement is a runtime estimate, not an accuracy experiment.

## CUDA determinism compatibility

Strict deterministic algorithms are enabled. On the verified server, native `AdaptiveAvgPool2d((3, 3))` backward raises an error under that setting because its CUDA backward implementation is nondeterministic.

The network therefore implements the same 3×3 adaptive averaging regions using explicit slices, means and concatenation. This preserves the intended pooling operation while allowing deterministic CUDA backward. A regression test compares output values and input gradients with native adaptive average pooling for the spatial sizes used by the networks. The full GPU preflight confirmed that the revised network can train under strict deterministic mode.

## Source and provenance

Python source, requirements and the formal configuration were frozen for the experiment. Each completed run records its configuration, source, dataset and environment identity, along with artifact hashes. The server launcher executed the grid sequentially; the two classifiers in each dataset/split pair shared an estimator.

The final result tables were generated after the full grid passed the aggregation checks. Means, sample standard deviations and estimated transition matrices come from the formal records, rather than the smoke configuration or preflight timing run. The publication preparation checked all original run artifact hashes, split and transition provenance, frozen configuration consistency, and the summary statistics against the per-run metrics.

The formal training implementation is preserved in commit `c57a81792756d1754ca2dda47c6c9a10af2c30e5` and in the full session archive. Remote changes made during training were subsequently merged by retaining their CPU pilot as separate modules and adding configuration routing. The formal model, loss, estimator, training loop and final configuration were preserved. All 51 combined formal/pilot tests passed after integration. Resuming archived runs requires the archived source and recorded environment because adding pilot modules changes the complete source fingerprint.

## Delivery files

- `dist/COMP5328_A2_code.zip`: submission source and an empty data directory.
- `dist/COMP5328_A2_results.zip`: all formal result records, model checkpoints, predictions and split arrays.
- `dist/COMP5328_A2_server_backup.tar.gz`: complete verified session, including the supplied data and uploaded source.

The compact evidence in the repository can be reviewed without downloading model weights. The full local archives preserve everything needed for inference and detailed verification. Six downloaded checkpoints (one dataset/method pair each) were loaded on the local CPU; all eight sampled predictions per checkpoint matched the class IDs saved on the server.
