# Noisy training and validation protocol

Implemented 28 October 2026. These are development pilots; the final experiment
design is not frozen and no clean-test accuracy is computed.

## Data and repeated holdouts

`load_noisy_pool()` reads only `Xtr` and `Str`. A local NumPy PCG64 generator
permutes the pool indices for each split seed, independently of model and
shuffle RNGs. The first 80% train the model; the remaining 20% validate it.
There is no stratification and no retraining on the full pool.

| Dataset | Training examples | Noisy validation examples |
| --- | ---: | ---: |
| FashionMNIST0.3 | 14,400 | 3,600 |
| FashionMNIST0.6 | 14,400 | 3,600 |
| CIFAR | 12,000 | 3,000 |

`python -m algorithm.splits` generates seeds 0–9 for all datasets. Actual
indices are saved in `results/splits/<dataset>/split_<seed>.npz`, with a JSON
sidecar containing dataset content hash, NumPy version, generator, seed, counts,
and archive hash. Existing files must match their metadata and regenerated
indices. Each split covers the entire pool exactly once, with no overlap.
The split CLI also checks that the ten training memberships differ.

Both methods use the same persisted dataset/seed split. The pilot initialization
seed is 1000 and the batch shuffle seed is 2000. The loaders shuffle training
only, keep validation order fixed, and retain the final incomplete batch.
`num_workers=0`, two CPU threads, seeded Python/NumPy/PyTorch, and deterministic
PyTorch operations are used. Cross-platform bitwise reproducibility is not
claimed. All transforms remain fixed float32 NCHW scaling by 255; no fitted
statistics or augmentation are used.

## Model and validation

The shared CNN has two 3x3 padded convolutions (32 and 64 channels), each followed
by ReLU and 2x2 max-pooling. Adaptive 4x4 average-pooling and a linear layer produce
three logits. It is randomly initialized; no pretrained data are used. Adam
uses learning rate 0.001 and weight decay 0.0001. These are pilot settings.

| Method | Training loss | Checkpoint selection | Noisy accuracy diagnostic |
| --- | --- | --- | --- |
| GCE, gamma=0.7 | `(1 - p[S]**gamma)/gamma` | Minimum held-out GCE against noisy labels | `argmax(p)` versus `Str` |
| Forward correction | `-log((p @ T)[S])` | Minimum held-out corrected noisy-label NLL | `argmax(p @ T)` versus `Str` |
| Optional CE baseline | `-log(p[S])` | Minimum held-out noisy-label CE | `argmax(p)` versus `Str` |

Using GCE for its own validation avoids introducing an unavailable CIFAR matrix
into the matrix-free classifier. This is a development selection rule, not an
unbiased estimate of clean risk. Noisy validation losses/accuracies need not rank
clean performance correctly under asymmetric or unknown noise. Reconsider and
freeze the final selection rule using noisy data only before final evaluation.
The course's symmetric-noise ranking result does not apply to arbitrary noise.

Validation uses `model.eval()` and inference mode, with no optimization. Losses
are averaged by example count, including the final short batch. The earliest
checkpoint improving loss by more than `min_delta` is retained. Patience counts
non-improving epochs, including ties. Default pilot configs run just two epochs,
so they establish execution and checkpoint behavior, not convergence.

Our matrix convention is `T[i,j]=P(S=j|Y=i)`, with rows summing to one and
observed probabilities `p @ T`. The supplied FashionMNIST arrays are used as
documented in the assignment design. The lecture notation instead uses clean
columns and column vectors: its corresponding matrix is our transpose. Unit
tests check asymmetric orientation and identity-T equivalence to ordinary CE.
Forward loss is calculated in log space. CIFAR forward training explicitly
fails until the transition estimator is implemented; no identity matrix is
substituted for the unknown matrix.

## Artifacts and remaining work

Each dataset/method/seed pilot saves `best.pt`, `history.json`, and `run.json`,
including the selected epoch, noisy metrics, all seeds, dataset/source/config/
split/environment identities, fixed preprocessing, transition source, class
counts, runtime, and artifact hashes. The checkpoint contains model weights and
configuration, but does not support resuming partway through an epoch.

`--resume` skips only verified completed pilots. Failed/interrupted pilots or
ones with damaged artifacts are restarted from initialization when identity
matches. Changed data/source/config/environment is rejected; use a fresh output
directory. A partially written split is rejected explicitly, never silently
replaced. Failed jobs have an explicit failed status.

Pilots always record `clean_test_accessed: false` and
`final_classifier_runs_completed: 0`. They are not included in final accuracy
summaries. Still required: CIFAR estimation and integration, noisy-only tuning,
final selection-rule freeze, clean-test evaluator, all 60 final evaluations,
accuracy aggregation, and the report. The full final config remains plan-only.

Implementation references: [PyTorch DataLoader](https://docs.pytorch.org/docs/2.14/data.html),
[PyTorch reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html),
and [Zhang and Sabuncu, GCE, Section 3.2](https://arxiv.org/html/1805.07836).

Verified pilot evidence and timings: [training_pilot_results.md](training_pilot_results.md).
