# COMP4328 / COMP5328 Assignment 2: learning with noisy labels

This project builds on the repository's data audit. It implements two
label-noise-robust classifiers, transition-matrix estimation, independent
random 80/20 splits, repeated experiments, result aggregation, prediction and
source packaging. Only the three supplied datasets are used; no extra data or
pretrained weights are downloaded.

## 1. Environment and data

Python 3.10 or newer is recommended. Verified environments:

| Use | Environment |
| --- | --- |
| Early CPU pipeline checks | Python 3.10.19, PyTorch 2.5.1 CPU, NumPy 2.0.1, Pillow 11.1.0 |
| Original GPU experiment (28 Oct, superseded, see section 9) | A100 80GB, Python 3.12.3, PyTorch 2.8.0+cu128 |
| Clean-ZIP reproduction check (3 Nov) | Fresh virtual environment, Python 3.14.3, PyTorch 2.15.0, NumPy 2.5.4, Pillow 12.3.0, 14-core CPU, no GPU |

Run every command from the repository (or extracted ZIP) root:

```bash
python -m pip install -r requirements.txt
```

CPU and CUDA are both supported. For a GPU build, follow the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/).
`--device cuda` fails explicitly when CUDA is unavailable; `auto` picks CUDA
when present. torchvision, scikit-learn and matplotlib are not needed for the
main pipeline.

After extracting the submission, place the three supplied files directly in
`data/`:

```text
data/FashionMNIST0.3.npz
data/FashionMNIST0.6.npz
data/CIFAR.npz
```

`data/` is the default location. If your datasets live elsewhere, add
`--data-dir <path>` to any command below.

Each archive must contain `Xtr, Str, Xts, Yts`. FashionMNIST `Xtr` has shape
`(18000, 784)` and CIFAR `Xtr` has shape `(15000, 32, 32, 3)`; each test set
has 3000 examples. The strict data validation and integer class IDs are kept;
class semantics are not inferred.

## 2. Quick checks

```bash
python -m unittest discover -s algorithm/tests -v
python -m algorithm.run_experiments --config algorithm/configs/smoke.json
python -m algorithm.summarize --results-dir results/smoke
python -m algorithm.run_experiments --config algorithm/configs/training_smoke.json
python -m algorithm.summarize --results-dir results/training_smoke --allow-incomplete
```

`smoke.json` only checks the data and writes a grid of noisy training examples.
`training_smoke.json` trains on 192 training and 48 validation examples per
dataset for one epoch with a small network and one seed. It runs both
classifiers on all three datasets and then exercises the evaluation path on
the full test set. Expect accuracies near 33% from this check. It is a program
check only and **must not be used for reported accuracy, matrix estimates or
robustness conclusions**. Debug results are always marked `debug=true`, and
summarizing them requires `--allow-incomplete`.

Measured on the 14-core CPU above: the unit tests take a few seconds and the
training smoke run takes about 10 seconds.

## 3. Formal experiment (60 classifier runs + 30 matrix estimates)

```bash
python -m algorithm.run_experiments --config algorithm/configs/final.json --plan
python -m algorithm.run_experiments --config algorithm/configs/final.json
python -m algorithm.summarize --results-dir results/final_standard_gce
```

Default configuration: 3 datasets x 2 classifiers x 10 different split seeds.
Each classifier trains for at most 30 epochs with patience 8. Training sets
contain 14,400 (FashionMNIST) or 12,000 (CIFAR) examples and validation sets
3,600 or 3,000. Every split is a random 80/20 split without replacement and
without stratification; splits for different seeds may overlap. On each split
the two classifiers share the split, initialization seed and batch order, so
they can be compared pairwise. The estimator has its own initialization and
trains for at most 20 epochs.

The network is a SmallCNN trained from scratch: channels 32/64/128, each block
has two 3x3 convolutions, BatchNorm, ReLU and max pooling, followed by 3x3
adaptive pooling and a fully connected head with dropout. The pooling is
implemented with explicit slices over the same averaging regions so that CUDA
backpropagation can run in strict deterministic mode; tests compare its outputs
and input gradients with native adaptive pooling. Optimization uses Adam,
initial learning rate 0.001, weight decay 0.0001 and a cosine learning-rate
schedule. Pixels are divided by 255 and converted to float32 NCHW; there is no
standardization or augmentation. Architecture and optimizer settings are
identical for both classifiers.

To run a single group of jobs:

```bash
python -m algorithm.run_experiments --config algorithm/configs/final.json --dataset CIFAR --method forward --seed 0 --device cpu
python -m algorithm.run_experiments --config algorithm/configs/final.json --resume
```

Use the same `--device` override on the first run if you want to resume it
later with the same configuration. `--resume` verifies the configuration,
source code, data, environment and artifact hashes and skips only verified
complete jobs. Incomplete jobs restart from scratch; they do not resume from an
intermediate epoch. After changing code, configuration, device or dependencies,
use a new `--results-dir`. Different subsets of jobs may be completed in the
same directory one after another, but never write to the same job directory
from two processes at once.

**Runtime.** On the A100 the original full grid took about 19 minutes. On the
14-core CPU above, one complete CIFAR split (estimator plus both classifiers)
took about 20 minutes, so the full grid on CPU takes several hours. A CUDA
machine is recommended for the full grid. Summarize only after every job has
finished, and never use test-set results to change hyperparameters.

**Reproducibility across hardware.** Runs are deterministic on a given device
and software stack, but CPU and GPU arithmetic differ slightly. On CIFAR the
best noisy-validation loss is barely below the chance value
(ln 3 = 1.0986), so these tiny differences can change the selected epoch and
move clean accuracy by many points. Example: CIFAR split 0 with the forward
method gave 75.3% on the A100 (epoch 9 selected) and 55.8% on CPU (epoch 4
selected), with nearly identical training curves. Compare results only within
one hardware/software setup, and report all numbers from a single run of the
grid.

## 4. Algorithms and leakage rules

Matrix orientation everywhere: `T[i,j] = P(S=j | Y=i)`, where Y is the clean
class and S the noisy label. If the network predicts the row vector
`p = P(Y|x)`, the noisy posterior is `p @ T`. Final predictions use
`argmax(p)`; T is not applied at clean test time.

### Forward correction

Objective: `L_forward = -log((softmax(logits) @ T)[s])`, implemented stably
with logsumexp. FashionMNIST0.3 and FashionMNIST0.6 use the matrices supplied
in the brief; CIFAR uses the matrix estimated independently on the same split.

Source: [Patrini et al., CVPR 2017, Making Deep Neural Networks Robust to Label Noise: A Loss Correction Approach](https://openaccess.thecvf.com/content_cvpr_2017/html/Patrini_Making_Deep_Neural_CVPR_2017_paper.html).

### Generalized cross-entropy (GCE)

The `gce` method is **standard, untruncated GCE** with q = 0.7:

```text
L_q(p, s) = (1 - p_s^q) / q
```

As q tends to 0 it approaches cross-entropy; at q = 1 it equals half the
multiclass L1/MAE loss. It reduces the gradient weight of examples whose
noisy label the model finds unlikely, which makes training more tolerant of
label noise.

GCE is **matrix-free**: it never reads T, either in training or in checkpoint
selection. This makes it a genuine contrast with forward correction, which
needs T. On CIFAR the GCE classifier therefore does not depend on the quality
of the estimated matrix.

Source: [Zhang & Sabuncu, NeurIPS 2018, Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels](https://arxiv.org/abs/1805.07836).

#### Optional extension (not used for the reported results)

`losses.gce_loss` also supports an importance-weighted GCE extension with a
forward-correction warm-up, enabled by setting `method_parameters.gce.reweight`
to `true` and/or `warmup_epochs` above 0. It uses T:

```text
w(x,s) = stop_gradient(p_s / max((p @ T)_s, 1e-8))
L = min(w, max_weight) * (1 - p_s^q) / q
```

The 28 October experiment used this extension with `max_weight = 10` and a
3-epoch warm-up. It was replaced because the checkpoint chosen for every CIFAR
run (and 5/10 FashionMNIST0.3 and 7/10 FashionMNIST0.6 runs) came from the
warm-up epochs, which train with the forward loss. That made the "GCE" column
largely a briefly trained forward model. Importance reweighting is also taught
in the course (W8), unlike GCE. If you run the extension, use a separate
results directory and describe it as an extension, not as standard GCE.

### Transition-matrix estimation

On the current 80% training partition, an auxiliary network is trained with
ordinary cross-entropy on the noisy labels. Its checkpoint is selected only by
cross-entropy on the 20% noisy validation partition. After restoring the best
checkpoint it predicts the noisy posterior `g(x) = P(S|x)` for the training
partition in its original order.

For each class i, the examples with the largest `g_i(x)` (top 1% by default)
are selected and their full posterior vectors are averaged to form row i of T.
`top_fraction = 0` selects a single argmax anchor per class; `quantile < 1`
excludes the most extreme-confidence examples (no exclusion by default). All
anchor indices are saved as original `Xtr` row numbers, and estimation never
reads `Xts`/`Yts`. Both classifiers on a split see the same split; only forward
correction uses the estimated matrix.

The method assumes that near-pure class anchors exist, that the posterior
estimate is good, and that clean and noisy class IDs are aligned. Under unknown
noise that is not diagonally dominant, the class permutation cannot be
identified from noisy data alone. The code records condition numbers and
diagonal-dominance diagnostics. It never permutes the matrix using test labels
and never forces a non-diagonally-dominant estimate into a known template.
With heavy noise (FashionMNIST0.6, and apparently CIFAR) the posterior
differences between classes are small and the estimate can be unstable; the
report should discuss this limitation using the formal results.

The estimator also runs on the two known-matrix datasets and records the
entrywise MAE and Frobenius error, but the classifiers there still use the
supplied true matrices. The formal CIFAR matrix must come from complete
training; debug matrices must not be used instead.

### Model selection and evaluation

Each epoch, every model is scored on its noisy validation partition, and that
score selects the best epoch and drives early stopping:

| Model | Selection loss (`selection_metric`) |
| --- | --- |
| Forward correction | Forward NLL `-log((p @ T)_s)` (`noisy_forward_nll`) |
| Standard GCE | Mean GCE loss of the noisy labels, no T (`noisy_gce`) |
| Estimator network | Cross-entropy on the noisy labels (`noisy_ce`) |

The noisy NLL of `p` is also logged for every model as a diagnostic
(`validation_noisy_nll`). The best model is restored before the clean test set
is loaded and top-1 accuracy computed. Training, early stopping,
hyperparameter choices and matrix estimation never access clean test labels.
The data audit may read the test arrays to check shapes and values, but it
does not train or select models.

## 5. Results and report data

```text
results/final_standard_gce/<dataset>/seed_000/
  split.npz                    # full 80/20 original row numbers
  estimator/
    best.pt, history.json
    transition.json            # matrix, anchors, diagnostics, known-matrix error
    run.json                   # run status and hash manifest
  forward/ (and gce/)
    best.pt, history.json      # history.json records selection_metric
    metrics.json               # accuracy, confusion matrix, best epoch, selection loss, time
    predictions.npz            # final clean-test posteriors and true labels
    run.json
```

The summary command verifies every artifact hash, run provenance and the
consistency of predictions with accuracy. It writes:

- `summary.md/json/csv`: mean and sample standard deviation (ddof=1) of
  accuracy per dataset and method.
- `runs.csv`: per-run results for report tables and further analysis.
- `transition_estimates` in `summary.json`: mean and standard deviation of the
  per-split matrices, plus estimation errors on the two known matrices.
- `paired_comparisons` in `summary.json`: mean and SD of the per-split
  accuracy difference GCE minus forward (percentage points).
- `accuracy_comparison.png`, `<dataset>_transition.png` and a
  `learning_curve.png` per run (red curve: the selection loss).

Each CIFAR run uses its own split's estimated matrix. The report may show the
mean and SD of the ten estimates, but that mean is not an input any run used.
A formal summary must contain the complete configured grid of at least 10
splits; debug or incomplete runs only produce a clearly labelled provisional
summary. For a single run the SD is null/N/A rather than a fabricated zero.

## 6. Standalone prediction

The input `.npy` is a raw uint8 batch with the same shape as the corresponding
dataset's original images:

```bash
python -m algorithm.predict --checkpoint results/final_standard_gce/CIFAR/seed_000/forward/best.pt --images images.npy --output predictions.npz
```

This saves clean class IDs (`predictions`) and `probabilities`. Only load
checkpoints you generated or trust; loading uses PyTorch `weights_only=True`.

## 7. Source submission package

```bash
python -m algorithm.package_submission --output dist/GroupIDXX_StudentIDxxxxxxxx_StudentIDxxxxxxxx_StudentIDxxxxxxxx.zip
```

Replace the example filename with the real group ID and all members' student
IDs. The archive contains `algorithm/`, the README, the run and method
documents from `docs/`, `requirements.txt`, `requirements_additional.txt` for
the optional studies, and an **empty** `data/`. It contains no data, results
directories, model weights, Git directory or caches. Build it from a clean
checkout so untracked local notes in `docs/` are not included. Name the report
PDF separately as the brief requires; the report appendix can cite this guide's
installation, formal training and summary commands.

## 8. File responsibilities

| Module | Responsibility |
| --- | --- |
| data.py / audit_data.py | Strict raw-data validation, deterministic preprocessing and audit |
| models.py / losses.py | Shared CNN, forward correction, GCE and the optional importance-weighted extension |
| transition.py | True matrices, anchor estimation and matrix constraint checks |
| training.py | Seeds, splits, training, validation selection and evaluation |
| experiments.py / run_experiments.py | Experiment scheduling, isolated test access, status records and resume |
| results.py / summarize.py | Integrity checks, means/standard deviations and plots |
| provenance.py | Atomic JSON writes and content hashes |
| predict.py / package_submission.py | Standalone prediction and source packaging |
| tests/ | Transform/label, test-leakage, matrix orientation, gradient, anchor, split, checkpoint and standard-GCE regression tests |

## 9. Results status

**The formal results must be regenerated with standard GCE.** Run the grid in
section 3 on a single CUDA machine (see [RERUN.md](RERUN.md)) and replace the
published evidence with the new summary.

The 28 October results in `experiment_results/final/` (training version
[c57a817](https://github.com/Xv-yn/assignment-2/commit/c57a81792756d1754ca2dda47c6c9a10af2c30e5))
used the importance-weighted GCE extension and are **superseded**. They are
kept only as an archive and must not be reported as standard GCE:

| Dataset | Forward | Importance-weighted GCE extension (superseded) |
| --- | ---: | ---: |
| FashionMNIST0.3 | 99.063 ± 0.136 | 98.997 ± 0.119 |
| FashionMNIST0.6 | 93.087 ± 1.548 | 92.137 ± 2.240 |
| CIFAR | 70.513 ± 6.684 | 64.440 ± 6.714 |

Model weights, prediction arrays, data and full run logs from that run are kept
in a teammate's local backup (`dist/COMP5328_A2_server_backup.tar.gz`), not in
Git. See [REMOTE_RUN_NOTES.md](REMOTE_RUN_NOTES.md).

## 10. Retained CPU pilot pipeline

The CPU pilot code added on the remote during training is kept. The shared
entry point chooses the pipeline from the configuration's `stage`:

```bash
python -m algorithm.run_experiments --config algorithm/configs/pilot.json
python -m algorithm.run_experiments --config algorithm/configs/pilot_forward.json
```

The pilots keep their own `pilot_models.py` and `train.py` and evaluate only
on the noisy validation set. `pilot_losses.py` re-exports the original API from
`losses.py` and keeps the original per-example unweighted GCE implementation.
`splits.py` and the original pilot tests are kept. Formal training uses
`models.py`, `losses.py`, `training.py` and `experiments.py`. Merge validation
records are in [MERGE_VERIFICATION.json](MERGE_VERIFICATION.json).

## 11. Additional estimator studies

The shared CLI and loss math have been consolidated; pilot entry points remain
compatible. Hao Lu's calibration, empirical pseudo-confusion and V4/V5 studies
are optional independent experiments, not replacements for `final.json`. See
[the study guide](HaoLu_Transition_README.md) and
[integration decisions](INTEGRATION.md). Install all study dependencies with
`python -m pip install -r requirements_additional.txt`. Existing caches without
data/code/artifact provenance must be regenerated into a new study directory.
