# Initial experimental design

Status updated 28 October 2026: the shared CNN, GCE, forward correction with
supplied FashionMNIST matrices, saved 80/20 splits and noisy-validation pilots
are implemented. CIFAR estimation and final evaluation remain. See
[training_protocol.md](training_protocol.md) for current implemented choices.
Primary sources were first read on 22 October 2026.

## Methods and assumptions

Let Y be the latent clean ID, S the observed noisy ID, and
`T[i,j] = P(S=j | Y=i)`. Class order is always `[0,1,2]`. For a row vector p of
model clean-class probabilities, forward correction trains with
`-log((p @ T)[S])`. It explicitly models class-conditional noise. Patrini et al.'s
population result requires a nonsingular true T and a suitable loss; it does
not guarantee finite neural-network accuracy. Use the supplied FashionMNIST
matrices and training-partition estimates for CIFAR. Anchor extraction needs
near-pure examples, accurate noisy probabilities, and justified clean-row
alignment. Memorization and absent anchors can spoil estimates. This design
uses training-only extraction, a restriction relative to the paper's experiments.
Source: [Patrini et al., CVPR 2017, Sections 4.2–4.3 and 5.1](https://arxiv.org/html/1609.03683).

The second objective is standard, untruncated generalized cross-entropy:
`(1 - p[S]**gamma) / gamma`, `0 < gamma <= 1`. It is matrix-free, approaches CE
as gamma approaches zero, and becomes `1-p[S]` at gamma=1. Gamma trades easier
optimization against reduced influence from low-probability labels. Assumptions
behind its robustness analysis limit claims under arbitrary class-conditional
noise. The original paper also proposes truncation; that variant is excluded
from the initial design. Source: [Zhang and Sabuncu, NeurIPS 2018, Sections 3.2–3.3](https://arxiv.org/html/1805.07836).

These are distinct robust methods because they change the objective and noise
treatment. Both will share a small CNN backbone and the same transforms,
splits, and comparable optimization budget within each dataset. An ordinary
CE classifier can be a diagnostic baseline, but does not replace either method.
Avoid ensembles and extra method variants until the required pipeline works.
GCE was not found in the supplied W2–W11 materials; forward correction and anchor
estimation are explicitly taught. See [course_coverage.md](course_coverage.md)
for the evidence and scope of this conclusion.

Supplied matrices, with clean rows and observed columns:

```text
FashionMNIST0.3: [[0.7, 0.3, 0.0],
                 [0.0, 0.7, 0.3],
                 [0.3, 0.0, 0.7]]
FashionMNIST0.6: [[0.4, 0.3, 0.3],
                 [0.3, 0.4, 0.3],
                 [0.3, 0.3, 0.4]]
CIFAR: unknown; no estimate exists yet.
```

## Questions and hypotheses

| Question | Initial hypothesis, not a result | Planned evidence |
| --- | --- | --- |
| Which method has better mean accuracy and lower variability across noise settings? | Known-T correction may help under asymmetric corruption; the ranking can depend on optimization and noise severity. | Paired ten-seed clean-test means and sample SDs for all six combinations |
| How reliable is estimation where true T is known? | Weak noisy posteriors and high corruption may worsen anchor estimates. | Freeze extraction rule, then measure entrywise MAE and row diagnostics on known-T datasets |
| How does estimated T affect classification? | Estimation errors may reduce the advantage of forward correction. | Controlled known-vs-estimated-T development comparisons; per-run CIFAR matrices and diagnostics |
| What are the practical trade-offs? | Matrix-free GCE avoids the separate CIFAR estimator fit, while gamma can affect convergence. | Measured pilot runtime, epoch counts, estimator overhead, and implementation complexity |

No experiment conclusions can be drawn from the image audit. All final claims
must replace these hypotheses with measured evidence.

## Implementation status and final-run contracts

The following original final-run specification remains provisional. Actual pilot
behavior and artifacts are documented in `training_protocol.md`; splitting and
pilot checkpoint/history support now exist, while final evaluation and matrix
estimation remain deferred.

- JSON configs use schema version 1; paths resolve from the repository root.
  `smoke.json` is a data-only check. `final.json` is explicitly a draft grid.
- Final split seeds are proposed as 0–9; separate initialization seeds are
  1000–1009. Each method will reuse the same saved split for a dataset/seed.
  The Part 6 split generator is implemented in `algorithm/splits.py`.
- Planned run ID: `<dataset>/<method>/split_<seed>`. Save indices, config,
  SHA-256 provenance, transform specification, selected checkpoint/history,
  matrix source/hash, runtime, status, and eventual test predictions/metrics.
  This training schema is a specification; smoke uses its own separate status.
- Source/config hashes exist for data smoke and training pilots. Neither type
  counts as a completed final classifier evaluation.
- Checkpoints, matrices, and final tables will live below `results/final/`.
  Pilot validation rules are fixed in the protocol; freeze final choices before test evaluation.
- Transform: float32 NCHW divided by 255; int64 labels; no resizing,
  standardization, augmentation, or label reconstruction. No fitted statistics
  exist; record this explicitly instead of fitting on the combined noisy pool.
- Same CNN within each dataset for both methods, always three output logits.
  The architecture and optimizer in `final.json` are candidate pilot settings;
  `algorithm/models.py` implements the current shared pilot CNN.
