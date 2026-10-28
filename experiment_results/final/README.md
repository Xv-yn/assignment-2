# Formal experiment evidence

This directory contains the compact evidence for 60 classifier runs and 30
transition-estimator runs over the complete ten-split grid. All original run
artifact hashes were checked against the downloaded session before publication.

Start with `summary.md` for accuracy means, sample standard deviations and
transition estimates. `runs.csv` contains per-run scores. `final.json` is the
frozen configuration. `preflight.json`, `tests.log` and `execution_status.json`
record the GPU preflight, unit-test run and completed session status.

Each dataset/split directory contains classifier metrics, training histories and
learning curves, plus the estimator's matrix and history. `provenance.json`
preserves every run's source, data, configuration and environment identity,
original artifact hashes, split hashes and hashes of the published evidence.

The full checkpoints, prediction arrays, split arrays, supplied datasets and
complete operational logs are retained in the verified local session backup.
They are intentionally excluded from this compact public evidence directory.
The original `run.json` manifests are represented in `provenance.json`; this
compact directory is not an input to the training-resume or full-artifact
aggregation commands. To repeat full verification or inference, use the local
backup or reproduce the runs from source and the supplied datasets.

No server address, SSH key, SSH configuration or uploaded payload is included.
