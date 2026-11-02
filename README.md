# COMP5328 Assignment 2

Noise-robust image classification and transition-matrix estimation using the supplied FashionMNIST and CIFAR data.

- [Installation, commands and methods](docs/RUNNING.md)
- Formal results: being regenerated with standard GCE, see [the rerun guide](docs/RERUN.md). The [28 October results](experiment_results/final/summary.md) used a GCE extension and are superseded.
- [Hao Lu's additional estimator and CIFAR studies](docs/HaoLu_Transition_README.md)
- [Merge decisions and code ownership](docs/INTEGRATION.md)

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m algorithm.run_experiments --config algorithm/configs/final.json
python -m algorithm.summarize --results-dir results/final_standard_gce
```

Put the three supplied `.npz` files directly in `data/` first. The full guide, including runtimes and the reproducibility caveat, is [docs/RUNNING.md](docs/RUNNING.md).

For optional HOG/feature studies and their full test suite, install `requirements_additional.txt`. These studies have their own settings and results; they do not replace the archived formal experiment.
