# COMP5328 Assignment 2

Noise-robust image classification and transition-matrix estimation using the supplied FashionMNIST and CIFAR data.

- [Installation, commands and methods](docs/RUNNING.md)
- [Formal results: 60 classifiers and 30 estimators](experiment_results/final/summary.md)
- [Hao Lu's additional estimator and CIFAR studies](docs/HaoLu_Transition_README.md)
- [Merge decisions and code ownership](docs/INTEGRATION.md)

Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m algorithm.run_experiments --config algorithm/configs/final.json --data-dir data
```

For optional HOG/feature studies and their full test suite, install `requirements_additional.txt`. These studies have their own settings and results; they do not replace the archived formal experiment.
