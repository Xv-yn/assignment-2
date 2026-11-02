# Rerunning the formal grid with standard GCE

**Why:** the 28 October results used an importance-weighted GCE extension with
a 3-epoch forward-correction warm-up. Every CIFAR GCE run (and 5/10
FashionMNIST0.3, 7/10 FashionMNIST0.6) selected a checkpoint from the warm-up
epochs, so the "GCE" column was mostly a briefly trained forward model. The
`gce` method is now standard GCE (q = 0.7, no reweighting, no warm-up). It
never uses T and selects its checkpoint by noisy-validation GCE loss. Forward
correction and the estimator are unchanged, but the source hash changed, so
the **whole grid** must be rerun on one machine so that both columns come from
the same hardware and code. See [RUNNING.md](RUNNING.md), sections 3–4.

**Who/where:** one CUDA machine (the A100 took about 19 minutes for the full
grid last time). A 14-core CPU needs several hours.

## Steps

From a clean checkout of the commit that contains this file:

```bash
git pull
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt     # CUDA build of PyTorch for GPU runs
# Put CIFAR.npz, FashionMNIST0.3.npz, FashionMNIST0.6.npz directly in data/

python -m unittest discover -s algorithm/tests          # expect: OK (skipped=1)
python -m algorithm.run_experiments --config algorithm/configs/final.json --plan
python -m algorithm.run_experiments --config algorithm/configs/final.json --device cuda
python -m algorithm.summarize --results-dir results/final_standard_gce
```

If the run is interrupted, repeat the training command with `--resume` added
(same `--device`). Do not change code or configuration between the start and
the end of the grid.

## Sanity checks before publishing

```bash
# Expect 60 rows and selection_metric == noisy_gce for every gce row
python - <<'EOF'
import csv
rows = list(csv.DictReader(open("results/final_standard_gce/runs.csv")))
print(len(rows), "runs")
print({(r["method"], r["selection_metric"]) for r in rows})
for d in ("FashionMNIST0.3", "FashionMNIST0.6", "CIFAR"):
    eps = [int(r["best_epoch"]) for r in rows if r["dataset"] == d and r["method"] == "gce"]
    print(d, "gce selected epochs:", sorted(eps))
EOF
```

## Publish the compact evidence

Copy everything except weights, predictions and split arrays (as was done for
`experiment_results/final/`):

```bash
rsync -a --exclude 'best.pt' --exclude 'predictions.npz' --exclude 'split.npz' \
  results/final_standard_gce/ experiment_results/final_standard_gce/
```

Keep a private backup of the full `results/final_standard_gce/` directory
(weights and predictions), then commit `experiment_results/final_standard_gce/`
and update:

- `README.md` (root): link the new `summary.md`.
- `docs/RUNNING.md` section 9: replace "must be regenerated" with the new
  table and the hardware/software versions used.
- The report: take every number from `experiment_results/final_standard_gce/`.
  Do not mix in values from `experiment_results/final/`.
