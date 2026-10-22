"""Regenerate an audit table from verified data-smoke records (no accuracy)."""

import argparse
import json
from pathlib import Path

from .provenance import sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted(args.results_dir.glob("*/run.json"))
    if not paths:
        parser.error("No data-smoke records found; classifier aggregation is deferred to Part 11")
    rows = []
    for path in paths:
        run = json.loads(path.read_text())
        if run.get("status") != "completed_data_smoke":
            raise ValueError(f"Incomplete/failed data smoke: {path}")
        if set(run.get("artifacts", {})) != {"audit.json", "training_grid.png"}:
            raise ValueError(f"Missing artifact manifest: {path}")
        for name, expected in run["artifacts"].items():
            if sha256_file(path.parent / name) != expected:
                raise ValueError(f"Artifact hash mismatch: {path.parent / name}")
        audit = json.loads((path.parent / "audit.json").read_text())
        rows.append({"dataset": audit["dataset"], "noisy_counts": audit["noisy_pool_class_counts"],
                     "batch": audit["batch"], "dataset_sha256": audit["dataset_sha256"]})
    lines = ["# Data audit summary", "", "Integrity/preprocessing checks only. Classifier runs completed: 0.", "",
             "| Dataset | Noisy ID 0 | Noisy ID 1 | Noisy ID 2 | Batch shape |", "| --- | ---: | ---: | ---: | --- |"]
    for row in rows:
        c = row["noisy_counts"]
        lines.append(f"| {row['dataset']} | {c['0']} | {c['1']} | {c['2']} | {row['batch']['shape']} |")
    (args.results_dir / "summary.md").write_text("\n".join(lines) + "\n")
    write_json(args.results_dir / "summary.json", {"kind": "data_audit", "classifier_runs_completed": 0, "rows": rows})
    print("\n".join(lines))


if __name__ == "__main__":
    main()
