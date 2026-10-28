"""Verified classifier aggregation; incomplete/debug results are never final."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .provenance import sha256_file, write_json


def verified_record(path, status, required):
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != status:
        raise ValueError(f"Incomplete run: {path}")
    if set(record.get("artifacts", {})) != set(required):
        raise ValueError(f"Incomplete artifact manifest: {path}")
    for name, expected in record["artifacts"].items():
        if not (path.parent / name).is_file() or sha256_file(path.parent / name) != expected:
            raise ValueError(f"Artifact hash mismatch: {path.parent / name}")
    return record


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def curve_plot(history, output):
    rows = history["epochs"]
    canvas = Image.new("RGB", (760, 400), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 10), "Training objective (blue) / noisy validation NLL (red)", fill="black")
    ymax = max(max(r["train_loss"], r["validation_noisy_nll"]) for r in rows) * 1.05
    ymax = max(ymax, 0.01)
    draw.line((60, 40, 60, 345, 730, 345), fill="black", width=2)
    for i in range(6):
        y = 345 - 300 * i / 5
        draw.text((5, y - 5), f"{ymax * i / 5:.2f}", fill="black")
        draw.line((60, y, 730, y), fill="#dddddd")
    for key, color in (("train_loss", "#1261a0"), ("validation_noisy_nll", "#c03434")):
        points = [(60 + 665 * i / max(1, len(rows) - 1), 345 - 300 * r[key] / ymax) for i, r in enumerate(rows)]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
    draw.text((60, 365), f"Epoch 1 -> {len(rows)}; selected epoch {history['best_epoch']}", fill="black")
    canvas.save(output)


def matrix_plot(matrix, output, title):
    canvas = Image.new("RGB", (480, 390), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 15), title, fill="black")
    draw.text((100, 40), "Columns: observed noisy label S", fill="black")
    draw.text((20, 65), "Rows: clean label Y", fill="black")
    for i in range(3):
        draw.text((40, 130 + 85 * i), str(i), fill="black")
        draw.text((135 + 105 * i, 85), str(i), fill="black")
        for j in range(3):
            value = matrix[i][j]
            color = (int(245 - 170 * value), int(248 - 100 * value), 250)
            x, y = 85 + 105 * j, 110 + 85 * i
            draw.rectangle((x, y, x + 100, y + 80), fill=color, outline="white")
            draw.text((x + 25, y + 32), f"{value:.4f}", fill="black")
    canvas.save(output)


def accuracy_plot(rows, output, provisional):
    canvas = Image.new("RGB", (900, 470), "white")
    draw = ImageDraw.Draw(canvas)
    title = "DEBUG / PARTIAL: " if provisional else ""
    draw.text((20, 15), title + "Clean test top-1 accuracy: mean +/- sample SD", fill="black")
    for tick in range(0, 101, 20):
        y = 380 - tick * 3
        draw.line((65, y, 875, y), fill="#dddddd")
        draw.text((20, y - 5), f"{tick}%", fill="black")
    draw.line((65, 70, 65, 380, 875, 380), fill="black", width=2)
    width = 800 / max(1, len(rows))
    for index, row in enumerate(rows):
        x = 75 + (index + .5) * width
        mean, sd = row["mean_top1_percent"], row["std_top1_percent"]
        y = 380 - mean * 3
        draw.rectangle((x - width * .23, y, x + width * .23, 380), fill="#4477aa" if row["method"] == "forward" else "#ee9966")
        if sd is not None:
            top, bottom = 380 - min(100, mean + sd) * 3, 380 - max(0, mean - sd) * 3
            draw.line((x, top, x, bottom), fill="black", width=2)
            for yy in (top, bottom):
                draw.line((x - 7, yy, x + 7, yy), fill="black", width=2)
        draw.text((x - 30, max(40, y - 35)), f"{mean:.2f}%", fill="black")
        draw.text((x - 60, 390), row["dataset"], fill="black")
        draw.text((x - 30, 410), f"{row['method']} n={row['runs']}", fill="black")
    canvas.save(output)


def aggregate(results_dir, allow_incomplete=False):
    paths = sorted(results_dir.glob("*/seed_*/*/run.json"))
    paths = [p for p in paths if p.parent.name in ("forward", "gce")]
    if not paths:
        raise ValueError("No classifier records found")
    groups, runs, seen, reference, estimator_paths = defaultdict(list), [], set(), None, set()
    for path in paths:
        record = verified_record(path, "completed_training", ["best.pt", "history.json", "metrics.json", "predictions.npz"])
        identity = record["identity"]
        fingerprint = (identity["config_sha256"], identity["source_sha256"], json.dumps(identity["environment"], sort_keys=True))
        if reference is not None and fingerprint != reference:
            raise ValueError("Cannot aggregate runs from different configs, sources or environments")
        reference = fingerprint
        metrics = json.loads((path.parent / "metrics.json").read_text(encoding="utf-8"))
        key = (metrics["dataset"], metrics["method"], metrics["split_seed"])
        if key in seen or key != (identity["dataset"], identity["component"], identity["split_seed"]):
            raise ValueError("Duplicate or inconsistent classifier identity")
        seen.add(key)
        with np.load(path.parent / "predictions.npz", allow_pickle=False) as predictions:
            actual = float((predictions["probabilities"].argmax(1) == predictions["targets"]).mean())
        if not np.isclose(actual, metrics["test_accuracy"], atol=1e-12):
            raise ValueError("Stored predictions disagree with reported accuracy")
        groups[key[:2]].append(metrics)
        runs.append(metrics)
        estimator_paths.add(path.parent.parent / "estimator" / "run.json")
        history = json.loads((path.parent / "history.json").read_text(encoding="utf-8"))
        curve_plot(history, path.parent / "learning_curve.png")
    config = record["config"]
    expected = {(d, m, s) for d in config["datasets"] for m in config["methods"] for s in config["split_seeds"]}
    complete = (seen == expected and not config.get("debug", False)
                and len(config["split_seeds"]) >= 10)
    if not complete and not allow_incomplete:
        raise ValueError("Final summary needs the full >=10-split grid. Use --allow-incomplete for explicitly provisional/debug output")
    rows = []
    for (dataset, method), entries in sorted(groups.items()):
        mean, std = mean_std([e["test_top1_percent"] for e in entries])
        rows.append({"dataset": dataset, "method": method, "runs": len(entries),
                     "mean_top1_percent": mean, "std_top1_percent": std,
                     "split_seeds": sorted(e["split_seed"] for e in entries)})
    transitions = defaultdict(list)
    for path in sorted(estimator_paths):
        er = verified_record(path, "completed_estimator", ["best.pt", "history.json", "transition.json"])
        estimate = json.loads((path.parent / "transition.json").read_text(encoding="utf-8"))
        transitions[er["identity"]["dataset"]].append(estimate)
    matrix_rows = []
    for dataset, entries in sorted(transitions.items()):
        stack = np.array([e["estimated_transition"] for e in entries])
        item = {"dataset": dataset, "runs": len(entries), "mean_transition": stack.mean(0).tolist(),
                "std_transition": stack.std(0, ddof=1).tolist() if len(entries) > 1 else None}
        for error in ("mean_absolute_error", "frobenius_error"):
            if error in entries[0]:
                item[error + "_mean"], item[error + "_std"] = mean_std([e[error] for e in entries])
        matrix_rows.append(item)
        matrix_plot(item["mean_transition"], results_dir / f"{dataset}_transition.png", f"{dataset}: mean estimated transition (n={len(entries)})")
    pairs = []
    for dataset in config["datasets"]:
        first = {e["split_seed"]: e for e in groups[(dataset, "forward")]}
        second = {e["split_seed"]: e for e in groups[(dataset, "gce")]}
        common = sorted(first.keys() & second.keys())
        if common:
            mean, std = mean_std([second[s]["test_top1_percent"] - first[s]["test_top1_percent"] for s in common])
            pairs.append({"dataset": dataset, "paired_runs": len(common), "gce_minus_forward_mean_pp": mean,
                          "gce_minus_forward_std_pp": std})
    summary = {"kind": "final_classification" if complete else "provisional_classification",
               "complete": complete, "debug": config.get("debug", False), "rows": rows,
               "transition_estimates": matrix_rows, "paired_comparisons": pairs,
               "missing_jobs": [list(x) for x in sorted(expected - seen)],
               "standard_deviation": "sample standard deviation, ddof=1; null for n=1",
               "course_coverage_confirmed": config.get("course_coverage_confirmed", False)}
    write_json(results_dir / "summary.json", summary)
    fields = ["dataset", "method", "split_seed", "test_top1_percent", "best_epoch", "validation_noisy_nll", "elapsed_seconds", "debug"]
    with (results_dir / "runs.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# " + ("Final" if complete else "PROVISIONAL / DEBUG") + " classification results", "",
             "Clean test top-1 accuracy (%), sample SD (ddof=1).", "",
             "| Dataset | Method | Runs | Mean (%) | SD (%) |", "| --- | --- | ---: | ---: | ---: |"]
    for row in rows:
        sd = "N/A" if row["std_top1_percent"] is None else f"{row['std_top1_percent']:.3f}"
        lines.append(f"| {row['dataset']} | {row['method']} | {row['runs']} | {row['mean_top1_percent']:.3f} | {sd} |")
    for item in matrix_rows:
        lines.extend(["", f"## {item['dataset']}: estimated transition mean", "", "```", np.array2string(np.array(item["mean_transition"]), precision=5), "```"])
        if "mean_absolute_error_mean" in item:
            lines.append(f"Mean entrywise absolute error: {item['mean_absolute_error_mean']:.5f}; mean Frobenius error: {item['frobenius_error_mean']:.5f}.")
    lines.extend(["", "Matrices are estimated independently per split. CIFAR classifiers use their own split's matrix, not this aggregate.",
                  "Check the course coverage requirement against your lecture material before submitting."])
    (results_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    accuracy_plot(rows, results_dir / "accuracy_comparison.png", not complete)
    return summary
