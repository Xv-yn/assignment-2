"""Run data smoke checks or inspect the planned training grid.

Parts 6–10 implement the future training backend. Training configurations
currently fail explicitly unless --plan is requested; no synthetic scores or
completed classifier records are produced.
"""

import argparse
from pathlib import Path

from .audit_data import audit_dataset, environment
from .data import DATASETS, ROOT, dataset_path
from .provenance import config_hash, sha256_file, source_hash, write_json
import json


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise ValueError("Expected configuration schema_version 1")
    datasets = config.get("datasets", [])
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS):
        raise ValueError("datasets must contain unique supported dataset names")
    if config.get("stage") not in ("data_smoke", "training"):
        raise ValueError("stage must be data_smoke or training")
    if type(config.get("batch_size")) is not int or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    if config.get("preprocessing") != {"scaling": "divide_by_255", "standardization": False, "augmentation": "none"}:
        raise ValueError("Only the documented deterministic preprocessing is implemented")
    if config["stage"] == "training":
        seeds = config.get("split_seeds", [])
        if len(seeds) < 10 or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("Training plan needs at least ten distinct nonnegative split seeds")
        if len(config.get("initialization_seeds", [])) != len(seeds):
            raise ValueError("Provide one initialization seed per split seed")
        if set(config.get("methods", [])) != {"forward", "gce"}:
            raise ValueError("Expected both proposed robust methods: forward and gce")
    elif type(config.get("audit_seed")) is not int or config["audit_seed"] < 0:
        raise ValueError("audit_seed must be a nonnegative integer")
    return config


def reusable_record(destination: Path, expected: dict) -> bool:
    record_path = destination / "run.json"
    if not record_path.exists():
        return False
    record = json.loads(record_path.read_text())
    if record.get("identity") != expected:
        raise ValueError(f"Existing record has different config/source/data/environment: {destination}; use a new results directory")
    if record.get("status") != "completed_data_smoke":
        return False
    for filename, expected_hash in record.get("artifacts", {}).items():
        artifact = destination / filename
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            return False
    return set(record.get("artifacts", {})) == {"audit.json", "training_grid.png"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--method", choices=("forward", "gce"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu",))
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args()
    config = read_config(resolve_path(str(args.config)))
    if args.device:
        config["device"] = args.device
    if config.get("device") != "cpu":
        parser.error("The Parts 1–5 data pipeline supports CPU; GPU training is deferred")
    if args.results_dir:
        config["results_dir"] = str(args.results_dir)
    selected = [args.dataset] if args.dataset else config["datasets"]
    if not set(selected) <= set(config["datasets"]):
        parser.error("Selected dataset is not present in the configuration")
    if config["stage"] == "training":
        if not args.plan:
            parser.error("Training is not implemented through Part 5. Use --plan to inspect the draft grid; implement Parts 6–10 before training.")
        seeds = [args.seed] if args.seed is not None else config["split_seeds"]
        if not set(seeds) <= set(config["split_seeds"]):
            parser.error("Selected seed is not in split_seeds")
        methods = [args.method] if args.method else config["methods"]
        jobs = [{"dataset": d, "method": m, "split_seed": s,
                 "initialization_seed": config["initialization_seeds"][config["split_seeds"].index(s)]}
                for d in selected for m in methods for s in seeds]
        print(json.dumps({"status": "planned_only", "job_count": len(jobs), "jobs": jobs}, indent=2))
        return
    if args.method is not None or args.seed is not None:
        parser.error("--method and --seed select training-plan jobs; smoke uses audit_seed and trains no method")
    if args.plan:
        print(json.dumps({"stage": "data_smoke", "datasets": selected}, indent=2))
        return
    output_root = resolve_path(config["results_dir"])
    data_dir = resolve_path(config["data_dir"])
    for name in selected:
        identity = {"dataset": name, "config_sha256": config_hash(config),
                    "source_sha256": source_hash(), "dataset_sha256": sha256_file(dataset_path(name, data_dir)),
                    "environment": environment()}
        destination = output_root / name
        if args.resume and reusable_record(destination, identity):
            print(f"{name}: skipped verified completed data smoke")
            continue
        if (destination / "run.json").exists() and not args.resume:
            parser.error(f"Existing record at {destination}; use --resume or a new --results-dir")
        write_json(destination / "run.json", {"status": "running_data_smoke", "identity": identity, "config": config})
        try:
            audit = audit_dataset(name, data_dir, destination, seed=config["audit_seed"], batch_size=config["batch_size"])
            write_json(destination / "audit.json", audit)
            artifacts = {p: sha256_file(destination / p) for p in ("audit.json", "training_grid.png")}
            write_json(destination / "run.json", {"status": "completed_data_smoke", "identity": identity,
                       "config": config, "artifacts": artifacts, "classifier_runs_completed": 0})
        except Exception as error:
            write_json(destination / "run.json", {"status": "failed_data_smoke", "identity": identity,
                       "config": config, "error": str(error)})
            raise
        print(f"{name}: data smoke passed; batch={audit['batch']['shape']}; noisy counts={audit['noisy_pool_class_counts']}")


if __name__ == "__main__":
    main()
