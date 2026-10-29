"""Run data audits or paired noise-robust classification experiments."""

import argparse
from pathlib import Path

from .audit_data import audit_dataset, environment
from .data import DATASETS, ROOT, dataset_path
from .pilot_config import validate_pilot_config, validate_pilot_training_fields
from .provenance import config_hash, sha256_file, source_hash, write_json
import json


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_config(path: Path) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Expected configuration schema_version 1")
    datasets = config.get("datasets", [])
    if not datasets or len(set(datasets)) != len(datasets) or not set(datasets) <= set(DATASETS):
        raise ValueError("datasets must contain unique supported dataset names")
    if config.get("stage") not in ("data_smoke", "training_pilot", "training"):
        raise ValueError("stage must be data_smoke, training_pilot or training")
    if type(config.get("batch_size")) is not int or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    if config.get("preprocessing") != {"scaling": "divide_by_255", "standardization": False, "augmentation": "none"}:
        raise ValueError("Only the documented deterministic preprocessing is implemented")
    if config["stage"] == "training_pilot":
        validate_pilot_training_fields(config)
    elif config["stage"] == "training":
        seeds = config.get("split_seeds", [])
        minimum = 1 if config.get("debug", False) else 10
        if len(seeds) < minimum or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError("Training plan needs at least ten distinct nonnegative split seeds")
        if len(config.get("initialization_seeds", [])) != len(seeds):
            raise ValueError("Provide one initialization seed per split seed")
        if len(config.get("methods", [])) != 2 or set(config.get("methods", [])) != {"forward", "gce"}:
            raise ValueError("Expected both proposed robust methods: forward and gce")
        validate_training_config(config)
    elif type(config.get("audit_seed")) is not int or config["audit_seed"] < 0:
        raise ValueError("audit_seed must be a nonnegative integer")
    return config


def validate_training_config(config):
    import math
    if config.get("device") not in ("cpu", "cuda", "auto"):
        raise ValueError("device must be cpu, cuda or auto")
    if config.get("train_fraction") != 0.8:
        raise ValueError("train_fraction must be 0.8")
    for key in ("epochs", "patience", "cpu_threads"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay"):
        value = config.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0 or (key == "learning_rate" and value == 0):
            raise ValueError(f"Invalid {key}")
    if config.get("optimizer") != "adam":
        raise ValueError("Only Adam is supported")
    if any(type(s) is not int or not 0 <= s < 2**32 - 100000 for s in config["initialization_seeds"]):
        raise ValueError("Invalid initialization seed")
    architecture = config["architecture"]
    channels = architecture["channels"]
    if (architecture["name"] != "small_cnn" or architecture["num_classes"] != 3
            or not 1 <= len(channels) <= 3 or any(type(c) is not int or c < 1 for c in channels)):
        raise ValueError("Invalid small_cnn architecture")
    params = config["method_parameters"]["gce"]
    if not 0 < params["gamma"] <= 1 or type(params["reweight"]) is not bool or params["max_weight"] <= 0:
        raise ValueError("Invalid GCE parameters")
    if type(params["warmup_epochs"]) is not int or params["warmup_epochs"] < 0:
        raise ValueError("Invalid warmup_epochs")
    estimator = config["estimator"]
    if (estimator["name"] != "noisy_ce_anchor" or estimator["pool"] != "current_training_partition_only"
            or type(estimator["epochs"]) is not int or estimator["epochs"] < 1
            or not 0 <= estimator["top_fraction"] <= 1 or not 0 < estimator["quantile"] <= 1):
        raise ValueError("Invalid anchor estimator configuration")
    if config.get("test_evaluation") != "after_validation_selection":
        raise ValueError("Test evaluation must follow validation selection")
    for key in ("debug_train_limit", "debug_val_limit"):
        if config.get("debug") and (type(config.get(key)) is not int or config[key] < 3):
            raise ValueError(f"Invalid {key}")


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


def main(argv=None, *, pilot_only=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS)
    parser.add_argument("--method", choices=("forward", "gce", "ce"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu",) if pilot_only else ("cpu", "cuda", "auto"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan", action="store_true")
    args = parser.parse_args(argv)
    config = read_config(resolve_path(str(args.config)))
    if args.device:
        config["device"] = args.device
    if args.data_dir:
        config["data_dir"] = str(args.data_dir.resolve())
    if args.results_dir:
        config["results_dir"] = str(args.results_dir)
    if pilot_only and config.get("device") != "cpu":
        parser.error("The preserved pilot entry point supports CPU only")
    selected = [args.dataset] if args.dataset else config["datasets"]
    if not set(selected) <= set(config["datasets"]):
        parser.error("Selected dataset is not present in the configuration")
    if config["stage"] in ("training", "training_pilot"):
        seeds = [args.seed] if args.seed is not None else config["split_seeds"]
        if not set(seeds) <= set(config["split_seeds"]):
            parser.error("Selected seed is not in split_seeds")
        methods = [args.method] if args.method else config["methods"]
        if not set(methods) <= set(config["methods"]):
            parser.error("Selected method is not present in the configuration")
        jobs = [{"dataset": d, "method": m, "split_seed": s,
                 "initialization_seed": config["initialization_seeds"][config["split_seeds"].index(s)]}
                for d in selected for m in methods for s in seeds]
        if args.plan:
            plan = {"status": "planned_only", "debug": config.get("debug", False), "job_count": len(jobs), "jobs": jobs}
            if pilot_only:
                plan["stage"] = config["stage"]
            print(json.dumps(plan, indent=2))
            return
        if pilot_only and config["stage"] == "training":
            parser.error("Use algorithm.run_experiments for formal training; algorithm.run_pilots is the CPU pilot compatibility entry point")
        if config["stage"] == "training_pilot":
            if config.get("device") != "cpu":
                parser.error("The preserved pilot implementation supports CPU only")
            if any(job["dataset"] == "CIFAR" and job["method"] == "forward" for job in jobs):
                parser.error("Pilot Forward uses known matrices only; use final.json for estimated CIFAR matrices")
            from .train import run_training_job
            for job in jobs:
                run_training_job(config, job["dataset"], job["method"], job["split_seed"], resume=args.resume)
            return
        from .experiments import run_training
        run_training(config, selected, methods, seeds, resolve_path(config["results_dir"]),
                     resolve_path(config["data_dir"]), args.resume)
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
