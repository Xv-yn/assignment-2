"""Run data checks, train noisy-validation pilots, or inspect the final grid."""

import argparse
import math
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
    if config.get("stage") not in ("data_smoke", "training_pilot", "training"):
        raise ValueError("stage must be data_smoke, training_pilot or training")
    if type(config.get("batch_size")) is not int or config["batch_size"] <= 0:
        raise ValueError("batch_size must be a positive integer")
    if config.get("preprocessing") != {"scaling": "divide_by_255", "standardization": False, "augmentation": "none"}:
        raise ValueError("Only the documented deterministic preprocessing is implemented")
    if config["stage"] in ("training", "training_pilot"):
        seeds = config.get("split_seeds", [])
        minimum = 10 if config["stage"] == "training" else 1
        if len(seeds) < minimum or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
            raise ValueError(f"Training needs at least {minimum} distinct nonnegative split seeds")
        initial = config.get("initialization_seeds", [])
        if len(initial) != len(seeds) or any(type(s) is not int or s < 0 for s in initial):
            raise ValueError("Provide one nonnegative integer initialization seed per split seed")
        methods = config.get("methods", [])
        if not methods or len(set(methods)) != len(methods) or not set(methods) <= {"forward", "gce", "ce"}:
            raise ValueError("Provide unique methods from forward, gce and ce")
        if config["stage"] == "training" and set(methods) != {"forward", "gce"}:
            raise ValueError("Expected both proposed robust methods: forward and gce")
        if config["stage"] == "training_pilot":
            validate_pilot_config(config)
    elif type(config.get("audit_seed")) is not int or config["audit_seed"] < 0:
        raise ValueError("audit_seed must be a nonnegative integer")
    return config


def validate_pilot_config(config):
    if config.get("train_fraction") != 0.8:
        raise ValueError("The assignment requires an exact 80/20 holdout")
    for field in ("epochs", "patience", "num_threads"):
        if type(config.get(field)) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    for field, allow_zero in (("learning_rate", False), ("weight_decay", True), ("min_delta", True)):
        value = config.get(field)
        if (type(value) not in (int, float) or not math.isfinite(value) or
                value < 0 or (not allow_zero and value == 0)):
            raise ValueError(f"Invalid {field}")
    shuffle_seeds = config.get("shuffle_seeds", [])
    if (len(shuffle_seeds) != len(config["split_seeds"]) or
            any(type(s) is not int or s < 0 for s in shuffle_seeds)):
        raise ValueError("Provide one nonnegative integer shuffle seed per split seed")
    architecture = config.get("architecture", {})
    channels = architecture.get("channels", [])
    if (architecture.get("name") != "small_cnn" or architecture.get("num_classes") != 3 or
            len(channels) != 2 or any(type(c) is not int or c <= 0 for c in channels)):
        raise ValueError("Expected small_cnn with two positive channel widths and three classes")
    gamma = config.get("method_parameters", {}).get("gce", {}).get("gamma")
    if type(gamma) not in (int, float) or not 0 < gamma <= 1:
        raise ValueError("GCE gamma must lie in (0, 1]")
    for field in ("data_dir", "results_dir", "splits_dir"):
        if not isinstance(config.get(field), str) or not config[field]:
            raise ValueError(f"Missing {field}")
    if config.get("optimizer") != "adam" or config.get("test_evaluation") != "disabled":
        raise ValueError("Pilots require Adam and disabled clean-test evaluation")


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
    parser.add_argument("--method", choices=("forward", "gce", "ce"))
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
        parser.error("The current data and training pipeline supports CPU")
    if args.results_dir:
        config["results_dir"] = str(args.results_dir)
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
            print(json.dumps({"status": "planned_only", "stage": config["stage"],
                              "job_count": len(jobs), "jobs": jobs}, indent=2))
            return
        if config["stage"] == "training":
            parser.error("Final experiments are not ready: CIFAR transition estimation and clean-test evaluation remain. Use --plan or a training_pilot config.")
        # Validate the entire selected grid before starting any expensive work.
        if any(j["dataset"] == "CIFAR" and j["method"] == "forward" for j in jobs):
            parser.error("CIFAR forward training requires the unimplemented transition estimator")
        from .train import run_training_job
        for job in jobs:
            run_training_job(config, job["dataset"], job["method"], job["split_seed"], resume=args.resume)
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
