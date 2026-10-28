"""One estimator per random split, paired classifiers, and artifact verification."""

import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from .data import dataset_path, load_noisy_pool, load_clean_test
from .provenance import config_hash, sha256_file, source_hash, write_json
from .training import (seed_everything, random_split_indices, make_loader,
                       fit, evaluate, clean_metrics)
from .transition import KNOWN, estimate_anchor, validate_transition


def resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable in this PyTorch installation")
    return value


def verified_complete(directory, identity, status):
    path = directory / "run.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("identity") != identity:
        raise ValueError(f"Existing run has different provenance: {directory}; choose a new results directory")
    artifacts = record.get("artifacts", {})
    return (record.get("status") == status and bool(artifacts)
            and all((directory / p).is_file() and sha256_file(directory / p) == h
                    for p, h in artifacts.items()))


def save_completed(directory, identity, config, status, filenames):
    write_json(directory / "run.json", {"status": status, "identity": identity, "config": config,
               "artifacts": {p: sha256_file(directory / p) for p in filenames}})


def run_training(config, datasets, methods, seeds, output_root, data_dir, resume=False):
    device = resolve_device(config["device"])
    torch.set_num_threads(config["cpu_threads"])
    env = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
           "device": device, "cuda_runtime": torch.version.cuda,
           "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
           "platform": platform.platform()}
    source = source_hash()
    for name in datasets:
        pool = load_noisy_pool(name, data_dir)
        data_hash = sha256_file(dataset_path(name, data_dir))
        for split_seed in seeds:
            init_seed = config["initialization_seeds"][config["split_seeds"].index(split_seed)]
            base_identity = {"dataset": name, "split_seed": split_seed, "initialization_seed": init_seed,
                             "config_sha256": config_hash(config), "source_sha256": source,
                             "dataset_sha256": data_hash, "environment": env}
            destination = output_root / name / f"seed_{split_seed:03d}"
            destination.mkdir(parents=True, exist_ok=True)
            train_indices, val_indices = random_split_indices(len(pool.images), split_seed)
            # Persist the complete 80/20 split even in explicitly labelled debug runs.
            split_path = destination / "split.npz"
            if split_path.exists():
                with np.load(split_path, allow_pickle=False) as old:
                    if not (np.array_equal(old["train"], train_indices) and np.array_equal(old["validation"], val_indices)):
                        raise ValueError("Saved split disagrees with requested seed")
            else:
                np.savez_compressed(split_path, train=train_indices, validation=val_indices)
            debug = config.get("debug", False)
            used_train = train_indices[:config.get("debug_train_limit", len(train_indices))] if debug else train_indices
            used_val = val_indices[:config.get("debug_val_limit", len(val_indices))] if debug else val_indices
            def loaders(seed):
                return (make_loader(pool.images, pool.noisy_labels, name, used_train, config["batch_size"], shuffle=True, seed=seed),
                        make_loader(pool.images, pool.noisy_labels, name, used_val, config["batch_size"]))

            estimator_dir = destination / "estimator"
            identity = dict(base_identity, component="estimator")
            if not (resume and verified_complete(estimator_dir, identity, "completed_estimator")):
                if (estimator_dir / "run.json").exists() and not resume:
                    raise ValueError(f"Existing estimator: {estimator_dir}; use --resume")
                write_json(estimator_dir / "run.json", {"identity": identity, "status": "running_estimator", "config": config})
                try:
                    train_loader, val_loader = loaders(init_seed + 100000)
                    model, fit_info = fit(name, config, train_loader, val_loader, device, init_seed + 100000,
                                          "ce", None, estimator_dir, epochs=config["estimator"]["epochs"])
                    anchor_loader = make_loader(pool.images, pool.noisy_labels, name, used_train, config["batch_size"])
                    probabilities = evaluate(model, anchor_loader, device)["probabilities"]
                    matrix, info = estimate_anchor(probabilities, config["estimator"]["top_fraction"],
                                                    config["estimator"]["quantile"])
                    # Anchor indices refer to the original Xtr rows, not loader offsets.
                    info["anchor_pool_indices"] = [[int(used_train[i]) for i in row] for row in info["anchor_local_indices"]]
                    record = {"estimated_transition": matrix.tolist(), "diagnostics": info, "fit": fit_info,
                              "debug": debug, "training_examples": len(used_train), "test_data_used": False}
                    if name in KNOWN:
                        delta = matrix - KNOWN[name]
                        record.update(known_transition=KNOWN[name].tolist(),
                                      mean_absolute_error=float(np.abs(delta).mean()),
                                      frobenius_error=float(np.linalg.norm(delta)))
                    write_json(estimator_dir / "transition.json", record)
                    save_completed(estimator_dir, identity, config, "completed_estimator", ["transition.json", "history.json", "best.pt"])
                    del model, train_loader, val_loader, anchor_loader
                except Exception as error:
                    write_json(estimator_dir / "run.json", {"identity": identity, "status": "failed_estimator", "error": str(error)})
                    raise
            estimate = json.loads((estimator_dir / "transition.json").read_text(encoding="utf-8"))
            transition = KNOWN[name] if name in KNOWN else validate_transition(estimate["estimated_transition"])
            for method in methods:
                job_dir = destination / method
                identity = dict(base_identity, component=method, split_sha256=sha256_file(split_path),
                                transition_sha256=sha256_file(estimator_dir / "transition.json"))
                if resume and verified_complete(job_dir, identity, "completed_training"):
                    print(f"{name} seed={split_seed} {method}: skipped verified completion", flush=True)
                    continue
                if (job_dir / "run.json").exists() and not resume:
                    raise ValueError(f"Existing classifier: {job_dir}; use --resume")
                write_json(job_dir / "run.json", {"identity": identity, "status": "running_training", "config": config})
                start = time.perf_counter()
                try:
                    train_loader, val_loader = loaders(init_seed)
                    model, fit_info = fit(name, config, train_loader, val_loader, device, init_seed,
                                          method, transition.tolist(), job_dir)
                    # Test API is reached only after validation-only checkpoint selection.
                    test = load_clean_test(name, data_dir)
                    test_loader = make_loader(test.images, test.clean_labels, name, np.arange(len(test.images)), config["batch_size"])
                    evaluation = evaluate(model, test_loader, device)
                    metrics = dict(clean_metrics(evaluation), **fit_info, dataset=name, method=method,
                                   split_seed=split_seed, initialization_seed=init_seed, debug=debug,
                                   transition=transition.tolist(), transition_source="known" if name in KNOWN else "estimated",
                                   training_examples=len(used_train), validation_examples=len(used_val),
                                   elapsed_seconds=time.perf_counter() - start)
                    write_json(job_dir / "metrics.json", metrics)
                    np.savez_compressed(job_dir / "predictions.npz", probabilities=evaluation["probabilities"], targets=evaluation["targets"])
                    save_completed(job_dir, identity, config, "completed_training", ["best.pt", "history.json", "metrics.json", "predictions.npz"])
                    print(f"{name} seed={split_seed} {method}: clean test {metrics['test_top1_percent']:.2f}%", flush=True)
                    del model, train_loader, val_loader, test_loader, test
                except Exception as error:
                    write_json(job_dir / "run.json", {"identity": identity, "status": "failed_training", "error": str(error)})
                    raise
