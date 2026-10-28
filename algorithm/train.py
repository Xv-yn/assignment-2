"""Pilot training on noisy holdouts only; clean-test arrays are never loaded."""

import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .audit_data import environment
from .data import (ROOT, NoisyPool, dataset_path, load_noisy_pool, preprocess_images,
                   preprocessing_spec)
from .losses import (KNOWN_TRANSITIONS, observed_log_probabilities, per_example_loss,
                     validate_transition)
from .models import SmallCNN
from .provenance import config_hash, sha256_file, source_hash, write_json
from .splits import save_holdout, validate_holdout


class NoisyPartition(Dataset):
    def __init__(self, pool: NoisyPool, indices: np.ndarray):
        if not isinstance(pool, NoisyPool):
            raise TypeError("Training and validation accept only a NoisyPool")
        self.pool = pool
        self.indices = indices.copy()
        self.indices.setflags(write=False)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        image = preprocess_images(self.pool.images[index:index + 1], self.pool.name)[0]
        return torch.from_numpy(image), int(self.pool.noisy_labels[index])


def make_loaders(pool, split, *, batch_size, shuffle_seed):
    validate_holdout(split, len(pool.noisy_labels))
    generator = torch.Generator().manual_seed(shuffle_seed)
    train = DataLoader(NoisyPartition(pool, split.train), batch_size=batch_size,
                       shuffle=True, generator=generator, num_workers=0, drop_last=False)
    validation = DataLoader(NoisyPartition(pool, split.validation), batch_size=batch_size,
                            shuffle=False, num_workers=0, drop_last=False,
                            generator=torch.Generator().manual_seed(0))
    return train, validation


def run_epoch(model, loader, method, *, gamma, transition, optimizer=None):
    model.train(optimizer is not None)
    total_loss, correct, count = 0.0, 0, 0
    context = torch.enable_grad() if optimizer is not None else torch.inference_mode()
    with context:
        for images, labels in loader:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            losses = per_example_loss(logits, labels, method, gamma=gamma, transition=transition)
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Non-finite loss")
            if optimizer is not None:
                losses.mean().backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError("Non-finite gradient")
                optimizer.step()
            scores = observed_log_probabilities(logits.detach(), transition) if method == "forward" else logits.detach()
            correct += int((scores.argmax(1) == labels).sum())
            count += len(labels)
            total_loss += float(losses.detach().sum())
    if not count:
        raise ValueError("Empty partition")
    return {"loss": total_loss / count, "noisy_accuracy_percent": 100 * correct / count,
            "examples": count}


def reusable_training_record(destination, identity):
    path = destination / "run.json"
    if not path.exists():
        return False
    record = json.loads(path.read_text())
    if record.get("identity") != identity:
        raise ValueError(f"Existing pilot has different provenance: {destination}; use a new results directory")
    if record.get("status") != "completed_training_pilot":
        return False
    if set(record.get("artifacts", {})) != {"best.pt", "history.json"}:
        return False
    return all((destination / name).is_file() and sha256_file(destination / name) == digest
               for name, digest in record["artifacts"].items())


def run_training_job(config, name, method, split_seed, *, resume=False):
    """A completed pilot is never a completed final classifier evaluation."""
    from .run_experiments import resolve_path

    if method == "forward" and name not in KNOWN_TRANSITIONS:
        raise ValueError("CIFAR forward training needs the transition estimator, which is not implemented yet")
    position = config["split_seeds"].index(split_seed)
    initialization_seed = config["initialization_seeds"][position]
    shuffle_seed = config["shuffle_seeds"][position]
    data_dir = resolve_path(config["data_dir"])
    dataset_digest = sha256_file(dataset_path(name, data_dir))
    # load_noisy_pool reads only Xtr and Str, including during validation.
    pool = load_noisy_pool(name, data_dir)
    split, split_path = save_holdout(resolve_path(config["splits_dir"]), name,
                                     len(pool.noisy_labels), split_seed, dataset_digest)
    runtime = environment() | {"framework": "PyTorch", "torch": str(torch.__version__),
                               "num_threads": config["num_threads"], "deterministic_algorithms": True}
    identity = {"dataset": name, "method": method, "split_seed": split_seed,
                "initialization_seed": initialization_seed, "shuffle_seed": shuffle_seed,
                "dataset_sha256": dataset_digest, "split_sha256": sha256_file(split_path),
                "config_sha256": config_hash(config), "source_sha256": source_hash(), "environment": runtime}
    destination = resolve_path(config["results_dir"]) / name / method / f"split_{split_seed}"
    if resume and reusable_training_record(destination, identity):
        print(f"{name}/{method}/{split_seed}: skipped verified completed pilot", flush=True)
        return json.loads((destination / "run.json").read_text())
    if (destination / "run.json").exists() and not resume:
        raise ValueError(f"Existing pilot: {destination}; use --resume or a new results directory")
    transition = torch.tensor(KNOWN_TRANSITIONS[name], dtype=torch.float32) if method == "forward" else None
    if transition is not None:
        validate_transition(transition)
    selection = "validation_forward_noisy_nll" if method == "forward" else f"validation_{method}_noisy_loss"
    record = {"identity": identity, "config": config, "kind": "training_pilot",
              "split_path": str(split_path.relative_to(ROOT)) if split_path.is_relative_to(ROOT) else str(split_path),
              "split_counts": {"train": len(split.train), "validation": len(split.validation)},
              "noisy_class_counts": {"train": np.bincount(pool.noisy_labels[split.train], minlength=3).tolist(),
                                     "validation": np.bincount(pool.noisy_labels[split.validation], minlength=3).tolist()},
              "preprocessing": preprocessing_spec(), "selection_metric": selection,
              "selection_rule": "minimum noisy validation loss; earliest improvement exceeding min_delta",
              "transition_convention": "T[i,j]=P(S=j|Y=i)",
              "transition": transition.tolist() if transition is not None else None,
              "transition_source": "assignment_supplied" if transition is not None else "unused",
              "clean_test_accessed": False, "final_classifier_runs_completed": 0}
    write_json(destination / "run.json", record | {"status": "running_training_pilot"})
    start = time.perf_counter()
    try:
        random.seed(initialization_seed)
        np.random.seed(initialization_seed)
        torch.manual_seed(initialization_seed)
        torch.set_num_threads(config["num_threads"])
        torch.use_deterministic_algorithms(True)
        model = SmallCNN(3 if name == "CIFAR" else 1, config["architecture"]["channels"])
        optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"])
        train, validation = make_loaders(pool, split, batch_size=config["batch_size"], shuffle_seed=shuffle_seed)
        gamma = config["method_parameters"]["gce"]["gamma"]
        best_loss, best_epoch, stale, history = float("inf"), None, 0, []
        for epoch in range(1, config["epochs"] + 1):
            training_metrics = run_epoch(model, train, method, gamma=gamma, transition=transition, optimizer=optimizer)
            validation_metrics = run_epoch(model, validation, method, gamma=gamma, transition=transition)
            history.append({"epoch": epoch, "training": training_metrics, "validation": validation_metrics})
            if validation_metrics["loss"] < best_loss - config["min_delta"]:
                best_loss, best_epoch, stale = validation_metrics["loss"], epoch, 0
                temporary = destination / "best.pt.tmp"
                torch.save({"model_state": model.state_dict(), "epoch": epoch,
                            "validation": validation_metrics, "identity": identity,
                            "config": config}, temporary)
                os.replace(temporary, destination / "best.pt")
            else:
                stale += 1
            write_json(destination / "history.json", history)
            print(f"{name}/{method}/{split_seed} epoch={epoch}: train_loss={training_metrics['loss']:.5f}, "
                  f"validation_loss={validation_metrics['loss']:.5f}, "
                  f"noisy_validation_accuracy={validation_metrics['noisy_accuracy_percent']:.2f}%", flush=True)
            if stale >= config["patience"]:
                break
        completed = record | {"status": "completed_training_pilot", "selected_epoch": best_epoch,
                              "selected_validation": history[best_epoch - 1]["validation"],
                              "epochs_completed": len(history), "elapsed_seconds": time.perf_counter() - start,
                              "artifacts": {p: sha256_file(destination / p) for p in ("best.pt", "history.json")}}
        write_json(destination / "run.json", completed)
        return completed
    except BaseException as error:
        write_json(destination / "run.json", record | {"status": "failed_training_pilot", "error": str(error)})
        raise
