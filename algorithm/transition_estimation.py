"""Noisy-label-only anchor-point transition matrix estimation.

T[i, j] = P(observed noisy label j | true class i).

The estimator learns noisy-label probabilities on an 80% training partition,
calibrates probabilities using the noisy 20% holdout, and estimates rows by
averaging the predicted noisy-label distributions for top-k anchor candidates.
Clean test labels (Yts) and clean test images (Xts) are NEVER loaded here.

Usage from repository root:
    python -m algorithm.transition_estimation --dataset FashionMNIST0.3 \
        --data-dir data --epochs 12 --output results/transition/FashionMNIST0.3.json

Uses PyTorch from the shared base requirements; feature/HOG studies have
their own optional ``requirements_additional.txt``.
The known FashionMNIST matrices are used only for POST-HOC evaluation.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import numpy as np

from .data import DATASETS, ROOT, load_noisy_pool, preprocess_images
from .transition import KNOWN as KNOWN_TRANSITIONS


# Shared published matrices are consulted only AFTER producing T_hat.


def split_noisy_pool(n: int, seed: int, train_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Random disjoint 80/20 indices, independent for each run seed."""
    if not (2 <= n and 0.0 < train_fraction < 1.0):
        raise ValueError("Need at least two samples and a fraction in (0, 1)")
    indices = np.random.default_rng(seed).permutation(n)
    cut = int(n * train_fraction)
    return indices[:cut], indices[cut:]


def _softmax(logits: np.ndarray) -> np.ndarray:
    stable = logits - np.max(logits, axis=1, keepdims=True)
    exponential = np.exp(stable)
    return exponential / np.sum(exponential, axis=1, keepdims=True)


def _cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray, temp: float) -> float:
    z = logits / temp
    z = z - np.max(z, axis=1, keepdims=True)
    log_normalizer = np.log(np.exp(z).sum(axis=1))
    return float(np.mean(log_normalizer - z[np.arange(len(labels)), labels]))


def calibrate_temperature(logits: np.ndarray, noisy_labels: np.ndarray) -> float:
    """Fit one scalar temperature using ONLY held-out NOISY labels.

    Small 1D golden-section optimization in log-temperature, no SciPy needed.
    It guards against overconfident anchor estimates, especially at high noise.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(noisy_labels)
    if logits.ndim != 2 or logits.shape[1] != 3 or labels.shape != (len(logits),):
        raise ValueError("Expected (N, 3) logits and (N,) noisy labels")
    if len(logits) < 2 or not np.isfinite(logits).all() or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Invalid logits or labels")
    lower, upper = math.log(0.25), math.log(20.0)
    inv_phi = (math.sqrt(5) - 1) / 2
    c = upper - inv_phi * (upper - lower)
    d = lower + inv_phi * (upper - lower)
    fc = _cross_entropy_from_logits(logits, labels, math.exp(c))
    fd = _cross_entropy_from_logits(logits, labels, math.exp(d))
    for _ in range(48):
        if fc < fd:
            upper, d, fd = d, c, fc
            c = upper - inv_phi * (upper - lower)
            fc = _cross_entropy_from_logits(logits, labels, math.exp(c))
        else:
            lower, c, fc = c, d, fd
            d = lower + inv_phi * (upper - lower)
            fd = _cross_entropy_from_logits(logits, labels, math.exp(d))
    return float(math.exp((lower + upper) / 2))


def estimate_transition_from_probabilities(
    noisy_probabilities: np.ndarray, *, top_k: int = 25,
) -> tuple[np.ndarray, dict]:
    """Average top-k per-class high-confidence P(noisy label | image).

    Top-k anchors reduce sensitivity to individual anomalous maxima. This is a
    heuristic relying on sufficiently pure anchor images and calibrated noisy
    probabilities; it is NOT a guaranteed-unbiased estimator in general.
    """
    probs = np.asarray(noisy_probabilities, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 3 or probs.shape[0] < 3:
        raise ValueError("Expected probabilities with shape (N >= 3, 3)")
    if not np.isfinite(probs).all() or np.min(probs) < 0 or np.max(probs) > 1:
        raise ValueError("Probabilities must be finite in [0,1]")
    if not np.allclose(probs.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Each row must sum to one")
    if not isinstance(top_k, int) or not 1 <= top_k <= len(probs):
        raise ValueError("top_k must be an integer between 1 and N")

    rows, anchors = [], {}
    for true_class in range(3):
        # Need deterministic ordering for reproducible tie breaking.
        indices = np.argsort(-probs[:, true_class], kind="stable")[:top_k]
        rows.append(probs[indices].mean(axis=0))
        anchors[str(true_class)] = {
            "top_k": top_k,
            "mean_selected_class_probability": float(probs[indices, true_class].mean()),
            "selected_class_probability_range": [float(probs[indices, true_class].min()),
                                                 float(probs[indices, true_class].max())],
        }
    transition = np.stack(rows)
    transition /= transition.sum(axis=1, keepdims=True)  # numerical guard
    return transition, anchors


def known_matrix_diagnostics(name: str, estimated: np.ndarray) -> dict | None:
    if name not in KNOWN_TRANSITIONS:
        return None
    delta = estimated - KNOWN_TRANSITIONS[name]
    return {"mae": float(np.abs(delta).mean()),
            "frobenius_error": float(np.linalg.norm(delta)),
            "true_transition": KNOWN_TRANSITIONS[name].tolist()}


def train_noisy_label_predictor(
    dataset_name: str, *, data_dir: Path, split_seed: int,
    epochs: int = 12, batch_size: int = 128, learning_rate: float = 1e-3,
    weight_decay: float = 1e-4, device: str = "cpu", num_threads: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train small CNN on noisy labels; return held-out logits and noisy labels.

    A single noisy validation split selects the best epoch. Validation labels
    calibrate noisy-label probabilities, not clean-label performance.
    """
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as error:
        raise RuntimeError("Install PyTorch for training: pip install torch") from error

    if epochs <= 0 or batch_size <= 0 or num_threads < 1:
        raise ValueError("epochs, batch_size and num_threads must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA device requested but unavailable")

    torch.set_num_threads(num_threads)
    init_seed = 1000 + split_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    random.seed(init_seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(init_seed)

    pool = load_noisy_pool(dataset_name, data_dir)
    train_idx, valid_idx = split_noisy_pool(len(pool.noisy_labels), split_seed)
    # Only the NOISY training pool is read, NEVER Xts/Yts.
    train_images = torch.from_numpy(preprocess_images(pool.images[train_idx], dataset_name))
    valid_images = torch.from_numpy(preprocess_images(pool.images[valid_idx], dataset_name))
    train_labels = torch.from_numpy(pool.noisy_labels[train_idx].astype(np.int64, copy=True))
    valid_labels = torch.from_numpy(pool.noisy_labels[valid_idx].astype(np.int64, copy=True))

    in_channels = 3 if dataset_name == "CIFAR" else 1
    # Stronger predictor for CIFAR: batch-normalized feature extraction before
    # pooling, retaining spatial information for hard noisy-label classification.
    net = nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.Conv2d(32, 32, 3, padding=1, bias=False),
        nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((2, 2)), nn.Flatten(),
        nn.Linear(128 * 2 * 2, 128), nn.ReLU(inplace=True),
        nn.Dropout(0.2), nn.Linear(128, 3),
    ).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(train_images, train_labels),
                        batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(init_seed))
    validation_loader = DataLoader(TensorDataset(valid_images, valid_labels),
                                   batch_size=batch_size * 2, num_workers=0)
    best_loss, best_epoch, best_params = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        net.train()
        train_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(net(xb), yb)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * len(xb)
        net.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for xb, yb in validation_loader:
                xb, yb = xb.to(device), yb.to(device)
                valid_loss += float(loss_fn(net(xb), yb)) * len(xb)
        valid_loss /= len(valid_labels)
        print(f"{dataset_name} epoch {epoch}/{epochs}: train noisy-CE={train_loss / len(train_labels):.4f}, validation noisy-CE={valid_loss:.4f}", flush=True)
        if valid_loss < best_loss:
            best_loss, best_epoch = valid_loss, epoch
            best_params = copy.deepcopy(net.state_dict())
    assert best_params is not None
    net.load_state_dict(best_params)
    net.eval()
    logits = []
    with torch.no_grad():
        for xb, _ in validation_loader:
            logits.append(net(xb.to(device)).cpu().numpy())
    observed = valid_labels.numpy()
    return np.concatenate(logits), observed, {
        "split_seed": split_seed, "initialization_seed": init_seed,
        "n_train": len(train_idx), "n_validation": len(valid_idx),
        "best_epoch": best_epoch, "best_validation_noisy_cross_entropy": best_loss,
        "epochs_max": epochs, "batch_size": batch_size, "architecture": "batchnorm_cnn_32_64_128",
        "data_access": "Xtr and Str only; no Xts/Yts",
        "model_selection": "minimum noisy validation cross entropy",
        "device": device,
    }


def run_estimator(dataset: str, *, data_dir: Path, split_seed: int,
                  epochs: int, batch_size: int, top_k: int, device: str, num_threads: int) -> dict:
    logits, labels, training_meta = train_noisy_label_predictor(
        dataset, data_dir=data_dir, split_seed=split_seed, epochs=epochs,
        batch_size=batch_size, device=device, num_threads=num_threads)
    temperature = calibrate_temperature(logits, labels)
    calibrated_probs = _softmax(logits / temperature)
    transition, anchor_meta = estimate_transition_from_probabilities(calibrated_probs, top_k=top_k)
    counts = np.bincount(labels, minlength=3).astype(float)
    observed_prior = counts / counts.sum()
    prior_ce = float(-np.sum(observed_prior[observed_prior > 0] * np.log(observed_prior[observed_prior > 0])))
    heldout_ce = _cross_entropy_from_logits(logits, labels, temperature)
    improvement = prior_ce - heldout_ce
    result = {
        "status": "preliminary_single_split_not_final",
        "classifier_runs_completed": 0,
        "dataset": dataset,
        "method": "noisy_CE_CNN_calibrated_top_k_anchor",
        "transition_convention": "T[i,j] = P(noisy label=j | clean class=i)",
        "estimated_transition_matrix": transition.tolist(),
        "temperature": temperature,
        "heldout_noisy_ce_after_calibration": heldout_ce,
        "constant_noisy_prior_ce_baseline": prior_ce,
        "noisy_validation_information_gain_nats": improvement,
        "noisy_validation_accuracy": float(np.mean(calibrated_probs.argmax(axis=1) == labels)),
        "weak_predictor_warning": bool(improvement < 0.01),
        "anchor_selection": anchor_meta,
        "training": training_meta,
        "known_matrix_validation_only": known_matrix_diagnostics(dataset, transition),
        "limitations": "Anchor-point assumption may fail; a near-uniform or high-noise matrix is hard to identify. Test labels not used.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=0, help="80/20 noisy pool split seed")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-threads", type=int, default=4)
    args = parser.parse_args()
    result = run_estimator(args.dataset, data_dir=args.data_dir, split_seed=args.seed,
                           epochs=args.epochs, batch_size=args.batch_size,
                           top_k=args.top_k, device=args.device, num_threads=args.num_threads)
    print("Estimated T (row = latent class, column = noisy label):")
    print(np.array(result["estimated_transition_matrix"]))
    if result["known_matrix_validation_only"]:
        print("Known-matrix diagnostic MAE:", result["known_matrix_validation_only"]["mae"])
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print("Wrote", args.output)
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
