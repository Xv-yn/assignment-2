"""Leakage-safe splitting, deterministic training and final-only evaluation."""

import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import preprocess_images, preprocess_labels
from .losses import forward_loss, gce_loss
from .models import build_model
from .provenance import write_json


def seed_everything(seed):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def random_split_indices(size, seed, fraction=0.8):
    if size < 5 or fraction != 0.8:
        raise ValueError("Assignment requires an 80/20 random split")
    indices = np.random.default_rng(seed).permutation(size)
    cut = int(size * fraction)
    return indices[:cut], indices[cut:]


def make_loader(images, labels, name, indices, batch_size, *, shuffle=False, seed=0):
    # Copy only this partition and keep tensors on CPU; bounded GPU memory.
    x = torch.from_numpy(preprocess_images(images[indices], name))
    y = torch.from_numpy(preprocess_labels(labels[indices]))
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def gce_uses_transition(params):
    """Standard GCE is matrix-free; only the optional extensions read T."""
    return bool(params.get("reweight", False)) or params.get("warmup_epochs", 0) > 0


def selection_metric(method, params, transition):
    """Name of the noisy-validation loss used for early stopping/checkpointing."""
    if method == "gce" and transition is None:
        return "noisy_gce"
    return "noisy_ce" if transition is None else "noisy_forward_nll"


@torch.no_grad()
def evaluate(model, loader, device, transition=None, gce_gamma=None):
    """Noisy NLL, noisy accuracy and clean-class probabilities for one loader.

    With ``gce_gamma`` set, also returns the mean (unweighted) GCE loss of the
    observed labels, which is the matrix-free selection criterion for GCE.
    """
    model.eval()
    total, correct, nll, gce = 0, 0, 0.0, 0.0
    probabilities, targets = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = F.cross_entropy(logits, labels) if transition is None else forward_loss(logits, labels, transition)
        p = logits.softmax(1)
        observed = p if transition is None else p @ transition
        nll += float(loss) * len(labels)
        if gce_gamma is not None:
            gce += float(gce_loss(logits, labels, gce_gamma)) * len(labels)
        correct += int((observed.argmax(1) == labels).sum())
        total += len(labels)
        probabilities.append(p.cpu().numpy())
        targets.append(labels.cpu().numpy())
    result = {"nll": nll / total, "accuracy": correct / total,
              "probabilities": np.concatenate(probabilities), "targets": np.concatenate(targets)}
    if gce_gamma is not None:
        result["gce"] = gce / total
    return result


def fit(name, config, train_loader, val_loader, device, seed, method, transition,
        output, *, epochs=None):
    seed_everything(seed)
    model = build_model(name, config["architecture"]).to(device)
    t = None if transition is None else torch.tensor(transition, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    epochs = epochs or config["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_loss, best_state, best_epoch, stale, history = float("inf"), None, 0, 0, []
    params = config["method_parameters"]["gce"]
    if method == "gce" and gce_uses_transition(params) and t is None:
        raise ValueError("GCE reweighting/warm-up needs a transition matrix")
    if method == "gce" and not gce_uses_transition(params) and t is not None:
        raise ValueError("Standard GCE is matrix-free; do not pass a transition matrix")
    metric = selection_metric(method, params, t)
    gce_gamma = params["gamma"] if metric == "noisy_gce" else None
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum, count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            if method == "ce":
                loss = F.cross_entropy(logits, labels)
            elif method == "forward" or (method == "gce" and epoch <= params.get("warmup_epochs", 0)):
                loss = forward_loss(logits, labels, t)
            elif method == "gce":
                loss = gce_loss(logits, labels, params["gamma"], t,
                                params["reweight"], params["max_weight"])
            else:
                raise ValueError(f"Unknown method {method}")
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            count += len(labels)
        score = evaluate(model, val_loader, device, t, gce_gamma)
        selection_loss = score["gce"] if metric == "noisy_gce" else score["nll"]
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               "validation_noisy_nll": score["nll"], "validation_noisy_accuracy": score["accuracy"],
               "validation_selection_loss": selection_loss,
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"{name} {method} epoch {epoch}/{epochs}: train={row['train_loss']:.4f}, "
              f"val_{metric}={selection_loss:.4f}", flush=True)
        if selection_loss < best_loss:
            best_loss, best_epoch, stale = selection_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        write_json(output / "history.json", {"method": method, "epochs": history,
                                            "best_epoch": best_epoch, "selection_metric": metric})
        scheduler.step()
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "dataset": name, "method": method,
                "architecture": copy.deepcopy(config["architecture"]), "transition": transition,
                "best_epoch": best_epoch, "selection_metric": metric,
                "validation_selection_loss": best_loss}, output / "best.pt")
    best_row = history[best_epoch - 1]
    return model, {"best_epoch": best_epoch, "selection_metric": metric,
                   "validation_selection_loss": best_loss,
                   "validation_noisy_nll": best_row["validation_noisy_nll"],
                   "epochs_trained": len(history)}


def clean_metrics(evaluation):
    labels = evaluation["targets"]
    predictions = evaluation["probabilities"].argmax(1)
    confusion = np.zeros((3, 3), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    return {"test_accuracy": float((labels == predictions).mean()),
            "test_top1_percent": float(100 * (labels == predictions).mean()),
            "test_count": len(labels), "confusion_matrix": confusion.tolist()}
