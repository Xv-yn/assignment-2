"""Strict supplied-data loading; training callers receive Xtr/Str only.

Clean test arrays have a separate API, reserved for integrity audits and the
final evaluator. No fitted normalization or stochastic augmentation is
used in the initial design.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("FashionMNIST0.3", "FashionMNIST0.6", "CIFAR")
KEYS = {"Xtr", "Str", "Xts", "Yts"}


def dataset_path(name: str, data_dir: Path | str = ROOT / "data") -> Path:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; expected one of {DATASETS}")
    path = Path(data_dir) / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing supplied dataset: {path}")
    return path


def _validate_pair(name: str, images: np.ndarray, labels: np.ndarray,
                   *, test: bool) -> None:
    color = name == "CIFAR"
    count = 3000 if test else (15000 if color else 18000)
    shape = (count, 32, 32, 3) if color else (count, 784)
    if images.shape != shape or labels.shape != (count,):
        raise ValueError(f"{name}: expected images {shape}, labels {(count,)}; "
                         f"got {images.shape}, {labels.shape}")
    if images.dtype != np.uint8:
        raise ValueError(f"{name}: images must have dtype uint8")
    expected_label_dtype = np.dtype("int64" if color else "uint8")
    if labels.dtype != expected_label_dtype:
        raise ValueError(f"{name}: labels must have dtype {expected_label_dtype}")
    if not np.isfinite(images).all() or not np.isfinite(labels).all():
        raise ValueError(f"{name}: non-finite images or labels")
    if images.min() < 0 or images.max() > 255:
        raise ValueError(f"{name}: pixels outside [0, 255]")
    if set(np.unique(labels).tolist()) != {0, 1, 2}:
        raise ValueError(f"{name}: expected exactly label IDs 0, 1, 2")


def _load_pair(name: str, data_dir: Path | str, *, test: bool):
    with np.load(dataset_path(name, data_dir), allow_pickle=False) as archive:
        if len(archive.files) != 4 or set(archive.files) != KEYS:
            raise ValueError(f"{name}: expected exactly keys {sorted(KEYS)}")
        # NpzFile lazily loads arrays. Training never reads Xts or Yts.
        images = archive["Xts" if test else "Xtr"]
        labels = archive["Yts" if test else "Str"]
    _validate_pair(name, images, labels, test=test)
    images.setflags(write=False)
    labels.setflags(write=False)
    return images, labels


@dataclass(frozen=True)
class NoisyPool:
    name: str
    images: np.ndarray
    noisy_labels: np.ndarray


@dataclass(frozen=True)
class CleanTest:
    name: str
    images: np.ndarray
    clean_labels: np.ndarray


def load_noisy_pool(name: str, data_dir: Path | str = ROOT / "data") -> NoisyPool:
    """Load the combined noisy pool, not a training split; split before fitting."""
    return NoisyPool(name, *_load_pair(name, data_dir, test=False))


def load_clean_test(name: str, data_dir: Path | str = ROOT / "data") -> CleanTest:
    """Reserved for data-integrity inspection and final evaluation."""
    return CleanTest(name, *_load_pair(name, data_dir, test=True))


def preprocess_images(images: np.ndarray, name: str) -> np.ndarray:
    """Deterministic uint8 -> contiguous float32 NCHW / 255; no fitting."""
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    if images.dtype != np.uint8:
        raise ValueError("Expected raw uint8 images, to prevent double scaling")
    if name == "CIFAR":
        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError("CIFAR batch must have shape (N, 32, 32, 3)")
        images = images.transpose(0, 3, 1, 2)
    else:
        if images.ndim != 2 or images.shape[1:] != (784,):
            raise ValueError("FashionMNIST batch must have shape (N, 784)")
        images = images.reshape(-1, 1, 28, 28)
    result = np.array(images, dtype=np.float32, order="C", copy=True)
    result /= np.float32(255)
    return result


def preprocess_labels(labels: np.ndarray) -> np.ndarray:
    if labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("Labels must be a one-dimensional integer array")
    if not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Labels must be class IDs 0, 1, 2")
    return labels.astype(np.int64, copy=True)


def preprocessing_spec() -> dict:
    return {
        "image_dtype": "float32", "layout": "NCHW", "scale_divisor": 255,
        "pixel_range": [0, 1], "label_dtype": "int64", "class_ids": [0, 1, 2],
        "resize": None, "standardization": None, "augmentation": None,
        "fitted_statistics": None,
        "statistics_note": "No fitted transform; identical deterministic scaling for all methods and partitions.",
    }
