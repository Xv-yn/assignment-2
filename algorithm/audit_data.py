"""Audit all four supplied arrays and render only noisy-pool examples."""

import argparse
import platform
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, __version__ as pillow_version

from .data import (DATASETS, ROOT, dataset_path, load_clean_test, load_noisy_pool,
                   preprocess_images, preprocess_labels, preprocessing_spec)
from .provenance import sha256_file, write_json


def array_summary(array: np.ndarray) -> dict:
    return {"shape": list(array.shape), "dtype": str(array.dtype),
            "min": int(array.min()), "max": int(array.max()),
            "all_finite": bool(np.isfinite(array).all())}


def training_grid(pool, output: Path, seed: int) -> list[int]:
    """Four examples per observed class, sampled without consulting clean labels."""
    rng = np.random.default_rng(seed)
    indices = [int(i) for label in range(3)
               for i in rng.choice(np.flatnonzero(pool.noisy_labels == label), 4, replace=False)]
    canvas = Image.new("RGB", (4 * 144, 3 * 154 + 32), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"{pool.name}: Xtr only; S = observed noisy ID", fill="black")
    for k, index in enumerate(indices):
        raw = pool.images[index]
        raw = raw if pool.name == "CIFAR" else raw.reshape(28, 28)
        tile = Image.fromarray(raw).convert("RGB").resize((112, 112), Image.Resampling.NEAREST)
        x, y = (k % 4) * 144 + 8, (k // 4) * 154 + 32
        canvas.paste(tile, (x, y))
        draw.text((x, y + 114), f"index {index}; S={int(pool.noisy_labels[index])}", fill="black")
    canvas.save(output)
    return indices


def audit_dataset(name: str, data_dir: Path, output_dir: Path, *, seed: int, batch_size: int) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = load_noisy_pool(name, data_dir)
    # Integrity only: no test image plotting, fitted transforms, or model scores.
    test = load_clean_test(name, data_dir)
    batch = preprocess_images(pool.images[:batch_size], name)
    labels = preprocess_labels(pool.noisy_labels[:batch_size])
    test_batch = preprocess_images(test.images[:batch_size], name)
    grid_path = output_dir / "training_grid.png"
    grid_indices = training_grid(pool, grid_path, seed)
    counts = np.bincount(pool.noisy_labels, minlength=3)
    return {
        "dataset": name, "dataset_sha256": sha256_file(dataset_path(name, data_dir)),
        "arrays": {"Xtr": array_summary(pool.images), "Str": array_summary(pool.noisy_labels),
                   "Xts": array_summary(test.images), "Yts": array_summary(test.clean_labels)},
        "noisy_pool_class_counts": {str(i): int(n) for i, n in enumerate(counts)},
        "noisy_pool_class_proportions": (counts / counts.sum()).tolist(),
        "batch": {"shape": list(batch.shape), "image_dtype": str(batch.dtype),
                  "label_dtype": str(labels.dtype), "min": float(batch.min()),
                  "max": float(batch.max()), "contiguous": bool(batch.flags.c_contiguous)},
        "deterministic_pool_transform": bool(np.array_equal(batch, preprocess_images(pool.images[:batch_size], name))),
        "deterministic_test_transform": bool(np.array_equal(test_batch, preprocess_images(test.images[:batch_size], name))),
        "preprocessing": preprocessing_spec(), "grid_indices": grid_indices,
        "grid_sampling_seed": seed, "grid_sha256": sha256_file(grid_path),
        "test_access": "array integrity and deterministic transform only; no performance evaluation",
    }


def environment() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "Pillow": pillow_version, "platform": platform.platform(),
            "device": "cpu", "framework": "NumPy data integrity audit"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "audit")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    args = parser.parse_args()
    for name in args.dataset or DATASETS:
        destination = args.output_dir / name
        record = audit_dataset(name, args.data_dir, destination, seed=2026, batch_size=32)
        record["environment"] = environment()
        write_json(destination / "audit.json", record)
        print(f"{name}: passed; noisy counts={record['noisy_pool_class_counts']}; {destination}")


if __name__ == "__main__":
    main()
