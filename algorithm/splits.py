"""Reproducible random 80/20 holdouts, shared across methods and saved as indices."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path

import numpy as np

from .data import DATASETS, ROOT, dataset_path, load_noisy_pool
from .provenance import sha256_file, write_json


@dataclass(frozen=True)
class Holdout:
    train: np.ndarray
    validation: np.ndarray


def validate_holdout(split: Holdout, count: int) -> None:
    for indices, size in ((split.train, count * 4 // 5),
                          (split.validation, count - count * 4 // 5)):
        if indices.dtype != np.int64 or indices.shape != (size,):
            raise ValueError("Holdout indices must be int64 with exact 80/20 sizes")
    combined = np.concatenate((split.train, split.validation))
    if not np.array_equal(np.sort(combined), np.arange(count)):
        raise ValueError("Holdouts must be disjoint and cover every noisy-pool example once")


def random_holdout(count: int, seed: int) -> Holdout:
    if type(count) is not int or count < 5 or count % 5:
        raise ValueError("Pool size must be a positive multiple of five")
    if type(seed) is not int or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    # Local generator: model initialisation and batch shuffling cannot change splits.
    order = np.random.Generator(np.random.PCG64(seed)).permutation(count).astype(np.int64)
    split = Holdout(order[:count * 4 // 5], order[count * 4 // 5:])
    validate_holdout(split, count)
    split.train.setflags(write=False)
    split.validation.setflags(write=False)
    return split


def save_holdout(root: Path, dataset: str, count: int, seed: int,
                 dataset_sha256: str) -> tuple[Holdout, Path]:
    """Reuse only the exact split for this dataset content and generator version."""
    import json

    split = random_holdout(count, seed)
    path = root / dataset / f"split_{seed}.npz"
    metadata_path = path.with_suffix(".json")
    identity = {"dataset": dataset, "dataset_sha256": dataset_sha256,
                "pool_size": count, "split_seed": seed, "train_fraction": 0.8,
                "generator": "numpy.PCG64.permutation.v1", "numpy": np.__version__}
    if path.exists() or metadata_path.exists():
        if not path.is_file() or not metadata_path.is_file():
            raise ValueError(f"Incomplete saved split: {path}; use a new split directory")
        metadata = json.loads(metadata_path.read_text())
        if metadata.get("identity") != identity or metadata.get("sha256") != sha256_file(path):
            raise ValueError(f"Saved split provenance mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"train", "validation"}:
                raise ValueError(f"Invalid split archive: {path}")
            saved = Holdout(archive["train"], archive["validation"])
        validate_holdout(saved, count)
        if not (np.array_equal(saved.train, split.train) and
                np.array_equal(saved.validation, split.validation)):
            raise ValueError(f"Saved split differs from its seed: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez(stream, train=split.train, validation=split.validation)
        os.replace(temporary, path)
        write_json(metadata_path, {"identity": identity, "sha256": sha256_file(path),
                                  "counts": {"train": len(split.train),
                                             "validation": len(split.validation)}})
    return split, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/splits")
    parser.add_argument("--dataset", choices=DATASETS, action="append")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        parser.error("Provide distinct nonnegative seeds")
    for name in args.dataset or DATASETS:
        pool = load_noisy_pool(name, args.data_dir)
        digest = sha256_file(dataset_path(name, args.data_dir))
        identities = set()
        for seed in args.seeds:
            split, path = save_holdout(args.output_dir, name, len(pool.noisy_labels), seed, digest)
            # Partition membership, rather than index order, must differ across runs.
            membership = np.sort(split.train).tobytes()
            if membership in identities:
                raise ValueError(f"Duplicate training partition for {name}, seed {seed}")
            identities.add(membership)
            print(f"{name} seed={seed}: train={len(split.train)}, validation={len(split.validation)}; {path}")


if __name__ == "__main__":
    main()
