"""Data leakage and transform regressions, runnable without extra test packages."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from algorithm.data import (ROOT, _validate_pair, dataset_path, load_noisy_pool,
                            preprocess_images, preprocess_labels)
from algorithm.provenance import config_hash, sha256_file, write_json
from algorithm.run_experiments import read_config, reusable_record


class DataTests(unittest.TestCase):
    def test_grayscale_spatial_order_scale_and_dtype(self):
        raw = np.zeros((2, 784), dtype=np.uint8)
        raw[0, 29] = 255
        raw[1, 0] = 128
        actual = preprocess_images(raw, "FashionMNIST0.3")
        self.assertEqual(actual.shape, (2, 1, 28, 28))
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual[0, 0, 1, 1], 1)
        self.assertAlmostEqual(float(actual[1, 0, 0, 0]), 128 / 255, places=7)
        self.assertEqual(raw[0, 29], 255)

    def test_color_channel_order(self):
        raw = np.zeros((1, 32, 32, 3), dtype=np.uint8)
        raw[0, 4, 7] = [255, 128, 0]
        actual = preprocess_images(raw, "CIFAR")
        self.assertEqual(actual.shape, (1, 3, 32, 32))
        np.testing.assert_allclose(actual[0, :, 4, 7], [1, 128 / 255, 0])
        self.assertTrue(actual.flags.c_contiguous)

    def test_deterministic_transform_and_no_double_scaling(self):
        raw = np.arange(784, dtype=np.uint16).astype(np.uint8).reshape(1, -1)
        first = preprocess_images(raw, "FashionMNIST0.6")
        np.testing.assert_array_equal(first, preprocess_images(raw, "FashionMNIST0.6"))
        with self.assertRaises(ValueError):
            preprocess_images(first, "FashionMNIST0.6")

    def test_invalid_layout(self):
        with self.assertRaises(ValueError):
            preprocess_images(np.zeros((1, 3, 32, 32), dtype=np.uint8), "CIFAR")

    def test_integer_label_conversion_preserves_ids(self):
        labels = np.array([2, 0, 1], dtype=np.uint8)
        actual = preprocess_labels(labels)
        self.assertEqual(actual.dtype, np.int64)
        np.testing.assert_array_equal(actual, labels)
        for bad in (np.array([3]), np.array([-1]), np.array([1.5])):
            with self.assertRaises(ValueError):
                preprocess_labels(bad)

    def test_full_shape_validation_and_missing_classes(self):
        images = np.zeros((18000, 784), dtype=np.uint8)
        labels = np.resize(np.array([0, 1, 2], dtype=np.uint8), 18000)
        _validate_pair("FashionMNIST0.3", images, labels, test=False)
        for x, s in ((images[:-1], labels), (images, labels.astype(np.int64)),
                     (images, np.zeros_like(labels)), (images.astype(np.float32), labels)):
            with self.assertRaises(ValueError):
                _validate_pair("FashionMNIST0.3", x, s, test=False)

    def test_training_loader_does_not_read_clean_arrays(self):
        class Archive:
            files = ["Xtr", "Str", "Xts", "Yts"]
            reads = []

            def __enter__(self): return self
            def __exit__(self, *args): return False

            def __getitem__(self, key):
                self.reads.append(key)
                if key == "Xtr": return np.zeros((18000, 784), dtype=np.uint8)
                if key == "Str": return np.resize(np.array([0, 1, 2], dtype=np.uint8), 18000)
                raise AssertionError(f"Training attempted to read {key}")

        archive = Archive()
        with patch("algorithm.data.dataset_path", return_value=Path("supplied.npz")), patch("algorithm.data.np.load", return_value=archive) as loader:
            pool = load_noisy_pool("FashionMNIST0.3")
        loader.assert_called_once_with(Path("supplied.npz"), allow_pickle=False)
        self.assertEqual(archive.reads, ["Xtr", "Str"])
        self.assertFalse(hasattr(pool, "clean_labels"))
        self.assertFalse(pool.images.flags.writeable)

    def test_unknown_dataset_rejected(self):
        with self.assertRaises(ValueError):
            dataset_path("../CIFAR")

    def test_extra_archive_keys_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            np.savez(Path(directory) / "CIFAR.npz", Xtr=[], Str=[], Xts=[], Yts=[], extra=[])
            with self.assertRaisesRegex(ValueError, "exactly keys"):
                load_noisy_pool("CIFAR", directory)


class ReproducibilityTests(unittest.TestCase):
    def test_hash_independent_of_config_key_order(self):
        self.assertEqual(config_hash({"a": 1, "b": 2}), config_hash({"b": 2, "a": 1}))
        self.assertNotEqual(config_hash({"a": 1}), config_hash({"a": 2}))

    def test_resume_checks_identity_completion_and_artifact_content(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            identity = {"config": "a", "source": "b", "data": "c"}
            for name in ("audit.json", "training_grid.png"):
                (destination / name).write_bytes(b"original")
            record = {"identity": identity, "status": "completed_data_smoke",
                      "artifacts": {name: sha256_file(destination / name) for name in ("audit.json", "training_grid.png")}}
            write_json(destination / "run.json", record)
            self.assertTrue(reusable_record(destination, identity))
            with self.assertRaisesRegex(ValueError, "different config"):
                reusable_record(destination, {"config": "changed"})
            (destination / "audit.json").write_bytes(b"tampered")
            self.assertFalse(reusable_record(destination, identity))
            record["status"] = "failed_data_smoke"
            write_json(destination / "run.json", record)
            self.assertFalse(reusable_record(destination, identity))

    def test_draft_final_plan_has_sixty_jobs(self):
        config = read_config(ROOT / "algorithm/configs/final.json")
        self.assertFalse(config["final_ready"])
        self.assertEqual(len(config["datasets"]) * len(config["methods"]) * len(config["split_seeds"]), 60)

    def test_duplicate_final_seeds_rejected(self):
        config = json.loads((ROOT / "algorithm/configs/final.json").read_text())
        config["split_seeds"] = [0] * 10
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, "distinct"):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
