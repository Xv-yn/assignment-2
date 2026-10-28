"""Synthetic temporary fixtures exercise final-grid and artifact safeguards."""

import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from algorithm.data import ROOT
from algorithm.package_submission import package
from algorithm.provenance import config_hash, sha256_file, write_json
from algorithm.results import aggregate, mean_std
from algorithm.run_experiments import read_config
from algorithm.transition import KNOWN


def fixture(root, seeds=range(10), debug=False):
    config = read_config(ROOT / "algorithm/configs/final.json")
    config["datasets"] = ["FashionMNIST0.3"]
    config["debug"] = debug
    def save(directory, identity, status, names):
        write_json(directory / "run.json", {"status": status, "identity": identity, "config": config,
                   "artifacts": {name: sha256_file(directory / name) for name in names}})
    for seed in seeds:
        base = root / "FashionMNIST0.3" / f"seed_{seed:03d}"
        identity = {"dataset": "FashionMNIST0.3", "split_seed": seed,
                    "source_sha256": "temporary-test-fixture", "config_sha256": config_hash(config), "environment": {"test": True}}
        estimator = base / "estimator"
        estimator.mkdir(parents=True)
        write_json(estimator / "transition.json", {"estimated_transition": KNOWN["FashionMNIST0.3"].tolist(),
                   "mean_absolute_error": 0, "frobenius_error": 0})
        write_json(estimator / "history.json", {})
        (estimator / "best.pt").write_bytes(b"unit test placeholder")
        save(estimator, dict(identity, component="estimator"), "completed_estimator", ["transition.json", "history.json", "best.pt"])
        for method in ("forward", "gce"):
            directory = base / method
            directory.mkdir()
            predictions = np.eye(3)[[0, 1, 2, 0]]
            targets = np.array([0, 1, 2, 2 if method == "forward" else 0])
            accuracy = float((predictions.argmax(1) == targets).mean())
            write_json(directory / "metrics.json", {"dataset": "FashionMNIST0.3", "method": method,
                       "split_seed": seed, "debug": debug, "test_accuracy": accuracy,
                       "test_top1_percent": 100 * accuracy})
            write_json(directory / "history.json", {"epochs": [{"train_loss": 1, "validation_noisy_nll": 1}], "best_epoch": 1})
            (directory / "best.pt").write_bytes(b"unit test placeholder")
            np.savez(directory / "predictions.npz", probabilities=predictions, targets=targets)
            save(directory, dict(identity, component=method), "completed_training", ["best.pt", "history.json", "metrics.json", "predictions.npz"])


class ResultsTests(unittest.TestCase):
    def test_sample_standard_deviation_and_single_run(self):
        mean, sd = mean_std([1, 2, 3])
        self.assertEqual((mean, sd), (2, 1))
        self.assertIsNone(mean_std([1])[1])

    def test_final_grid_and_paired_difference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            summary = aggregate(root)
            self.assertTrue(summary["complete"])
            self.assertEqual([r["runs"] for r in summary["rows"]], [10, 10])
            self.assertEqual(summary["paired_comparisons"][0]["gce_minus_forward_mean_pp"], 25)
            self.assertEqual(summary["transition_estimates"][0]["runs"], 10)
            self.assertTrue((root / "accuracy_comparison.png").is_file())

    def test_partial_grid_is_provisional_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, seeds=[0])
            with self.assertRaisesRegex(ValueError, "full >=10"):
                aggregate(root)
            summary = aggregate(root, allow_incomplete=True)
            self.assertFalse(summary["complete"])
            self.assertEqual(len(summary["missing_jobs"]), 18)

    def test_debug_cannot_become_final(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, debug=True)
            with self.assertRaisesRegex(ValueError, "full >=10"):
                aggregate(root)

    def test_tampered_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, seeds=[0])
            path = root / "FashionMNIST0.3/seed_000/forward/metrics.json"
            path.write_text("{}")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                aggregate(root, allow_incomplete=True)

    def test_mixed_sources_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root, seeds=[0])
            path = root / "FashionMNIST0.3/seed_000/forward/run.json"
            record = json.loads(path.read_text())
            record["identity"]["source_sha256"] = "different"
            write_json(path, record)
            with self.assertRaisesRegex(ValueError, "different configs, sources"):
                aggregate(root, allow_incomplete=True)

    def test_package_has_empty_data_and_all_runtime_modules(self):
        with tempfile.TemporaryDirectory() as directory:
            path = package(Path(directory) / "source.zip")
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                self.assertIn("data/", names)
                self.assertIn("algorithm/experiments.py", names)
                self.assertIn("algorithm/configs/final.json", names)
                self.assertIn("requirements.txt", names)
                self.assertFalse(any(n.endswith((".npz", ".pt", ".pyc")) or "/.git/" in n for n in names))
                self.assertEqual([n for n in names if n.startswith("data/")], ["data/"])


if __name__ == "__main__":
    unittest.main()
