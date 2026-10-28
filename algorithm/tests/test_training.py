"""Mathematical, partition-leakage and checkpoint-selection regressions."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from algorithm.data import CleanTest, NoisyPool, ROOT
from algorithm.pilot_losses import (KNOWN_TRANSITIONS, per_example_loss, validate_transition)
from algorithm.pilot_models import SmallCNN
from algorithm.provenance import sha256_file
from algorithm.run_pilots import read_config
from algorithm.splits import Holdout, random_holdout, save_holdout, validate_holdout
from algorithm.train import (NoisyPartition, make_loaders, reusable_training_record,
                             run_epoch, run_training_job)


class SplitTests(unittest.TestCase):
    def test_required_counts_distinct_memberships_and_independent_rng(self):
        for count in (18000, 15000):
            memberships = set()
            for seed in range(10):
                split = random_holdout(count, seed)
                self.assertEqual(len(split.train), count * 4 // 5)
                self.assertEqual(len(split.validation), count // 5)
                self.assertEqual(len(np.intersect1d(split.train, split.validation)), 0)
                memberships.add(np.sort(split.train).tobytes())
                np.random.seed(99)
                np.random.random(100)
                torch.manual_seed(77)
                np.testing.assert_array_equal(split.train, random_holdout(count, seed).train)
            self.assertEqual(len(memberships), 10)

    def test_overlapping_incomplete_and_wrongly_typed_splits_rejected(self):
        for split in (Holdout(np.arange(8), np.array([7, 9])),
                      Holdout(np.arange(7), np.array([8, 9])),
                      Holdout(np.arange(8, dtype=float), np.array([8, 9]))):
            with self.assertRaises(ValueError):
                validate_holdout(split, 10)
        for count, seed in ((11, 0), (0, 0), (10, -1), (10, True)):
            with self.assertRaises(ValueError):
                random_holdout(count, seed)

    def test_saved_split_reused_and_tampering_or_changed_data_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split, path = save_holdout(root, "CIFAR", 15000, 0, "data-a")
            original_hash = sha256_file(path)
            again, _ = save_holdout(root, "CIFAR", 15000, 0, "data-a")
            self.assertEqual(original_hash, sha256_file(path))
            np.testing.assert_array_equal(split.train, again.train)
            with self.assertRaisesRegex(ValueError, "provenance"):
                save_holdout(root, "CIFAR", 15000, 0, "data-b")
            path.write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "provenance"):
                save_holdout(root, "CIFAR", 15000, 0, "data-a")


class LossTests(unittest.TestCase):
    def test_forward_identity_and_asymmetric_orientation(self):
        logits = torch.tensor([[0.2, 1.4, -0.5], [1.1, -0.7, 0.3]], dtype=torch.float64)
        labels = torch.tensor([0, 2])
        expected = F.cross_entropy(logits, labels, reduction="none")
        torch.testing.assert_close(per_example_loss(logits, labels, "forward", transition=torch.eye(3)), expected)
        transition = torch.tensor(KNOWN_TRANSITIONS["FashionMNIST0.3"], dtype=torch.float64)
        expected = -(logits.softmax(1) @ transition)[torch.arange(2), labels].log()
        torch.testing.assert_close(per_example_loss(logits, labels, "forward", transition=transition), expected)

    def test_gce_formula_limits_and_finite_extreme_gradients(self):
        logits = torch.tensor([[0.2, 1.4, -0.5]], dtype=torch.float64)
        labels = torch.tensor([1])
        p = logits.softmax(1)[0, 1]
        torch.testing.assert_close(per_example_loss(logits, labels, "gce", gamma=0.7), ((1 - p**0.7) / 0.7).reshape(1))
        torch.testing.assert_close(per_example_loss(logits, labels, "gce", gamma=1), (1-p).reshape(1))
        torch.testing.assert_close(per_example_loss(logits, labels, "gce", gamma=1e-8), F.cross_entropy(logits, labels, reduction="none"))
        for method in ("gce", "forward"):
            extreme = torch.tensor([[1000., -1000., 0.]], requires_grad=True)
            transition = torch.tensor(KNOWN_TRANSITIONS["FashionMNIST0.3"])
            loss = per_example_loss(extreme, torch.tensor([1]), method, transition=transition).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(extreme.grad).all())

    def test_invalid_matrix_and_gamma_fail(self):
        for matrix in (torch.ones(3, 3), torch.eye(2), torch.full((3, 3), float("nan"))):
            with self.assertRaises(ValueError):
                validate_transition(matrix)
        with self.assertRaises(ValueError):
            per_example_loss(torch.ones(1, 3), torch.tensor([0]), "gce", gamma=0)


def tiny_pool():
    rng = np.random.default_rng(42)
    return NoisyPool("FashionMNIST0.3", rng.integers(0, 256, (30, 784), dtype=np.uint8),
                     np.resize(np.array([0, 1, 2], dtype=np.uint8), 30))


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_loaders_expose_only_their_partition_and_repeat_shuffling(self):
        pool, split = tiny_pool(), random_holdout(30, 2)
        first, validation = make_loaders(pool, split, batch_size=7, shuffle_seed=2000)
        second, _ = make_loaders(pool, split, batch_size=7, shuffle_seed=2000)
        np.testing.assert_array_equal(first.dataset.indices, split.train)
        np.testing.assert_array_equal(validation.dataset.indices, split.validation)
        for (images, labels), (again, repeated_labels) in zip(first, second):
            torch.testing.assert_close(images, again)
            torch.testing.assert_close(labels, repeated_labels)
            self.assertEqual(images.dtype, torch.float32)
            self.assertEqual(labels.dtype, torch.int64)
        self.assertEqual(sum(len(y) for _, y in validation), 6)
        test = CleanTest(pool.name, pool.images, pool.noisy_labels)
        with self.assertRaises(TypeError):
            NoisyPartition(test, split.train)

    def test_training_updates_weights_and_validation_does_not(self):
        train, validation = make_loaders(tiny_pool(), random_holdout(30, 0), batch_size=7, shuffle_seed=3)
        model = SmallCNN(1, [4, 8])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = copy.deepcopy(model.state_dict())
        training = run_epoch(model, train, "gce", gamma=0.7, transition=None, optimizer=optimizer)
        after_training = copy.deepcopy(model.state_dict())
        self.assertTrue(any(not torch.equal(before[k], after_training[k]) for k in before))
        result = run_epoch(model, validation, "gce", gamma=0.7, transition=None)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, after_training[key])
        self.assertEqual(training["examples"], 24)
        self.assertEqual(result["examples"], 6)
        self.assertFalse(model.training)

    def test_selection_early_stopping_failure_and_resume_artifacts(self):
        config = read_config(ROOT / "algorithm/configs/pilot.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            supplied = root / "supplied.npz"
            supplied.write_bytes(b"mock noisy pool content")
            config.update(results_dir=str(root / "results"), splits_dir=str(root / "splits"),
                          epochs=6, patience=2, architecture={"name": "small_cnn", "channels": [4, 8], "num_classes": 3})
            pool = tiny_pool()
            validation_losses = iter([0.8, 0.6, 0.6, 0.9])

            def measured_epoch(model, loader, method, *, gamma, transition, optimizer=None):
                if optimizer is not None:
                    return run_epoch(model, loader, method, gamma=gamma, transition=transition, optimizer=optimizer)
                return {"loss": next(validation_losses), "noisy_accuracy_percent": 50., "examples": 6}

            with patch("algorithm.train.dataset_path", return_value=supplied), patch("algorithm.train.load_noisy_pool", return_value=pool), patch("algorithm.train.run_epoch", side_effect=measured_epoch):
                record = run_training_job(config, pool.name, "gce", 0)
                self.assertEqual(record["selected_epoch"], 2)
                self.assertEqual(record["epochs_completed"], 4)
                self.assertFalse(record["clean_test_accessed"])
                destination = root / "results" / pool.name / "gce/split_0"
                saved = torch.load(destination / "best.pt", weights_only=True)
                self.assertEqual(saved["epoch"], 2)
                self.assertTrue(reusable_training_record(destination, record["identity"]))
                run_training_job(config, pool.name, "gce", 0, resume=True)
                (destination / "history.json").write_text("[]")
                self.assertFalse(reusable_training_record(destination, record["identity"]))
                with patch("algorithm.train.run_epoch", side_effect=FloatingPointError("bad loss")):
                    with self.assertRaises(FloatingPointError):
                        run_training_job(config, pool.name, "gce", 0, resume=True)
                failed = json.loads((destination / "run.json").read_text())
                self.assertEqual(failed["status"], "failed_training_pilot")

    def test_cifar_forward_refuses_before_loading_data(self):
        with patch("algorithm.train.load_noisy_pool", side_effect=AssertionError("should not load")):
            with self.assertRaisesRegex(ValueError, "transition estimator"):
                run_training_job({}, "CIFAR", "forward", 0)

    def test_invalid_pilot_config_cannot_enable_test_or_change_split(self):
        config = json.loads((ROOT / "algorithm/configs/pilot.json").read_text())
        for change in ({"train_fraction": 1.0}, {"test_evaluation": "enabled"},
                       {"shuffle_seeds": [-1]}, {"learning_rate": float("nan")}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "config.json"
                path.write_text(json.dumps(config | change))
                with self.assertRaises(ValueError):
                    read_config(path)


if __name__ == "__main__":
    unittest.main()
