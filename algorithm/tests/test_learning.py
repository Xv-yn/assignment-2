"""Numerical loss, transition, split and model regressions."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from algorithm.data import ROOT
from algorithm.losses import forward_loss, gce_loss
from algorithm.models import build_model, DeterministicAdaptivePool
from algorithm.run_experiments import read_config
from algorithm.training import random_split_indices, seed_everything, clean_metrics, fit
from algorithm.transition import KNOWN, estimate_anchor, validate_transition
from algorithm.experiments import verified_complete
from algorithm.provenance import write_json, sha256_file
from unittest.mock import patch
from torch.utils.data import DataLoader, TensorDataset


class LossTests(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[1.2, -.5, .4], [-.2, .7, 1.5]], requires_grad=True)
        self.labels = torch.tensor([0, 2])

    def test_forward_identity_equals_cross_entropy(self):
        torch.testing.assert_close(forward_loss(self.logits, self.labels, torch.eye(3)), F.cross_entropy(self.logits, self.labels))

    def test_forward_orientation_matches_hand_calculation(self):
        matrix = torch.tensor(KNOWN["FashionMNIST0.3"], dtype=torch.float32)
        probabilities = self.logits.softmax(1)
        expected = -torch.stack([.7 * probabilities[0, 0] + .3 * probabilities[0, 2],
                                 .3 * probabilities[1, 1] + .7 * probabilities[1, 2]]).log().mean()
        torch.testing.assert_close(forward_loss(self.logits, self.labels, matrix), expected)

    def test_gce_q_one_is_mae_half(self):
        expected = (1 - self.logits.softmax(1)[torch.arange(2), self.labels]).mean()
        torch.testing.assert_close(gce_loss(self.logits, self.labels, q=1), expected)

    def test_reweighted_gce_identity(self):
        torch.testing.assert_close(gce_loss(self.logits, self.labels),
                                   gce_loss(self.logits, self.labels, transition=torch.eye(3), reweight=True))

    def test_importance_weight_gradient_is_detached(self):
        matrix = torch.tensor(KNOWN["FashionMNIST0.3"], dtype=torch.float32)
        p = self.logits.softmax(1)
        weights = (p[torch.arange(2), self.labels] / (p @ matrix)[torch.arange(2), self.labels]).detach()
        expected = (((1 - p[torch.arange(2), self.labels].pow(.7)) / .7) * weights).mean()
        a = torch.autograd.grad(expected, self.logits, retain_graph=True)[0]
        b = torch.autograd.grad(gce_loss(self.logits, self.labels, transition=matrix, reweight=True), self.logits)[0]
        torch.testing.assert_close(a, b)

    def test_extreme_forward_logits_have_finite_gradients(self):
        logits = torch.tensor([[10000., -10000., 0.]], requires_grad=True)
        loss = forward_loss(logits, torch.tensor([2]), torch.tensor(KNOWN["FashionMNIST0.3"], dtype=torch.float32))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(loss.detach()), 9000)
        self.assertGreater(float(logits.grad.abs().sum()), 1)

    def test_invalid_q_rejected(self):
        for q in (0, -1, 1.1):
            with self.assertRaises(ValueError):
                gce_loss(self.logits, self.labels, q)


class TransitionTests(unittest.TestCase):
    def test_anchor_recovers_known_rows_at_pure_points(self):
        for matrix in KNOWN.values():
            posteriors = np.vstack([matrix, matrix.mean(0)])
            estimated, diagnostics = estimate_anchor(posteriors, top_fraction=0)
            np.testing.assert_allclose(estimated, matrix)
            self.assertEqual(diagnostics["anchor_local_indices"], [[0], [1], [2]])

    def test_anchor_averaging_and_quantile(self):
        p = np.array([[.9, .05, .05], [.8, .1, .1], [.1, .8, .1], [.1, .1, .8]])
        matrix, info = estimate_anchor(p, .5)
        np.testing.assert_allclose(matrix[0], [.85, .075, .075])
        matrix, info = estimate_anchor(p, 0, .9)
        self.assertNotIn(0, info["anchor_local_indices"][0])

    def test_singular_estimate_serializable(self):
        matrix, info = estimate_anchor(np.full((5, 3), 1 / 3))
        with tempfile.TemporaryDirectory() as directory:
            write_json(Path(directory) / "t.json", {"matrix": matrix.tolist(), "diagnostics": info})

    def test_invalid_transition_and_posteriors(self):
        for matrix in (np.eye(2), np.ones((3, 3)), np.full((3, 3), np.nan), -np.eye(3)):
            with self.assertRaises(ValueError):
                validate_transition(matrix)
        for probabilities in (np.empty((0, 3)), np.ones((5, 3)), np.ones((5, 2))):
            with self.assertRaises(ValueError):
                estimate_anchor(probabilities)


class TrainingTests(unittest.TestCase):
    def test_deterministic_pool_matches_adaptive_values_and_gradients(self):
        for height in (3, 4, 7, 8):
            x = torch.rand(2, 4, height, height, requires_grad=True)
            actual = DeterministicAdaptivePool()(x)
            expected = F.adaptive_avg_pool2d(x, (3, 3))
            torch.testing.assert_close(actual, expected)
            weights = torch.rand_like(actual)
            a = torch.autograd.grad((actual * weights).sum(), x, retain_graph=True)[0]
            b = torch.autograd.grad((expected * weights).sum(), x)[0]
            torch.testing.assert_close(a, b)

    def test_random_splits_are_exact_disjoint_reproducible_and_change(self):
        train, val = random_split_indices(15000, 7)
        self.assertEqual((len(train), len(val)), (12000, 3000))
        self.assertEqual(len(set(train) & set(val)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[train, val]), np.arange(15000))
        np.testing.assert_array_equal(train, random_split_indices(15000, 7)[0])
        self.assertFalse(np.array_equal(train, random_split_indices(15000, 8)[0]))

    def test_model_shapes_and_initialization(self):
        config = read_config(ROOT / "algorithm/configs/training_smoke.json")
        for name in ("FashionMNIST0.3", "CIFAR"):
            seed_everything(10)
            model = build_model(name, config["architecture"])
            seed_everything(10)
            second = build_model(name, config["architecture"])
            for a, b in zip(model.parameters(), second.parameters()):
                torch.testing.assert_close(a, b)
            shape = (2, 3, 32, 32) if name == "CIFAR" else (2, 1, 28, 28)
            self.assertEqual(model(torch.rand(shape)).shape, (2, 3))

    def test_checkpoint_selected_by_noisy_validation_nll(self):
        config = read_config(ROOT / "algorithm/configs/training_smoke.json")
        config.update(epochs=3, patience=3)
        loader = DataLoader(TensorDataset(torch.rand(6, 4), torch.tensor([0, 1, 2, 0, 1, 2])), batch_size=3)
        states = []
        def mock_evaluate(model, *args):
            states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
            return {"nll": [1., .5, 2.][len(states)-1], "accuracy": .3}
        with tempfile.TemporaryDirectory() as directory:
            with patch("algorithm.training.build_model", side_effect=lambda *args: torch.nn.Linear(4, 3)), patch("algorithm.training.evaluate", side_effect=mock_evaluate):
                model, info = fit("CIFAR", config, loader, loader, "cpu", 1, "ce", None, Path(directory))
            self.assertEqual(info["best_epoch"], 2)
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, states[1][key])
            checkpoint = torch.load(Path(directory) / "best.pt", weights_only=True)
            self.assertEqual(checkpoint["best_epoch"], 2)

    def test_clean_metrics_top1_and_confusion(self):
        result = clean_metrics({"probabilities": np.eye(3)[[0, 1, 2, 1]], "targets": np.array([0, 1, 2, 0])})
        self.assertEqual(result["test_accuracy"], .75)
        self.assertEqual(result["test_count"], 4)
        self.assertEqual(result["confusion_matrix"], [[1, 1, 0], [0, 1, 0], [0, 0, 1]])

    def test_resume_requires_artifact_integrity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metrics.json").write_text("{}")
            record = {"identity": {"seed": 1}, "status": "completed_training", "artifacts": {"metrics.json": sha256_file(root / "metrics.json")}}
            write_json(root / "run.json", record)
            self.assertTrue(verified_complete(root, {"seed": 1}, "completed_training"))
            (root / "metrics.json").write_text("changed")
            self.assertFalse(verified_complete(root, {"seed": 1}, "completed_training"))
            with self.assertRaises(ValueError):
                verified_complete(root, {"seed": 2}, "completed_training")

    def test_final_config_rejects_under_ten_splits(self):
        config = read_config(ROOT / "algorithm/configs/final.json")
        config["split_seeds"] = [0]
        config["initialization_seeds"] = [1000]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_json(path, config)
            with self.assertRaisesRegex(ValueError, "at least ten"):
                read_config(path)


if __name__ == "__main__":
    unittest.main()
