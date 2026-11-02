"""Standard GCE is matrix-free in training and checkpoint selection."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

from algorithm.data import ROOT
from algorithm.losses import gce_loss
from algorithm.run_experiments import read_config
from algorithm.training import evaluate, fit, gce_uses_transition, selection_metric


def tiny_loader():
    torch.manual_seed(0)
    return DataLoader(TensorDataset(torch.rand(6, 4), torch.tensor([0, 1, 2, 0, 1, 2])), batch_size=3)


def linear_model(*args):
    return torch.nn.Linear(4, 3)


class StandardGCETests(unittest.TestCase):
    def setUp(self):
        self.config = read_config(ROOT / "algorithm/configs/training_smoke.json")
        self.config.update(epochs=3, patience=3)

    def test_configs_use_standard_gce(self):
        for name in ("final.json", "training_smoke.json"):
            params = read_config(ROOT / "algorithm/configs" / name)["method_parameters"]["gce"]
            self.assertFalse(gce_uses_transition(params), name)

    def test_selection_metric_names(self):
        params = {"gamma": .7, "reweight": False, "warmup_epochs": 0}
        self.assertEqual(selection_metric("gce", params, None), "noisy_gce")
        self.assertEqual(selection_metric("forward", params, torch.eye(3)), "noisy_forward_nll")
        self.assertEqual(selection_metric("ce", params, None), "noisy_ce")

    def test_evaluate_reports_unweighted_gce(self):
        model = torch.nn.Linear(4, 3)
        loader = tiny_loader()
        images, labels = next(iter(DataLoader(loader.dataset, batch_size=6)))
        with torch.no_grad():
            expected = float(gce_loss(model(images), labels, .7))
        self.assertAlmostEqual(evaluate(model, loader, "cpu", None, .7)["gce"], expected, places=6)

    def test_standard_gce_never_touches_transition(self):
        loader = tiny_loader()
        with tempfile.TemporaryDirectory() as directory, \
                patch("algorithm.training.build_model", side_effect=linear_model), \
                patch("algorithm.training.forward_loss", side_effect=AssertionError("forward loss used")):
            _, info = fit("CIFAR", self.config, loader, loader, "cpu", 1, "gce", None, Path(directory))
            self.assertEqual(info["selection_metric"], "noisy_gce")
            checkpoint = torch.load(Path(directory) / "best.pt", weights_only=True)
            self.assertIsNone(checkpoint["transition"])

    def test_standard_gce_rejects_a_matrix(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("algorithm.training.build_model", side_effect=linear_model):
            with self.assertRaises(ValueError):
                fit("CIFAR", self.config, tiny_loader(), tiny_loader(), "cpu", 1, "gce",
                    torch.eye(3).tolist(), Path(directory))

    def test_gce_checkpoint_follows_validation_gce_not_nll(self):
        # NLL prefers epoch 1, GCE prefers epoch 3: selection must follow GCE.
        scores = [{"nll": .1, "gce": .9}, {"nll": .5, "gce": .8}, {"nll": .9, "gce": .2}]
        calls = []
        def mock_evaluate(model, loader, device, transition=None, gce_gamma=None):
            self.assertIsNone(transition)
            self.assertEqual(gce_gamma, .7)
            calls.append(1)
            return dict(scores[len(calls) - 1], accuracy=.3)
        with tempfile.TemporaryDirectory() as directory, \
                patch("algorithm.training.build_model", side_effect=linear_model), \
                patch("algorithm.training.evaluate", side_effect=mock_evaluate):
            _, info = fit("CIFAR", self.config, tiny_loader(), tiny_loader(), "cpu", 1, "gce", None, Path(directory))
        self.assertEqual(info["best_epoch"], 3)
        self.assertAlmostEqual(info["validation_selection_loss"], .2)
        self.assertAlmostEqual(info["validation_noisy_nll"], .9)

    def test_optional_variant_still_requires_matrix(self):
        self.config["method_parameters"]["gce"].update(reweight=True, warmup_epochs=1)
        with tempfile.TemporaryDirectory() as directory, \
                patch("algorithm.training.build_model", side_effect=linear_model):
            with self.assertRaises(ValueError):
                fit("CIFAR", self.config, tiny_loader(), tiny_loader(), "cpu", 1, "gce", None, Path(directory))


if __name__ == "__main__":
    unittest.main()
