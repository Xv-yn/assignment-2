"""Shared CLI and preserved pilot imports, without writing experiment files."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from algorithm.data import ROOT
from algorithm import run_experiments, run_pilots


class SharedCliTests(unittest.TestCase):
    def capture(self, entry, *arguments):
        stream = io.StringIO()
        with redirect_stdout(stream):
            entry(list(arguments))
        return json.loads(stream.getvalue())

    def assert_cli_error(self, entry, *arguments):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            entry(list(arguments))
        self.assertEqual(error.exception.code, 2)

    def test_pilot_helpers_are_canonical_reexports(self):
        for name in ("resolve_path", "read_config", "reusable_record", "validate_pilot_config"):
            self.assertIs(getattr(run_pilots, name), getattr(run_experiments, name))

    def test_both_entry_points_plan_the_same_pilot_jobs(self):
        arguments = ("--config", "algorithm/configs/pilot.json", "--plan")
        unified = self.capture(run_experiments.main, *arguments)
        compatibility = self.capture(run_pilots.main, *arguments)
        self.assertEqual(unified["jobs"], compatibility["jobs"])
        self.assertEqual(unified["job_count"], 3)
        self.assertEqual(compatibility["stage"], "training_pilot")

    def test_both_entry_points_use_the_same_audit_plan(self):
        arguments = ("--config", "algorithm/configs/smoke.json", "--dataset", "CIFAR", "--plan")
        self.assertEqual(self.capture(run_experiments.main, *arguments),
                         self.capture(run_pilots.main, *arguments))

    def test_pilot_routes_to_original_backend_with_overrides(self):
        for entry in (run_experiments.main, run_pilots.main):
            with patch("algorithm.train.run_training_job") as run:
                entry(["--config", "algorithm/configs/pilot.json", "--dataset", "FashionMNIST0.3",
                       "--seed", "0", "--resume", "--results-dir", "results/test-cli",
                       "--data-dir", "../datasets"])
            run.assert_called_once()
            self.assertEqual(run.call_args.args[1:], ("FashionMNIST0.3", "gce", 0))
            self.assertEqual(Path(run.call_args.args[0]["results_dir"]), Path("results/test-cli"))
            self.assertEqual(run.call_args.args[0]["data_dir"], str(Path("../datasets").resolve()))
            self.assertTrue(run.call_args.kwargs["resume"])

    def test_pilot_validation_rules_are_preserved(self):
        base = json.loads((ROOT / "algorithm/configs/pilot.json").read_text(encoding="utf-8"))
        changes = ({"split_seeds": []}, {"split_seeds": [0, 0]},
                   {"initialization_seeds": [-1]}, {"shuffle_seeds": [True]},
                   {"methods": ["gce", "gce"]}, {"methods": ["unknown"]},
                   {"test_evaluation": "after_validation_selection"}, {"min_delta": -1},
                   {"num_threads": 0}, {"train_fraction": .9})
        for change in changes:
            for reader in (run_experiments.read_config, run_pilots.read_config):
                with self.subTest(change=change), patch.object(Path, "read_text", return_value=json.dumps(base | change)):
                    with self.assertRaises(ValueError):
                        reader(Path("unused.json"))

    def test_formal_dispatch_remains_on_formal_backend(self):
        with patch("algorithm.experiments.run_training") as run:
            run_experiments.main(["--config", "algorithm/configs/final.json", "--dataset", "CIFAR",
                                  "--method", "forward", "--seed", "0"])
        run.assert_called_once()
        self.assertEqual(run.call_args.args[1:4], (["CIFAR"], ["forward"], [0]))

    def test_legacy_entry_cannot_accidentally_start_formal_training(self):
        with patch("algorithm.experiments.run_training") as run:
            self.assert_cli_error(run_pilots.main, "--config", "algorithm/configs/final.json", "--device", "cpu")
            run.assert_not_called()
        plan = self.capture(run_pilots.main, "--config", "algorithm/configs/final.json", "--device", "cpu", "--plan")
        self.assertEqual(plan["job_count"], 60)

    def test_pilot_gpu_and_unknown_method_fail_before_training(self):
        with patch("algorithm.train.run_training_job") as run:
            for entry in (run_experiments.main, run_pilots.main):
                self.assert_cli_error(entry, "--config", "algorithm/configs/pilot.json", "--device", "cuda")
                self.assert_cli_error(entry, "--config", "algorithm/configs/pilot.json", "--method", "forward")
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
