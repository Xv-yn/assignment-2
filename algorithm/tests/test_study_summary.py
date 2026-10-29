"""Study reports must derive their claims from the supplied records."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from algorithm.summarize_transition_v3 import METHODS, summarize


class StudySummaryTests(unittest.TestCase):
    def test_strong_records_do_not_inherit_old_weak_predictor_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('FashionMNIST0.3', 'FashionMNIST0.6', 'CIFAR'):
                folder = root / name
                folder.mkdir()
                for seed in range(10):
                    estimators = {method: {'T_hat': np.eye(3).tolist(),
                                           'validation_known_T': {'mae': 0.0}}
                                  for method in METHODS}
                    record = {'dataset': name, 'seed': seed, 'information_gain_nats': .1,
                              'noisy_validation_accuracy': .8, 'estimators': estimators}
                    # Force a mix of selected Fashion methods, preserving positive signal.
                    if name != 'CIFAR' and seed == 0:
                        record['information_gain_nats'] = .02
                    (folder / f'seed{seed}.json').write_text(json.dumps(record))
            output = summarize(root, root / 'summary')
            report = (root / 'summary/TRANSITION_V3_REPORT.md').read_text()
            self.assertEqual(output['CIFAR']['ill_conditioned_splits'], 0)
            self.assertEqual(output['CIFAR']['low_information_gain_splits'], 0)
            self.assertIn('per-seed mixed', report)
            self.assertNotIn('0.0044', report)
            self.assertNotIn('LightGBM', report)
            self.assertNotIn('ill-conditioned', output['CIFAR']['reason_not_validated'])


if __name__ == '__main__':
    unittest.main()
