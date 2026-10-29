"""Pure-Numpy tests; no dataset/clean labels/GPU required."""
import unittest
import numpy as np
from algorithm.transition_estimation import (
    KNOWN_TRANSITIONS, calibrate_temperature, estimate_transition_from_probabilities,
    known_matrix_diagnostics, split_noisy_pool, _softmax,
)


class TransitionEstimationTests(unittest.TestCase):
    def test_exact_pure_anchors_recover_known_asymmetric_matrix(self):
        t = KNOWN_TRANSITIONS["FashionMNIST0.3"]
        p_clean = np.repeat(np.eye(3), 40, axis=0)
        q_noisy = p_clean @ t
        estimated, meta = estimate_transition_from_probabilities(q_noisy, top_k=10)
        np.testing.assert_allclose(estimated, t, atol=1e-12)
        self.assertEqual(sorted(meta), ["0", "1", "2"])

    def test_near_uniform_transition(self):
        t = KNOWN_TRANSITIONS["FashionMNIST0.6"]
        estimate, _ = estimate_transition_from_probabilities(np.repeat(t, 40, axis=0), top_k=10)
        np.testing.assert_allclose(estimate, t, atol=1e-12)
        self.assertAlmostEqual(known_matrix_diagnostics("FashionMNIST0.6", estimate)["mae"], 0.)

    def test_probabilities_well_formed_and_rows_stochastic(self):
        rng = np.random.default_rng(20)
        prob = rng.dirichlet([3, 2, 1], size=100)
        estimate, _ = estimate_transition_from_probabilities(prob, top_k=15)
        self.assertEqual(estimate.shape, (3, 3))
        self.assertTrue(np.isfinite(estimate).all())
        self.assertTrue(np.all(estimate >= 0))
        np.testing.assert_allclose(estimate.sum(axis=1), np.ones(3))

    def test_bad_probability_inputs_rejected(self):
        bad = [np.zeros((4, 3)), np.full((4, 3), np.nan),
               np.array([[1, -0.1, 0.1]] * 4), np.ones((4, 4))]
        for case in bad:
            with self.assertRaises(ValueError):
                estimate_transition_from_probabilities(case)
        with self.assertRaises(ValueError):
            estimate_transition_from_probabilities(np.ones((4, 3))/3, top_k=5)

    def test_calibration_uses_noisy_labels_only(self):
        rng = np.random.default_rng(9)
        logits = rng.normal(size=(300, 3)) * 5
        labels = rng.integers(0, 3, size=300)
        temp = calibrate_temperature(logits, labels)
        self.assertGreater(temp, 1)
        calibrated = _softmax(logits / temp)
        self.assertTrue(np.allclose(calibrated.sum(axis=1), 1))

    def test_splits_reproducible_disjoint(self):
        a, b = split_noisy_pool(18000, 123)
        a2, b2 = split_noisy_pool(18000, 123)
        self.assertEqual((len(a), len(b)), (14400, 3600))
        np.testing.assert_array_equal(a, a2)
        np.testing.assert_array_equal(b, b2)
        self.assertEqual(len(set(a).intersection(set(b))), 0)
        c, _ = split_noisy_pool(18000, 124)
        self.assertFalse(np.array_equal(a, c))

    def test_unknown_matrix_has_no_ground_truth_diagnostic(self):
        self.assertIsNone(known_matrix_diagnostics("CIFAR", np.eye(3)))


if __name__ == "__main__":
    unittest.main()
