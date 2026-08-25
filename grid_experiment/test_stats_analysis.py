import unittest

import numpy as np

from grid_experiment.stats_analysis import (
    bh_adjust,
    bootstrap_mean_ci,
    probability_of,
    wilcoxon_p,
)


class StatsAnalysisTests(unittest.TestCase):
    def test_probability_of_baseline_class_across_flip(self):
        self.assertAlmostEqual(probability_of("HP", "HP", 0.8), 0.8)
        self.assertAlmostEqual(probability_of("HP", "SSA", 0.8), 0.2)

    def test_bh_adjust_known_values(self):
        np.testing.assert_allclose(bh_adjust([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])

    def test_bootstrap_is_deterministic(self):
        self.assertEqual(bootstrap_mean_ci([1, 2, 3]), bootstrap_mean_ci([1, 2, 3]))

    def test_all_zero_wilcoxon(self):
        self.assertEqual(wilcoxon_p([0, 0, 0]), 1.0)


if __name__ == "__main__":
    unittest.main()
