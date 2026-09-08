import unittest

import numpy as np
import pandas as pd

from optimization.ga_isolation_forest import (
    Chromosome,
    TRAINING_FEATURE_COLUMNS,
    k_fold_score_chromosome,
)


class KFoldScoreChromosomeTest(unittest.TestCase):
    def setUp(self):
        # A small, deterministic synthetic dataset: mostly-normal rows drawn
        # from one cluster plus a minority of clear outlier rows, so
        # IsolationForest has something real to separate rather than pure
        # noise -- this test is about the k-fold plumbing (does it return a
        # populated mean/std/fold_scores), not about tuning for a specific
        # score.
        rng = np.random.default_rng(42)
        n_normal, n_anomaly = 180, 20

        normal = pd.DataFrame(
            {column: rng.normal(loc=0.0, scale=1.0, size=n_normal) for column in TRAINING_FEATURE_COLUMNS}
        )
        anomaly = pd.DataFrame(
            {column: rng.normal(loc=8.0, scale=1.0, size=n_anomaly) for column in TRAINING_FEATURE_COLUMNS}
        )

        self.X = pd.concat([normal, anomaly], ignore_index=True)
        self.y = pd.Series(["Normal"] * n_normal + ["Anomaly"] * n_anomaly)

        self.chromosome = Chromosome(
            feature_mask=[1] * len(TRAINING_FEATURE_COLUMNS),
            n_estimators=50,
            max_samples=0.8,
            max_features=1.0,
            contamination=0.1,
            bootstrap=False,
        )

    def test_returns_populated_mean_and_std(self):
        score = k_fold_score_chromosome(self.chromosome, self.X, self.y, k=5, random_state=42)

        self.assertIsInstance(score.mean, float)
        self.assertIsInstance(score.std, float)
        self.assertEqual(len(score.fold_scores), 5)
        self.assertTrue(all(isinstance(value, float) for value in score.fold_scores))
        self.assertTrue(0.0 <= score.mean <= 1.0)
        self.assertGreaterEqual(score.std, 0.0)
        # std computed from the same fold_scores mean/std reports
        self.assertAlmostEqual(score.mean, float(np.mean(score.fold_scores)), places=9)
        self.assertAlmostEqual(score.std, float(np.std(score.fold_scores)), places=9)

    def test_empty_feature_mask_short_circuits_without_populated_fold_scores(self):
        empty_chromosome = Chromosome(
            feature_mask=[0] * len(TRAINING_FEATURE_COLUMNS),
            n_estimators=50,
            max_samples=0.8,
            max_features=1.0,
            contamination=0.1,
        )
        score = k_fold_score_chromosome(empty_chromosome, self.X, self.y, k=5, random_state=42)

        self.assertEqual(score.mean, 0.0)
        self.assertEqual(score.std, 0.0)
        self.assertEqual(score.fold_scores, [0.0] * 5)


if __name__ == "__main__":
    unittest.main()
