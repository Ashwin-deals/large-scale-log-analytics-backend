import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from smart_storage import baselines
from smart_storage.data import refit_on_rows
from smart_storage.optimizer import (
    GENE_NAMES,
    ScoringVariant,
    build_report,
    compare_to_baselines,
    decode_policy,
    run_ga,
    stratified_split,
)
from smart_storage.policy import (
    COLD,
    ERROR_WEIGHT,
    HOT,
    WARM,
    BlockTable,
    Normalizer,
    StoragePolicy,
    assign_tiers,
    evaluate_policy,
    importance,
    outcome_from_kept_mask,
    penalized_fitness,
)


def make_variant(frame, seed=42, warm_credit=0.5, population=10, generations=4, model_info=None):
    """A tuned ScoringVariant on `frame`, the way the script builds one."""
    tune_idx, holdout_idx = stratified_split(frame["true_label"], 0.5, seed)
    tf, hf = frame.iloc[tune_idx], frame.iloc[holdout_idx]
    norm = Normalizer.fit(tf["anomaly_score"], tf["event_frequency"], tf["block_size"])
    tune, holdout = BlockTable.from_frame(tf, norm), BlockTable.from_frame(hf, norm)
    _, result = run_ga(tune, population_size=population, num_generations=generations,
                       num_parents_mating=4, verbose=False, warm_credit=warm_credit)
    return ScoringVariant(model_info or {"version": 2, "model_path": "m.pkl"}, norm, tune, holdout, result)


def make_frame(n=2000, anomaly_rate=0.1, seed=0):
    """Synthetic blocks where the model's score is informative but imperfect,
    anomalies tend to be small, and event_frequency is heavy-tailed."""
    rng = np.random.default_rng(seed)
    is_anomaly = rng.random(n) < anomaly_rate
    score = np.where(is_anomaly, rng.normal(0.6, 0.25, n), rng.normal(0.2, 0.2, n))
    return pd.DataFrame(
        {
            "block_id": [f"blk_{i}" for i in range(n)],
            "anomaly_score": score,
            "predicted_label": np.where(score > 0.55, "Anomaly", "Normal"),
            "true_label": np.where(is_anomaly, "Anomaly", "Normal"),
            "event_frequency": rng.integers(2, 60, n),
            "block_size": rng.integers(0, 1000, n),
            "is_error": 0,
        }
    )


def manual_table(scores, ef, anomaly, flagged=None, is_error=None):
    """A BlockTable whose importance equals `scores` (all weight on the first
    part, pre-divided by the non-error share so the final importance is exact)."""
    n = len(scores)
    parts = np.zeros((3, n))
    parts[0] = np.asarray(scores, dtype=float) / (1.0 - ERROR_WEIGHT)
    return BlockTable(
        parts=parts,
        is_error=np.zeros(n) if is_error is None else np.asarray(is_error, dtype=float),
        event_frequency=np.asarray(ef, dtype=np.int64),
        is_anomaly=np.asarray(anomaly, dtype=bool),
        is_flagged=np.asarray(anomaly if flagged is None else flagged, dtype=bool),
        anomaly_score=np.asarray(scores, dtype=float),
    )


def anomaly_only_policy(t_hot, t_warm, f):
    return StoragePolicy(1.0, 0.0, 0.0, t_hot, t_warm, f)


class NormalizerTest(unittest.TestCase):
    def test_parts_are_in_unit_range_and_ordered(self):
        frame = make_frame()
        norm = Normalizer.fit(frame["anomaly_score"], frame["event_frequency"], frame["block_size"])
        parts = norm.parts(frame["anomaly_score"], frame["event_frequency"], frame["block_size"])
        self.assertEqual(parts.shape, (3, len(frame)))
        self.assertGreaterEqual(parts.min(), 0.0)
        self.assertLessEqual(parts.max(), 1.0)
        # Higher raw score -> higher normalized score.
        order = np.argsort(frame["anomaly_score"].to_numpy())
        self.assertTrue(np.all(np.diff(parts[0][order]) >= 0))

    def test_values_outside_the_fitted_range_clip(self):
        norm = Normalizer.fit([0.0, 1.0], [1, 10], [10, 20])
        parts = norm.parts([-5.0, 5.0], [1, 1000], [-1, 10**9])
        self.assertEqual(parts[0].tolist(), [0.0, 1.0])
        self.assertEqual(parts[1, 0], 0.0)
        self.assertEqual(parts[1, 1], 1.0)
        self.assertLess(parts[2, 0], 0.01)
        self.assertEqual(parts[2, 1], 1.0)

    def test_percentile_gives_tied_values_the_same_mid_rank(self):
        # Half the blocks share one size, as with the real 64MiB blocks.
        sizes = np.array([0] * 100 + [64] * 100)
        norm = Normalizer.fit(np.zeros(200), np.ones(200), sizes)
        percentile = norm.size_percentile([0, 64])
        self.assertAlmostEqual(percentile[0], 0.25, delta=0.01)
        self.assertAlmostEqual(percentile[1], 0.75, delta=0.01)

    def test_constant_column_does_not_divide_by_zero(self):
        norm = Normalizer.fit([3.0, 3.0], [5, 5], [7, 7])
        self.assertTrue(np.all(np.isfinite(norm.parts([3.0], [5], [7]))))

    def test_round_trips_through_json(self):
        frame = make_frame()
        norm = Normalizer.fit(frame["anomaly_score"], frame["event_frequency"], frame["block_size"])
        restored = Normalizer.from_dict(json.loads(json.dumps(norm.to_dict())))
        args = (frame["anomaly_score"], frame["event_frequency"], frame["block_size"])
        np.testing.assert_array_equal(norm.parts(*args), restored.parts(*args))


class PolicyTest(unittest.TestCase):
    def test_weights_are_rescaled_to_sum_to_one(self):
        policy = StoragePolicy(2.0, 1.0, 1.0, 0.5, 0.2, 0.1)
        self.assertAlmostEqual(policy.weights.sum(), 1.0)
        self.assertAlmostEqual(policy.weights[0], 0.5)

    def test_validation_rejects_bad_policies(self):
        bad = [
            StoragePolicy(1, 1, 1, 0.3, 0.3, 0.1),   # t_warm == t_hot
            StoragePolicy(1, 1, 1, 0.3, 0.5, 0.1),   # t_warm > t_hot
            StoragePolicy(1, 1, 1, 1.5, 0.5, 0.1),   # t_hot > 1
            StoragePolicy(1, 1, 1, 0.5, -0.1, 0.1),  # negative t_warm
            StoragePolicy(0, 0, 0, 0.5, 0.2, 0.1),   # no weight at all
            StoragePolicy(-1, 1, 1, 0.5, 0.2, 0.1),  # negative weight
            StoragePolicy(1, 1, 1, 0.5, 0.2, 0.0),   # f must be positive
            StoragePolicy(1, 1, 1, 0.5, 0.2, 1.5),   # f above 1 would keep more than the whole block
            StoragePolicy(float("nan"), 1, 1, 0.5, 0.2, 0.1),
        ]
        for policy in bad:
            with self.assertRaises(ValueError, msg=str(policy)):
                policy.validate()
        StoragePolicy(1, 1, 1, 0.5, 0.2, 0.1).validate()

    def test_round_trips_through_dict(self):
        policy = StoragePolicy(3.0, 1.0, 1.0, 0.6, 0.25, 0.2)
        restored = StoragePolicy.from_dict(json.loads(json.dumps(policy.to_dict())))
        np.testing.assert_allclose(restored.weights, policy.weights)
        self.assertEqual((restored.t_hot, restored.t_warm, restored.summary_fraction), (0.6, 0.25, 0.2))

    def test_error_blocks_get_the_fixed_error_share(self):
        table = manual_table([0.0, 0.0], [10, 10], [False, False], is_error=[0, 1])
        scores = importance(table, anomaly_only_policy(0.9, 0.1, 0.1))
        self.assertAlmostEqual(scores[0], 0.0)
        self.assertAlmostEqual(scores[1], ERROR_WEIGHT)

    def test_importance_is_the_weighted_sum_of_parts(self):
        table = BlockTable(
            parts=np.array([[1.0], [0.5], [0.0]]),
            is_error=np.zeros(1), event_frequency=np.array([10]),
            is_anomaly=np.zeros(1, bool), is_flagged=np.zeros(1, bool), anomaly_score=np.zeros(1),
        )
        policy = StoragePolicy(2.0, 1.0, 1.0, 0.5, 0.1, 0.1)  # normalized: .5 / .25 / .25
        expected = (1 - ERROR_WEIGHT) * (0.5 * 1.0 + 0.25 * 0.5 + 0.25 * 0.0)
        self.assertAlmostEqual(importance(table, policy)[0], expected)


class TierAccountingTest(unittest.TestCase):
    def test_boundaries_belong_to_the_higher_tier(self):
        policy = anomaly_only_policy(0.8, 0.4, 0.1)
        tiers = assign_tiers(np.array([0.8, 0.79, 0.4, 0.39, 1.0, 0.0]), policy)
        self.assertEqual(tiers.tolist(), [HOT, WARM, WARM, COLD, HOT, COLD])

    def test_lines_kept_per_tier(self):
        # Hot keeps all 100; warm keeps 1 metadata line + round(0.1 * 100) = 11
        # and 1 + round(0.1 * 4) = 1 (summary rounds to 0); cold keeps 1 whatever its size.
        table = manual_table(
            scores=[0.9, 0.5, 0.5, 0.1, 0.1],
            ef=[100, 100, 4, 50, 2],
            anomaly=[True, False, False, False, False],
        )
        outcome = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.1))
        self.assertEqual(outcome.tier_counts, {"HOT": 1, "WARM": 2, "COLD": 2})
        self.assertEqual(outcome.tier_lines, {"HOT": 100, "WARM": 12, "COLD": 2})
        self.assertEqual(outcome.total_lines, 256)
        self.assertEqual(outcome.kept_lines, 114)
        self.assertAlmostEqual(outcome.storage_saved_pct, 100 * (1 - 114 / 256))

    def test_summary_rounds_half_to_even_like_python(self):
        # f * ef = 2.5 -> round() gives 2, not 3.
        table = manual_table([0.5], [25], [False])
        outcome = evaluate_policy(table, anomaly_only_policy(0.9, 0.1, 0.1))
        self.assertEqual(outcome.tier_lines["WARM"], 1 + round(0.1 * 25))

    def test_retention_weights_hot_warm_and_cold(self):
        # Anomalies: one HOT, one WARM, one COLD.
        table = manual_table(
            scores=[0.9, 0.5, 0.1, 0.9],
            ef=[10, 10, 10, 10],
            anomaly=[True, True, True, False],
            flagged=[True, False, False, True],
        )
        outcome = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.5), warm_credit=0.5)
        self.assertAlmostEqual(outcome.full_retention, 1 / 3)
        self.assertAlmostEqual(outcome.coverage, 2 / 3)
        self.assertAlmostEqual(outcome.weighted_retention, (1 + 0.5) / 3)
        self.assertEqual(outcome.tier_anomalies, {"HOT": 1, "WARM": 1, "COLD": 1})
        self.assertAlmostEqual(outcome.flagged_retention, 1.0)

    def test_warm_credit_bounds_reduce_to_full_retention_and_coverage(self):
        # f = 0.3 keeps 3 of 10 lines, above the 0.25 reference, so WARM earns full credit.
        table = manual_table([0.9, 0.5, 0.5, 0.1], [10] * 4, [True, True, True, True])
        policy = anomaly_only_policy(0.8, 0.4, 0.3)
        none = evaluate_policy(table, policy, warm_credit=0.0)
        full = evaluate_policy(table, policy, warm_credit=1.0)
        self.assertAlmostEqual(none.weighted_retention, none.full_retention)
        self.assertAlmostEqual(full.weighted_retention, full.coverage)
        # Credit changes retention, never cost.
        self.assertEqual(none.kept_lines, full.kept_lines)

    def test_fitness_and_target_use_weighted_retention(self):
        table = manual_table([0.9, 0.5], [10, 10], [True, True])
        outcome = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.3), warm_credit=0.8)
        self.assertAlmostEqual(outcome.weighted_retention, 0.9)
        self.assertTrue(outcome.meets_target(0.9))
        self.assertFalse(outcome.meets_target(0.91))
        self.assertAlmostEqual(outcome.fitness(0.95, 1000.0), outcome.storage_saved_pct - 50.0)

    def test_warm_credit_lets_a_cheap_summary_replace_a_full_copy(self):
        # Two big anomalous blocks. All-HOT keeps 200 lines; demoting one to a
        # metadata line + 25-line summary (the 0.25 reference share) keeps 126 and, at
        # credit 0.8, still meets 0.9.
        table = manual_table([0.9, 0.5], [100, 100], [True, True])
        all_hot = evaluate_policy(table, anomaly_only_policy(0.4, 0.2, 0.25), warm_credit=0.8)
        one_warm = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.25), warm_credit=0.8)
        self.assertEqual((all_hot.kept_lines, one_warm.kept_lines), (200, 126))
        self.assertTrue(one_warm.meets_target(0.9))
        self.assertGreater(one_warm.fitness(0.9), all_hot.fitness(0.9))

    def test_warm_credit_is_proportional_to_the_share_kept(self):
        # One WARM anomaly of 100 lines; credit = 0.5 * min(1, summary / 25).
        table = manual_table([0.5], [100], [True])
        for f, expected in ((0.05, 0.5 * 5 / 25), (0.1, 0.5 * 10 / 25), (0.25, 0.5), (0.5, 0.5)):
            outcome = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, f), warm_credit=0.5)
            self.assertAlmostEqual(outcome.weighted_retention, expected, msg=f"f={f}")
            self.assertAlmostEqual(outcome.warm_credit_earned, expected)
        self.assertAlmostEqual(outcome.coverage, 1.0)
        self.assertAlmostEqual(outcome.full_retention, 0.0)

    def test_a_summary_that_costs_what_cold_costs_earns_nothing(self):
        # The old collapse: an 18-line block at f = 0.0565 kept 1 line -- the
        # COLD cost -- yet earned the flat 0.5. Now round(0.0565 * 18) = 1
        # summary line on top of the metadata line, earning 0.5 * 1 / 4.5;
        # and at f = 0.02 the summary rounds to 0, so it is COLD in all but name.
        table = manual_table([0.5], [18], [True])
        thin = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.0565), warm_credit=0.5)
        self.assertEqual(thin.tier_lines["WARM"], 2)
        self.assertAlmostEqual(thin.weighted_retention, 0.5 / 4.5)
        empty = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.02), warm_credit=0.5)
        self.assertEqual(empty.tier_lines["WARM"], 1)
        self.assertEqual(empty.weighted_retention, 0.0)

    def test_warm_never_costs_less_than_cold(self):
        table = manual_table([0.5, 0.1], [2, 2], [False, False])
        outcome = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.01))
        self.assertEqual(outcome.tier_lines["WARM"], outcome.tier_lines["COLD"])

    def test_reference_fraction_is_configurable_and_validated(self):
        table = manual_table([0.5], [100], [True])
        policy = anomaly_only_policy(0.8, 0.4, 0.1)  # keeps 10 lines
        strict = evaluate_policy(table, policy, warm_credit=0.5, warm_ref_fraction=0.5)
        loose = evaluate_policy(table, policy, warm_credit=0.5, warm_ref_fraction=0.1)
        self.assertAlmostEqual(strict.weighted_retention, 0.5 * 10 / 50)
        self.assertAlmostEqual(loose.weighted_retention, 0.5)
        self.assertEqual(strict.kept_lines, loose.kept_lines)  # credit never changes cost
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                evaluate_policy(table, policy, warm_ref_fraction=bad)

    def test_warm_credit_earned_is_zero_without_warm_anomalies(self):
        table = manual_table([0.9, 0.5], [10, 10], [True, False])
        outcome = evaluate_policy(table, anomaly_only_policy(0.8, 0.4, 0.3))
        self.assertEqual(outcome.warm_credit_earned, 0.0)
        self.assertEqual(outcome.to_dict()["warm_ref_fraction"], 0.25)

    def test_no_anomalies_means_full_retention(self):
        table = manual_table([0.1, 0.2], [5, 5], [False, False])
        outcome = evaluate_policy(table, anomaly_only_policy(0.9, 0.5, 0.1))
        self.assertEqual((outcome.weighted_retention, outcome.full_retention, outcome.coverage), (1.0, 1.0, 1.0))

    def test_everything_hot_saves_nothing(self):
        table = manual_table([0.9, 0.95], [10, 20], [True, False])
        outcome = evaluate_policy(table, anomaly_only_policy(0.5, 0.1, 0.1))
        self.assertEqual(outcome.storage_saved_pct, 0.0)
        self.assertEqual(outcome.full_retention, 1.0)

    def test_fitness_penalizes_only_a_shortfall(self):
        self.assertEqual(penalized_fitness(60.0, 0.97, 0.95), 60.0)
        self.assertEqual(penalized_fitness(60.0, 0.95, 0.95), 60.0)
        self.assertAlmostEqual(penalized_fitness(60.0, 0.90, 0.95, penalty_weight=1000.0), 60.0 - 50.0)
        # A deeper miss always scores worse than a shallower one at the same savings.
        self.assertLess(penalized_fitness(60.0, 0.80, 0.95), penalized_fitness(60.0, 0.90, 0.95))

    def test_kept_mask_uses_cold_accounting_for_dropped_blocks(self):
        table = manual_table([0.9, 0.1, 0.1], [10, 30, 60], [True, False, True])
        outcome = outcome_from_kept_mask(table, np.array([True, False, False]))
        self.assertEqual(outcome.kept_lines, 10 + 1 + 1)
        # No WARM tier, so all three retention numbers coincide.
        self.assertEqual({outcome.weighted_retention, outcome.full_retention, outcome.coverage}, {0.5})


class BaselineTest(unittest.TestCase):
    def setUp(self):
        frame = make_frame(n=4000, seed=3)
        norm = Normalizer.fit(frame["anomaly_score"], frame["event_frequency"], frame["block_size"])
        self.table = BlockTable.from_frame(frame, norm)

    def test_keep_all_saves_nothing_and_keeps_every_anomaly(self):
        outcome = baselines.keep_all(self.table)
        self.assertEqual(outcome.storage_saved_pct, 0.0)
        self.assertEqual(outcome.full_retention, 1.0)

    def test_keep_flagged_keeps_exactly_the_flagged_blocks(self):
        outcome = baselines.keep_flagged(self.table)
        self.assertEqual(outcome.tier_counts["HOT"], int(self.table.is_flagged.sum()))
        self.assertEqual(outcome.flagged_retention, 1.0)
        self.assertLess(outcome.full_retention, 1.0)  # the model misses some real anomalies

    def test_top_k_keeps_the_highest_scores(self):
        outcome = baselines.top_k_percent(self.table, 10)
        self.assertEqual(outcome.tier_counts["HOT"], 400)
        kept_scores = np.sort(self.table.anomaly_score)[-400:]
        self.assertGreater(kept_scores.min(), np.percentile(self.table.anomaly_score, 89))

    def test_top_k_for_target_reaches_the_target(self):
        k = baselines.smallest_top_k_for_target(self.table, 0.9)
        self.assertGreaterEqual(baselines.top_k_percent(self.table, k).full_retention, 0.9)
        self.assertLess(baselines.top_k_percent(self.table, max(k - 1.0, 0.0)).full_retention, 0.9)

    def test_matching_baselines_land_near_the_requested_line_budget(self):
        target = int(self.table.event_frequency.sum() * 0.4)
        top = baselines.top_k_matching_lines(self.table, target)
        self.assertAlmostEqual(top.kept_lines, target, delta=int(self.table.event_frequency.max()))
        for run in baselines.random_matching_lines(self.table, target, n_trials=3, seed=1):
            self.assertAlmostEqual(run.kept_lines, target, delta=int(self.table.event_frequency.max()))

    def test_ranking_by_score_beats_random_at_equal_cost(self):
        target = int(self.table.event_frequency.sum() * 0.4)
        top = baselines.top_k_matching_lines(self.table, target)
        random_runs = baselines.random_matching_lines(self.table, target, n_trials=10, seed=2)
        self.assertGreater(top.full_retention, np.mean([r.full_retention for r in random_runs]))

    def test_random_baseline_is_reproducible_for_a_seed(self):
        target = int(self.table.event_frequency.sum() * 0.5)
        a = baselines.random_matching_lines(self.table, target, n_trials=2, seed=5)
        b = baselines.random_matching_lines(self.table, target, n_trials=2, seed=5)
        self.assertEqual([r.full_retention for r in a], [r.full_retention for r in b])


class SplitTest(unittest.TestCase):
    def test_halves_are_disjoint_complete_and_stratified(self):
        labels = make_frame(n=3000, anomaly_rate=0.08, seed=4)["true_label"]
        tune, holdout = stratified_split(labels, 0.5, 42)
        self.assertEqual(len(set(tune) & set(holdout)), 0)
        self.assertEqual(len(tune) + len(holdout), len(labels))
        rate = lambda idx: (labels.iloc[idx] == "Anomaly").mean()
        self.assertAlmostEqual(rate(tune), rate(holdout), delta=0.005)
        self.assertAlmostEqual(len(tune) / len(labels), 0.5, delta=0.01)

    def test_split_is_reproducible_for_a_seed(self):
        labels = make_frame(n=500)["true_label"]
        a, b = stratified_split(labels, 0.5, 9), stratified_split(labels, 0.5, 9)
        np.testing.assert_array_equal(a[0], b[0])
        self.assertFalse(np.array_equal(a[0], stratified_split(labels, 0.5, 10)[0]))


class OptimizerTest(unittest.TestCase):
    def test_decoded_genes_always_form_a_valid_policy(self):
        rng = np.random.default_rng(0)
        for _ in range(200):
            genes = [rng.random(), rng.random(), rng.random(), rng.uniform(0.01, 1), rng.random(), rng.uniform(0.01, 0.5)]
            policy = decode_policy(genes)
            policy.validate()
            self.assertLess(policy.t_warm, policy.t_hot)
        self.assertEqual(len(GENE_NAMES), 6)

    def test_all_zero_weights_fall_back_to_equal(self):
        policy = decode_policy([0, 0, 0, 0.5, 0.5, 0.1])
        np.testing.assert_allclose(policy.weights, [1 / 3] * 3)

    def test_ga_meets_the_target_and_beats_the_baselines_on_holdout(self):
        frame = make_frame(n=6000, anomaly_rate=0.08, seed=11)
        tune_idx, holdout_idx = stratified_split(frame["true_label"], 0.5, 42)
        tune_frame, holdout_frame = frame.iloc[tune_idx], frame.iloc[holdout_idx]
        norm = Normalizer.fit(tune_frame["anomaly_score"], tune_frame["event_frequency"], tune_frame["block_size"])
        tune, holdout = BlockTable.from_frame(tune_frame, norm), BlockTable.from_frame(holdout_frame, norm)

        _, result = run_ga(tune, population_size=40, num_generations=40, num_parents_mating=15, verbose=False)
        self.assertEqual(result.generations_completed, 40)
        self.assertEqual(len(result.best_fitness_per_generation), 40)
        self.assertGreaterEqual(result.best_fitness_per_generation[-1], result.best_fitness_per_generation[0])

        tuned = evaluate_policy(tune, result.best_policy)
        held_out = evaluate_policy(holdout, result.best_policy)
        self.assertGreaterEqual(tuned.weighted_retention, 0.95)
        self.assertGreater(held_out.storage_saved_pct, 10.0)
        # Holdout is a different sample than tune, so allow a little slack on the target.
        self.assertGreaterEqual(held_out.weighted_retention, 0.90)

        comparison = compare_to_baselines(tune, holdout, held_out, 0.95, 1000.0, n_random_trials=5)
        for name in ("keep_all", "keep_flagged_only", "random_sampling_same_savings"):
            self.assertTrue(comparison[name]["ga_better"], name)
        # Here the model's score is the only informative signal, so ranking by it
        # (top-k) is near-optimal and the GA can only match it, not beat it.
        # (On the real data the other signals help; see storage_report.json.)
        ga_fitness = held_out.fitness(0.95, 1000.0)
        for name in ("top_k_percent_tuned", "top_k_percent_same_savings"):
            self.assertGreater(ga_fitness, comparison[name]["fitness"] - 5.0, name)

    def test_same_seed_gives_the_same_policy(self):
        frame = make_frame(n=1500, seed=2)
        norm = Normalizer.fit(frame["anomaly_score"], frame["event_frequency"], frame["block_size"])
        table = BlockTable.from_frame(frame, norm)
        kwargs = dict(population_size=10, num_generations=5, num_parents_mating=4, verbose=False, random_seed=7)
        a = run_ga(table, **kwargs)[1].best_policy.to_dict()
        b = run_ga(table, **kwargs)[1].best_policy.to_dict()
        self.assertEqual(a, b)

    def test_a_stricter_target_cannot_save_more(self):
        frame = make_frame(n=3000, seed=6)
        norm = Normalizer.fit(frame["anomaly_score"], frame["event_frequency"], frame["block_size"])
        table = BlockTable.from_frame(frame, norm)
        kwargs = dict(population_size=24, num_generations=25, num_parents_mating=10, verbose=False)
        loose = evaluate_policy(table, run_ga(table, retention_target=0.6, **kwargs)[1].best_policy)
        strict = evaluate_policy(table, run_ga(table, retention_target=0.99, **kwargs)[1].best_policy)
        self.assertGreater(loose.storage_saved_pct, strict.storage_saved_pct)

    def test_run_ga_can_be_called_twice_in_one_process(self):
        # pygad mutates gene_space/gene_type in place; the module constants must survive that.
        frame = make_frame(n=500, seed=1)
        table = BlockTable.from_frame(frame, Normalizer.fit(frame["anomaly_score"], frame["event_frequency"], frame["block_size"]))
        for _ in range(2):
            run_ga(table, population_size=6, num_generations=2, num_parents_mating=3, verbose=False)

    def test_report_payloads_have_the_documented_shape(self):
        frame = make_frame(n=2000, seed=8)
        clean = make_variant(frame, seed=1, generations=3, model_info={"kind": "tune_half_refit", "version": 2, "model_path": "c.pkl"})
        shifted = frame.assign(anomaly_score=frame["anomaly_score"] + np.random.default_rng(0).normal(0, 0.05, len(frame)))
        deployed = make_variant(shifted, seed=1, generations=3, model_info={"kind": "deployed", "version": 2, "model_path": "d.pkl"})

        policy_payload, report = build_report(clean, deployed, 0.95, 1000.0, 0.5, 1, warm_credit=0.5, n_random_trials=3)
        json.dumps(policy_payload)
        json.dumps(report)  # must be serializable as-is
        for key in ("policy", "normalizer", "retention_target", "warm_credit", "warm_ref_fraction", "split", "model"):
            self.assertIn(key, policy_payload)
        self.assertEqual(report["config"]["warm_ref_fraction"], 0.25)
        for key in ("mean_share_of_block_kept", "mean_credit_per_warm_anomaly", "warm_ref_fraction"):
            self.assertIn(key, report["warm_tier"])
        self.assertEqual(policy_payload["model"]["kind"], "tune_half_refit")  # the clean policy is the one served
        for key in ("holdout", "tuning", "baselines", "ga_history", "split", "ga_beats_baselines", "config",
                    "warm_tier", "deployed_model_scoring", "holdout_comparison"):
            self.assertIn(key, report)
        for key in ("storage_saved_pct", "weighted_retention", "full_retention", "coverage", "tier_counts", "tier_lines", "tier_anomalies"):
            self.assertIn(key, report["holdout"])
        for name in ("keep_all", "random_sampling_same_savings", "keep_flagged_only", "top_k_percent_tuned"):
            self.assertIn(name, report["baselines"])
            self.assertIn(name, report["deployed_model_scoring"]["baselines"])
        self.assertEqual(report["deployed_model_scoring"]["model"]["kind"], "deployed")
        self.assertEqual(len(report["ga_history"]["best_fitness_per_generation"]), 3)
        self.assertEqual(sum(report["holdout"]["tier_counts"].values()), len(clean.holdout_table))

        # One row per policy, one column per scoring variant, each matching its own section.
        rows = {row["name"]: row for row in report["holdout_comparison"]}
        self.assertEqual(set(rows), {"ga_policy", "keep_all", "keep_flagged_only", "top_k_percent_tuned",
                                     "top_k_percent_same_savings", "random_sampling_same_savings"})
        self.assertEqual(rows["ga_policy"]["clean"]["storage_saved_pct"], report["holdout"]["storage_saved_pct"])
        self.assertEqual(rows["ga_policy"]["deployed"]["storage_saved_pct"],
                         report["deployed_model_scoring"]["holdout"]["storage_saved_pct"])

    def test_report_without_a_comparison_variant_has_no_second_column(self):
        _, report = build_report(make_variant(make_frame(n=800), generations=2), None, 0.95, 1000.0, 0.5, 42, n_random_trials=2)
        self.assertNotIn("deployed_model_scoring", report)
        self.assertNotIn("holdout_comparison", report)

    def test_warm_diagnostics_describe_the_warm_tier(self):
        variant = make_variant(make_frame(n=2000, seed=4), generations=3)
        _, report = build_report(variant, None, 0.95, 1000.0, 0.5, 42, n_random_trials=2)
        warm, holdout = report["warm_tier"], report["holdout"]
        self.assertEqual(warm["warm_blocks"], holdout["tier_counts"]["WARM"])
        self.assertEqual(warm["warm_anomalies"], holdout["tier_anomalies"]["WARM"])
        if warm["warm_blocks"]:
            self.assertAlmostEqual(warm["mean_lines_per_warm_block"], holdout["tier_lines"]["WARM"] / warm["warm_blocks"])


class RefitTest(unittest.TestCase):
    def setUp(self):
        from sklearn.ensemble import IsolationForest

        rng = np.random.default_rng(0)
        n = 1000
        self.merged = pd.DataFrame({
            "block_id": [f"blk_{i}" for i in range(n)],
            "event_frequency": rng.integers(2, 60, n),
            "block_size": rng.integers(0, 1000, n),
            "component_encoded": rng.integers(0, 5, n),
            "is_error": 0,
            "Label": np.where(rng.random(n) < 0.1, "Anomaly", "Normal"),
        })
        self.template = IsolationForest(n_estimators=20, max_samples=0.5, contamination=0.05, random_state=3)
        self.template.fit(self.merged[["event_frequency", "component_encoded"]])

    def test_refit_keeps_hyperparameters_and_sees_only_the_given_rows(self):
        tune_idx, holdout_idx = stratified_split(self.merged["Label"], 0.5, 42)
        model = refit_on_rows(self.template, self.merged, tune_idx)
        self.assertIsNot(model, self.template)
        self.assertEqual(model.get_params(), self.template.get_params())
        self.assertEqual(list(model.feature_names_in_), ["event_frequency", "component_encoded"])
        # max_samples is a fraction, so the fitted sample size reveals how many rows it saw.
        self.assertEqual(model.max_samples_, int(0.5 * len(tune_idx)))
        self.assertEqual(self.template.max_samples_, int(0.5 * len(self.merged)))

    def test_refit_is_independent_of_the_held_out_rows(self):
        tune_idx, holdout_idx = stratified_split(self.merged["Label"], 0.5, 42)
        changed = self.merged.copy()
        changed.loc[holdout_idx, "event_frequency"] = 10_000  # holdout contents must not matter
        a = refit_on_rows(self.template, self.merged, tune_idx)
        b = refit_on_rows(self.template, changed, tune_idx)
        probe = self.merged[["event_frequency", "component_encoded"]]
        np.testing.assert_array_equal(a.decision_function(probe), b.decision_function(probe))


class StorageApiTest(unittest.TestCase):
    """Routes against synthetic data and a temp policy/report, via a bare Flask
    app (importing app.py would need a live MongoDB)."""

    @classmethod
    def setUpClass(cls):
        import storage_api

        cls.storage_api = storage_api
        cls.frame = make_frame(n=3000, anomaly_rate=0.08, seed=21)

    def setUp(self):
        import os
        import time

        import jwt
        from flask import Flask

        api = self.storage_api
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        tmp = Path(self.tmp.name)
        self.model_file = tmp / "tune_half.pkl"
        self.model_file.write_bytes(b"stand-in; _scored_frame is patched")

        variant = make_variant(self.frame, model_info={"kind": "tune_half_refit", "version": 2, "model_path": str(self.model_file)})
        policy_payload, report = build_report(variant, variant, 0.95, 1000.0, 0.5, 42, warm_credit=0.5, n_random_trials=2)
        self.policy = variant.policy
        self.holdout = variant.holdout_table
        (tmp / "policy.json").write_text(json.dumps(policy_payload))
        (tmp / "report.json").write_text(json.dumps(report))

        self.scored_with = []

        def scored_frame(model_path):
            self.scored_with.append(model_path)
            return self.frame

        self.deployed_version = 2
        patches = [
            mock.patch.object(api, "POLICY_PATH", tmp / "policy.json"),
            mock.patch.object(api, "REPORT_PATH", tmp / "report.json"),
            mock.patch.object(api, "_scored_frame", scored_frame),
            mock.patch.object(api, "resolve_deployed_model", lambda: {"version": self.deployed_version, "model_path": "m.pkl"}),
            mock.patch.dict(os.environ, {"JWT_SECRET": "test-secret"}),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        api._cache.update({"key": None, "tables": None})

        app = Flask(__name__)
        app.register_blueprint(api.storage_bp)
        self.client = app.test_client()
        token = jwt.encode({"sub": "t", "exp": int(time.time()) + 600}, "test-secret", algorithm="HS256")
        self.headers = {"Authorization": f"Bearer {token}"}

    def simulate(self, payload):
        return self.client.post("/api/storage/simulate", json=payload, headers=self.headers)

    def test_routes_require_a_token(self):
        self.assertEqual(self.client.get("/api/storage/summary").status_code, 401)
        self.assertEqual(self.client.get("/api/storage/policy").status_code, 401)
        self.assertEqual(self.client.post("/api/storage/simulate", json={}).status_code, 401)

    def test_summary_reports_holdout_numbers_for_both_scorings(self):
        body = self.client.get("/api/storage/summary", headers=self.headers).get_json()
        for key in ("weighted_retention", "full_retention", "coverage", "storage_saved_pct"):
            self.assertIn(key, body["holdout"])
        self.assertIn("baselines", body)
        self.assertIn("holdout_comparison", body)
        # GA history is trimmed from both sections.
        self.assertNotIn("ga_history", body)
        self.assertNotIn("ga_history", body["deployed_model_scoring"])
        self.assertIn("ga_summary", body["deployed_model_scoring"])
        self.assertFalse(body["stale"])

    def test_policy_route_returns_the_tuned_policy(self):
        body = self.client.get("/api/storage/policy", headers=self.headers).get_json()
        self.assertAlmostEqual(sum(body["policy"]["weights"].values()), 1.0)
        self.assertLess(body["policy"]["thresholds"]["warm"], body["policy"]["thresholds"]["hot"])
        self.assertEqual(body["model"]["kind"], "tune_half_refit")

    def test_simulate_without_overrides_reproduces_the_holdout_report(self):
        body = self.simulate({}).get_json()
        expected = evaluate_policy(self.holdout, self.policy, 0.5)
        self.assertEqual(body["split"], "holdout")
        self.assertEqual(body["warm_credit"], 0.5)
        for key in ("storage_saved_pct", "weighted_retention", "full_retention", "coverage"):
            self.assertAlmostEqual(body["outcome"][key], getattr(expected, key), places=6, msg=key)

    def test_simulate_scores_with_the_policy_model_not_the_deployed_one(self):
        self.simulate({})
        self.assertEqual(self.scored_with, [str(self.model_file)])
        self.simulate({"t_hot": 0.5})  # cached after the first call
        self.assertEqual(len(self.scored_with), 1)

    def test_warm_credit_override_changes_retention_not_cost(self):
        # Put most blocks in WARM so the credit visibly matters; summaries at
        # half the block are above the reference share, so credit 1 is full credit.
        payload = {"t_hot": 0.9, "t_warm": 0.01, "summary_fraction": 0.5}
        none = self.simulate({**payload, "warm_credit": 0.0}).get_json()
        full = self.simulate({**payload, "warm_credit": 1.0}).get_json()
        self.assertEqual(none["outcome"]["kept_lines"], full["outcome"]["kept_lines"])
        self.assertAlmostEqual(none["outcome"]["weighted_retention"], none["outcome"]["full_retention"])
        self.assertAlmostEqual(full["outcome"]["weighted_retention"], full["outcome"]["coverage"])
        self.assertGreater(full["outcome"]["weighted_retention"], none["outcome"]["weighted_retention"])

    def test_warm_ref_fraction_override_changes_retention_not_cost(self):
        payload = {"t_hot": 0.9, "t_warm": 0.01, "summary_fraction": 0.1, "warm_credit": 1.0}
        loose = self.simulate({**payload, "warm_ref_fraction": 0.05}).get_json()
        strict = self.simulate({**payload, "warm_ref_fraction": 1.0}).get_json()
        self.assertEqual(strict["warm_ref_fraction"], 1.0)
        self.assertEqual(loose["outcome"]["kept_lines"], strict["outcome"]["kept_lines"])
        self.assertGreater(loose["outcome"]["weighted_retention"], strict["outcome"]["weighted_retention"])

    def test_raising_t_hot_keeps_fewer_hot_blocks(self):
        loose = self.simulate({"t_hot": 0.2, "t_warm": 0.1}).get_json()
        tight = self.simulate({"t_hot": 0.9, "t_warm": 0.1}).get_json()
        self.assertGreater(tight["outcome"]["storage_saved_pct"], loose["outcome"]["storage_saved_pct"])
        self.assertLessEqual(tight["outcome"]["tier_counts"]["HOT"], loose["outcome"]["tier_counts"]["HOT"])
        self.assertLessEqual(tight["outcome"]["full_retention"], loose["outcome"]["full_retention"])

    def test_simulate_supports_other_splits_and_a_custom_target(self):
        body = self.simulate({"split": "all", "retention_target": 0.5}).get_json()
        self.assertEqual(body["outcome"]["n_blocks"], len(self.frame))
        self.assertEqual(body["retention_target"], 0.5)

    def test_simulate_rejects_invalid_input(self):
        bad_payloads = [
            {"t_warm": 0.99, "t_hot": 0.5},
            {"t_hot": "high"},
            {"t_hot": True},
            {"summary_fraction": 0},
            {"weights": {"anomaly_score": 0, "event_frequency": 0, "block_size": 0}},
            {"weights": {"nonsense": 1}},
            {"weights": [1, 2, 3]},
            {"split": "everything"},
            {"retention_target": 1.5},
            {"warm_credit": 1.5},
            {"warm_credit": -0.1},
            {"warm_credit": "half"},
            {"warm_ref_fraction": 0},
            {"warm_ref_fraction": 1.5},
        ]
        for payload in bad_payloads:
            response = self.simulate(payload)
            self.assertEqual(response.status_code, 400, msg=str(payload))
            self.assertIn("error", response.get_json())

    def test_partial_weight_override_keeps_the_other_weights(self):
        body = self.simulate({"weights": {"block_size": 0}}).get_json()
        self.assertEqual(body["policy"]["weights"]["block_size"], 0.0)

    def test_routes_404_with_a_hint_until_the_script_has_run(self):
        Path(self.storage_api.POLICY_PATH).unlink()
        Path(self.storage_api.REPORT_PATH).unlink()
        for response in (
            self.client.get("/api/storage/summary", headers=self.headers),
            self.client.get("/api/storage/policy", headers=self.headers),
            self.simulate({}),
        ):
            self.assertEqual(response.status_code, 404)
            self.assertIn("optimize_storage_policy", response.get_json()["error"])

    def test_simulate_404s_when_the_policy_model_is_missing(self):
        self.model_file.unlink()
        response = self.simulate({})
        self.assertEqual(response.status_code, 404)
        self.assertIn("optimize_storage_policy", response.get_json()["error"])

    def test_policy_tuned_on_an_older_model_is_flagged_stale(self):
        self.deployed_version = 3
        self.assertTrue(self.client.get("/api/storage/policy", headers=self.headers).get_json()["stale"])
        self.assertTrue(self.client.get("/api/storage/summary", headers=self.headers).get_json()["stale"])
        self.assertTrue(self.simulate({}).get_json()["stale"])


if __name__ == "__main__":
    unittest.main()
