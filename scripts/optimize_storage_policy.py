"""Tune the smart-storage policy with a genetic algorithm and report holdout numbers.

Splits the labeled blocks 50/50 (stratified on the ground-truth label), then
runs the same tune -> holdout protocol on two score sources:

  clean     the deployed model's hyperparameters refitted on the tune half
            only, so holdout scores come from a model that never saw them.
            Its policy is the one saved and served by the API.
  deployed  the deployed model as-is (fitted on every block), kept as a
            comparison column.

Each run fits its own normalizer and GA on the tune half and measures the
policy and every baseline on the holdout half.

Writes:
    data/storage/isolation_forest_tune_half.pkl  the clean refit
    data/storage/policy_best.json                the clean policy, its normalizer, how to reproduce the split
    data/storage/storage_report.json             both runs: savings, retention, tiers, baselines, GA history

Usage:
    python3 scripts/optimize_storage_policy.py
    python3 scripts/optimize_storage_policy.py --warm-credit 0.3 --retention-target 0.9
    python3 scripts/optimize_storage_policy.py --warm-ref-fraction 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.isolation_forest import save_model
from smart_storage.data import (
    POLICY_PATH,
    REPORT_PATH,
    STORAGE_DIR,
    TUNE_HALF_MODEL_PATH,
    load_labeled_blocks,
    refit_on_rows,
    resolve_deployed_model,
    score_blocks,
)
from smart_storage.optimizer import DEFAULT_TEST_SIZE, RANDOM_STATE, ScoringVariant, build_report, run_ga, stratified_split
from smart_storage.policy import PENALTY_WEIGHT, RETENTION_TARGET, WARM_CREDIT, WARM_REF_FRACTION, BlockTable, Normalizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--retention-target", type=float, default=RETENTION_TARGET,
                        help="minimum weighted anomaly retention (default %(default)s)")
    parser.add_argument("--warm-credit", type=float, default=WARM_CREDIT,
                        help="most retention credit an anomaly kept as a WARM summary can earn; HOT = 1, COLD = 0 (default %(default)s)")
    parser.add_argument("--warm-ref-fraction", type=float, default=WARM_REF_FRACTION,
                        help="share of a block a WARM summary must keep to earn the full warm credit; "
                             "thinner summaries earn proportionally less (default %(default)s)")
    parser.add_argument("--penalty-weight", type=float, default=PENALTY_WEIGHT)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--population", type=int, default=80)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--parents", type=int, default=30)
    parser.add_argument("--random-trials", type=int, default=20,
                        help="repeats for the random-sampling baseline, which is averaged")
    return parser.parse_args()


def tune_variant(name, frame, tune_idx, holdout_idx, model_info, args) -> ScoringVariant:
    tune_frame, holdout_frame = frame.iloc[tune_idx], frame.iloc[holdout_idx]
    # Normalization is fitted on the tuning half only and reused as-is on the holdout.
    normalizer = Normalizer.fit(tune_frame["anomaly_score"], tune_frame["event_frequency"], tune_frame["block_size"])
    tune_table = BlockTable.from_frame(tune_frame, normalizer)
    holdout_table = BlockTable.from_frame(holdout_frame, normalizer)

    print(f"\nGA [{name}]: population {args.population}, {args.generations} generations, "
          f"target {args.retention_target}, warm credit {args.warm_credit} at f_ref {args.warm_ref_fraction}")
    _, ga_result = run_ga(
        tune_table,
        population_size=args.population,
        num_generations=args.generations,
        num_parents_mating=args.parents,
        retention_target=args.retention_target,
        penalty_weight=args.penalty_weight,
        random_seed=args.seed,
        warm_credit=args.warm_credit,
        warm_ref_fraction=args.warm_ref_fraction,
    )
    print(f"  done in {ga_result.wall_clock_seconds:.1f}s")
    return ScoringVariant(model_info, normalizer, tune_table, holdout_table, ga_result)


def print_section(title: str, section: dict):
    policy, warm = section["policy"], section["warm_tier"]
    print(f"\n{title}")
    print(f"  weights     {{{', '.join(f'{k}: {v:.3f}' for k, v in policy['weights'].items())}}}")
    print(f"  thresholds  hot={policy['thresholds']['hot']:.4f}  warm={policy['thresholds']['warm']:.4f}")
    print(f"  summary f   {policy['summary_fraction']:.4f}")
    holdout = section["holdout"]
    print(f"  holdout tiers  blocks {holdout['tier_counts']}  lines {holdout['tier_lines']}  anomalies {holdout['tier_anomalies']}")
    if warm["warm_blocks"]:
        print(f"  WARM: {warm['mean_lines_per_warm_block']:.2f} lines/block kept from a mean of "
              f"{warm['mean_event_frequency_of_warm_blocks']:.1f}; "
              f"{warm['share_of_warm_blocks_kept_at_one_line']:.1%} of WARM blocks keep only 1 line (the COLD cost)")
        if warm.get("mean_credit_per_warm_anomaly") is not None:
            print(f"  WARM credit earned per anomaly: {warm['mean_credit_per_warm_anomaly']:.3f} "
                  f"(max {warm['warm_credit_max']}, full at f_ref {warm['warm_ref_fraction']})")
    print(f"  holdout retention  full={holdout['full_retention']:.4f}  weighted={holdout['weighted_retention']:.4f}  "
          f"coverage={holdout['coverage']:.4f}  saved={holdout['storage_saved_pct']:.2f}%")


def print_table(report: dict):
    rows = report.get("holdout_comparison", [])
    columns = [c for c in ("clean", "deployed") if rows and c in rows[0]]
    header = f"{'HOLDOUT':<30}" + "".join(
        f"{c + ' saved%':>14}{'weighted':>10}{'full':>8}{'cover':>8}{'fitness':>10}" for c in columns
    )
    print("\n" + header)
    for row in rows:
        line = f"  {row['name']:<28}"
        for c in columns:
            r = row[c]
            line += (f"{r['storage_saved_pct']:>14.2f}{r['weighted_retention']:>10.4f}{r['full_retention']:>8.4f}"
                     f"{r['coverage']:>8.4f}{r['fitness']:>10.2f}")
        print(line)


def main():
    args = parse_args()
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    deployed = resolve_deployed_model()
    merged = load_labeled_blocks()
    tune_idx, holdout_idx = stratified_split(merged["Label"], args.test_size, args.seed)
    print(f"{len(merged):,} labeled blocks -> tune {len(tune_idx):,} / holdout {len(holdout_idx):,}")

    template = joblib.load(deployed["model_path"])
    print(f"Refitting v{deployed['version']} hyperparameters on the tune half only ({len(tune_idx):,} blocks) ...")
    clean_model = refit_on_rows(template, merged, tune_idx)
    save_model(clean_model, TUNE_HALF_MODEL_PATH)
    clean_info = {
        "kind": "tune_half_refit",
        "version": deployed["version"],  # the deployed version whose hyperparameters were refitted
        "model_path": str(TUNE_HALF_MODEL_PATH),
        "hyperparameters_from": deployed["model_path"],
        "hyperparameters": {k: v for k, v in template.get_params().items() if k not in ("n_jobs", "verbose")},
        "features": list(template.feature_names_in_),
        "fitted_on": f"tune half only ({len(tune_idx):,} blocks)",
    }
    deployed_info = {
        "kind": "deployed",
        "version": deployed["version"],
        "model_path": deployed["model_path"],
        "fitted_on": "every labeled block (both halves are in-sample)",
    }

    clean = tune_variant("clean", score_blocks(clean_model, merged), tune_idx, holdout_idx, clean_info, args)
    in_sample = tune_variant("deployed", score_blocks(template, merged), tune_idx, holdout_idx, deployed_info, args)

    policy_payload, report = build_report(
        clean, in_sample, args.retention_target, args.penalty_weight, args.test_size, args.seed,
        args.warm_credit, args.random_trials, args.warm_ref_fraction,
    )
    POLICY_PATH.write_text(json.dumps(policy_payload, indent=2), encoding="utf-8")
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {TUNE_HALF_MODEL_PATH}\nWrote {POLICY_PATH}\nWrote {REPORT_PATH}")

    print_section("Clean policy (tune-half model) -- saved and served", report)
    print_section("Deployed-model policy -- comparison only", report["deployed_model_scoring"])
    print_table(report)
    print(f"\nGA beats every baseline on holdout fitness -- clean: {'yes' if report['ga_beats_baselines'] else 'NO'}, "
          f"deployed: {'yes' if report['deployed_model_scoring']['ga_beats_baselines'] else 'NO'}")


if __name__ == "__main__":
    main()
