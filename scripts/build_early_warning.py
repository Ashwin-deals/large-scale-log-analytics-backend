"""Train and evaluate the early-warning detectors: one Isolation Forest per
prefix length k, each scoring a block from only its first k events.

Reads the event-level cleaned log (data/processed/hdfs_cleaned.csv) in
chunks, builds prefix features for every k in one pass (cached under
data/features/early_warning/, rebuilt when the cleaned log is newer), splits
the labeled blocks into stratified train/test, and for each k fits on train,
GA-tunes the alert threshold on train, and evaluates once on test.

Writes:
    data/models/early_warning/isolation_forest_k{k}.pkl  model (+ threshold as attributes)
    data/models/early_warning/k{k}.json                  threshold, features, test metrics
    data/evaluation/early_warning_report.json            all k's + the precision-vs-lead-time trade-off

Usage:
    python3 scripts/build_early_warning.py
    python3 scripts/build_early_warning.py --ks 3 5 10 20 --rebuild-features
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from detection.isolation_forest import save_model
from detection.train import DEFAULT_LABELS_PATH
from early_warning.model import DEFAULT_TEST_SIZE, POSITIVE_LABEL, RANDOM_STATE, stratified_split, train_and_evaluate_k, trade_off
from early_warning.prefix_features import (
    BOOKKEEPING_COLUMNS,
    DEFAULT_CHUNKSIZE,
    DEFAULT_CLEANED_PATH,
    DEFAULT_PREFIX_DIR,
    PREFIX_KS,
    build_prefix_features,
    feature_columns,
    require_cleaned_log,
)

MODELS_DIR = Path("data/models/early_warning")
REPORT_PATH = Path("data/evaluation/early_warning_report.json")
SCAN_STATS_PATH = DEFAULT_PREFIX_DIR / "scan_stats.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cleaned", type=Path, default=DEFAULT_CLEANED_PATH)
    parser.add_argument("--ks", type=int, nargs="+", default=list(PREFIX_KS))
    parser.add_argument("--chunksize", type=int, default=DEFAULT_CHUNKSIZE)
    parser.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--population", type=int, default=20)
    parser.add_argument("--generations", type=int, default=30)
    parser.add_argument("--rebuild-features", action="store_true", help="rescan the cleaned log even if cached prefix features exist")
    return parser.parse_args()


def prefix_path(k: int) -> Path:
    return DEFAULT_PREFIX_DIR / f"prefix_k{k}.csv"


def load_or_build_prefix_features(cleaned: Path, ks, chunksize: int, rebuild: bool):
    cached = all(prefix_path(k).exists() for k in ks) and SCAN_STATS_PATH.exists()
    fresh = cached and all(prefix_path(k).stat().st_mtime > cleaned.stat().st_mtime for k in ks)
    # A cache written before a bookkeeping column existed must be rebuilt, not reused.
    fresh = fresh and all(set(BOOKKEEPING_COLUMNS) <= set(pd.read_csv(prefix_path(k), nrows=0).columns) for k in ks)
    if fresh and not rebuild:
        print(f"Using cached prefix features in {DEFAULT_PREFIX_DIR}/ (pass --rebuild-features to rescan)")
        stats = json.loads(SCAN_STATS_PATH.read_text(encoding="utf-8"))
        return {k: pd.read_csv(prefix_path(k)) for k in ks}, stats

    print(f"Scanning {cleaned} in chunks of {chunksize:,} rows for k = {sorted(ks)} ...")
    start = time.perf_counter()
    frames, stats = build_prefix_features(cleaned, ks, chunksize=chunksize, verbose=True)
    stats["scan_seconds"] = time.perf_counter() - start
    stats["chunksize"] = chunksize
    DEFAULT_PREFIX_DIR.mkdir(parents=True, exist_ok=True)
    for k, frame in frames.items():
        frame.to_csv(prefix_path(k), index=False)
    SCAN_STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"  {stats['rows']:,} rows, {stats['blocks']:,} blocks in {stats['scan_seconds']:.0f}s")
    return frames, stats


def main():
    args = parse_args()
    try:
        cleaned = require_cleaned_log(args.cleaned)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    ks = sorted(set(args.ks))
    frames, scan_stats = load_or_build_prefix_features(cleaned, ks, args.chunksize, args.rebuild_features)

    # One label vector and one split for every k: all frames list blocks in
    # the same first-appearance order.
    reference_ids = frames[ks[0]]["block_id"]
    for k in ks[1:]:
        if not frames[k]["block_id"].equals(reference_ids):
            raise RuntimeError(f"prefix_k{k}.csv lists blocks in a different order than prefix_k{ks[0]}.csv; rebuild with --rebuild-features")
    labels = pd.read_csv(DEFAULT_LABELS_PATH).set_index("BlockId")["Label"]
    label_per_block = reference_ids.map(labels)
    labeled = label_per_block.notna().to_numpy()
    print(f"{int(labeled.sum()):,} of {len(labeled):,} blocks have a label ({int((~labeled).sum()):,} unlabeled, excluded)")
    for k in ks:
        frames[k] = frames[k][labeled].reset_index(drop=True)
    is_anomaly = (label_per_block[labeled] == POSITIVE_LABEL).to_numpy()
    train_idx, test_idx = stratified_split(label_per_block[labeled].to_numpy(), args.test_size, args.seed)
    print(f"train {len(train_idx):,} / test {len(test_idx):,} blocks; anomaly rate {is_anomaly.mean():.2%}")

    columns = [c for c in frames[ks[0]].columns if c.startswith("count_")]
    all_features = feature_columns([c[len("count_"):] for c in columns])
    ga_kwargs = {"population_size": args.population, "num_generations": args.generations, "random_seed": args.seed}

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    per_k = {}
    for k in ks:
        print(f"\nk = {k}: fitting on {len(train_idx):,} train blocks, GA-tuning the threshold ...")
        result = train_and_evaluate_k(k, frames[k], is_anomaly, train_idx, test_idx, all_features, ga_kwargs)
        model_path = MODELS_DIR / f"isolation_forest_k{k}.pkl"
        result.model.prefix_k_ = k
        result.model.alert_threshold_ = result.ga.threshold
        result.model.alert_quantile_ = result.ga.alert_quantile
        save_model(result.model, model_path)
        entry = result.to_dict()
        entry["model_path"] = str(model_path)
        (MODELS_DIR / f"k{k}.json").write_text(json.dumps({key: entry[key] for key in ("k", "model_path", "features", "threshold", "test")}, indent=2), encoding="utf-8")
        per_k[k] = entry
        t = entry["test"]
        print(f"  test: precision {t['precision']:.4f} recall {t['recall']:.4f} F1 {t['f1']:.4f} "
              f"false-alarm {t['false_alarm_rate']:.4%} | mean lead (true alerts) {t['lead_time']['mean_over_true_alerts']:.3f}")

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source": {"cleaned_log": str(cleaned), **scan_stats},
        "config": {
            "ks": ks, "test_size": args.test_size, "seed": args.seed,
            "model": "IsolationForest (detection.isolation_forest.build_isolation_forest defaults), fitted on train blocks only",
            "threshold": "GA over alert_quantile (threshold = that quantile of train scores), fitness = train F1(Anomaly)",
            "prefix_order": "first k events of each block in file (arrival) order",
            "lead_time": "fraction of a block's events not yet seen when the alert fires: (total - min(k, total)) / total; 0 if the block has <= k events",
            "lead_time_seconds": "log-time seconds from the k-th event (alert) to the block's last event; mean and median over true alerts; 0 if the block has <= k events",
        },
        "split": {
            "train": {"blocks": int(len(train_idx)), "anomalies": int(is_anomaly[train_idx].sum())},
            "test": {"blocks": int(len(test_idx)), "anomalies": int(is_anomaly[test_idx].sum())},
            "unlabeled_blocks_excluded": int((~labeled).sum()),
        },
        "results": {str(k): v for k, v in per_k.items()},
        "trade_off": trade_off(per_k),
        # Same alerts, only on blocks still running when they fire (> k events):
        # the early-warning performance proper. See early_warning/model.py.
        "trade_off_in_progress": trade_off(per_k, split="test_in_progress"),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote {REPORT_PATH} and {len(ks)} models under {MODELS_DIR}/")

    print(f"\n{'TEST':<6}{'precision':>10}{'recall':>8}{'F1':>8}{'false alarm':>13}{'lead (TP)':>11}{'lead/alert':>12}{'<=k events':>12}")
    for k in ks:
        t = per_k[k]["test"]
        print(f"  k={k:<3}{t['precision']:>10.4f}{t['recall']:>8.4f}{t['f1']:>8.4f}{t['false_alarm_rate']:>13.4%}"
              f"{t['lead_time']['mean_over_true_alerts']:>11.3f}{t['lead_time']['expected_lead_per_alert']:>12.3f}"
              f"{per_k[k]['test_blocks_with_at_most_k_events']:>12,}")
    print(f"\n{'STILL RUNNING':<14}{'blocks':>9}{'anomalies':>11}{'precision':>10}{'recall':>8}{'F1':>8}{'lead (TP)':>11}"
          f"{'median s':>10}{'mean s':>10}")
    for k in ks:
        t = per_k[k]["test_in_progress"]
        lead = t["lead_time"]
        seconds = lambda v: f"{v:>10,.0f}" if v is not None else f"{'n/a':>10}"
        print(f"  k={k:<11}{t['blocks']:>9,}{t['anomalies']:>11,}{t['precision']:>10.4f}{t['recall']:>8.4f}{t['f1']:>8.4f}"
              f"{lead['mean_over_true_alerts']:>11.3f}{seconds(lead.get('median_seconds_over_true_alerts'))}"
              f"{seconds(lead.get('mean_seconds_over_true_alerts'))}")
    for key, label in (("trade_off", "all test blocks"), ("trade_off_in_progress", "still-running blocks")):
        best = report[key]
        print(f"Pareto front (precision vs lead, {label}): k = {best['pareto_front']}; best trade-off: k = {best['best_k']}")


if __name__ == "__main__":
    main()
