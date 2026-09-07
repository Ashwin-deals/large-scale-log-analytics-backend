# Adaptation cycle V2 -> V3

All metrics below are computed on the full 575,061-block labeled dataset
(`detection/evaluate.py`, `Anomaly` as the positive class), so the three
rows are directly comparable regardless of what data each candidate was
*trained* on.

| Version | Data used to train/select              | Accuracy | Precision | Recall | F1     | Promoted?               |
|---------|-----------------------------------------|---------:|----------:|-------:|-------:|--------------------------|
| V1      | Full dataset (default IsolationForest)  | 0.8194   | 0.1034    | 0.6740 | 0.1794 | baseline                 |
| V2      | Full dataset, GA-optimized              | 0.9832   | 0.7686    | 0.6077 | 0.6788 | yes (over V1)            |
| V3      | Later chronological half only, GA-optimized | 0.7573 | 0.0881 | 0.7791 | 0.1582 | **no** (worse than V2)   |

## Data slice used for V3

`features.csv` rows are ordered by each block's earliest event (the cleaned
logs are sorted by `datetime` before being grouped by `block_id` with
`sort=False` in `feature_engineering/feature_extractor.py`), and the inner
join in `detection/train.load_labeled_features()` preserves that row order.
So splitting the 575,061 merged rows at the midpoint gives a genuine
chronological earlier/later split, not an arbitrary one — confirmed by a
real shift in anomaly rate across the split:

- Earlier half (287,530 blocks): 3.54% anomaly rate
- Later half (287,531 blocks): 2.32% anomaly rate

V3's GA fitness selection (a 40k stratified subsample) and the candidate's
final `.fit()` both ran on the **later** half only, via:

```
python3 scripts/retrain_and_evaluate.py --data-slice late
```

This was chosen over a re-seeded random subsample of the full dataset
because it simulates the realistic "adaptation cycle" scenario the loop is
meant to support — retraining on a new window of incoming data — rather
than just resampling the same stationary distribution GA already saw for
V2. Evaluation (the numbers in the table above) always runs against the
full dataset regardless of slice, so the promotion decision compares V3
fairly against V1's and V2's full-dataset scores.

## Result: V3 was rejected

`evaluate_and_promote()` correctly did **not** promote V3 (F1 0.1582 doesn't
beat V2's 0.6788). `data/models/current_version.json` still points at V2,
and the V3 candidate was archived rather than discarded:
`data/models/archive/isolation_forest_candidate_rejected_20260904T163508218073.pkl`.
`data/models/version_history.json` now has three entries: V1->V2 promotion,
a tied-F1 rejection (from an earlier candidate), and this V3 rejection.

### Why V3 underperforms so badly

GA's fitness on its own training subsample (drawn from the later half) hit
0.6891 — essentially matching V2's fitness score of 0.6822. The collapse
only shows up once that model is scored against the *full* dataset: V3's
`contamination=0.029` was calibrated so ~2.9% of the *later-half* data gets
flagged, but applied to the full population its actual predicted-anomaly
rate is `(135,844 + 13,119) / 575,061 ≈ 26%` — a ~9x miscalibration. That
means the earlier and later halves of this HDFS trace differ enough in
their component/event-type/IP mix that a model fit only on the later half
systematically misreads a large share of the earlier half as anomalous.
This is itself a useful finding: a single time-window retrain without
periodic evaluation against the full/global population can silently
regress generalization even while its own fitness metric looks fine.

## Feature subset and hyperparameters: V2 vs V3

| Field | V2 | V3 |
|---|---|---|
| Selected features | `component_encoded`, `event_frequency`, `destination_ip_encoded` | `component_encoded`, `event_type_encoded`, `event_frequency`, `destination_ip_encoded` |
| n_estimators | 150 | 75 |
| max_samples | 0.727 | 0.548 |
| max_features | 0.796 | 0.585 |
| contamination | 0.0235 | 0.0289 |

V3's GA run picked up one extra feature (`event_type_encoded`) and settled
on a smaller forest (fewer trees, smaller per-tree sample/feature
fractions) with a higher contamination threshold. Given V3 was trained on
roughly half the data (287k vs 575k rows) and a different time window, some
divergence in the optimal hyperparameters is expected — the GA is fitting
to a different (and, per the anomaly-rate check above, distributionally
different) subsample each time. The differences aren't dramatic on their
own; it's the combination with the narrower training window that produced
the generalization gap above, not the hyperparameters in isolation.
