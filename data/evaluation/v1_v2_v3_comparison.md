# HDFS model versions: V1, V2, V3

V2 is the deployed model. V3 was promoted automatically on 9 Sep 2026 and
rolled back to V2 by hand about nine hours later. This file records both
the published (in-sample) numbers and held-out numbers. It also records
what is known about the rollback.

Sources: `data/evaluation/baseline_isolation_forest_metrics.json`,
`isolation_forest_v2_metrics.json`, `isolation_forest_v3_metrics.json`,
`holdout_comparison.json` (from `scripts/evaluate_holdout.py`),
`data/models/version_history.json`, `data/models/current_version.json` and git.

## 1. Published metrics (in-sample)

Each model is scored on all 575,061 labeled blocks, the same blocks it was
fitted on. These numbers are therefore training scores.

| Version | Accuracy | Precision | Recall | F1 | Flagged | Caught | False alarms | Missed | State |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V1 baseline | 0.7180 | 0.0656 | 0.6514 | 0.1191 | 167,283 | 10,968 | 156,315 | 5,870 | Retired |
| V2 GA-optimized | 0.9755 | 0.6037 | 0.4705 | **0.5288** | 13,122 | 7,922 | 5,200 | 8,916 | **Deployed** |
| V3 GA-optimized | 0.9787 | 0.7069 | 0.4666 | 0.5622 | 11,114 | 7,857 | 3,257 | 8,981 | Rolled back |

| | V1 | V2 | V3 |
|---|---|---|---|
| Features | all 7 | `component`, `event_frequency`, `destination_ip` | `block_size`, `event_frequency` |
| Trees | 100 | 150 | 75 |
| max_samples | auto (256) | 0.727 | 0.135 |
| max_features | 1.0 | 0.796 | 0.595 |
| contamination | auto | 0.0235 | 0.0209 |

These are the numbers after the 8 Sep refit on stable category encoding.
Older slides show V2 at F1 0.6788 and V1 at 0.1794. Those came from before
that refit, and `version_history.json` records the change.

## 2. Held-out metrics

`scripts/evaluate_holdout.py` refits each version's exact hyperparameters
and features (sklearn `clone`, `random_state` included) on a stratified 70%
training split. It then scores the 30% the refit never saw.

Data used: all **575,061 blocks (16,838 anomalies)**. Split seed 42: train
402,542 blocks (11,787 anomalies), test 172,519 blocks (5,051 anomalies).

(An earlier run on a 150,000-block sample suggested V2 and V3 were tied;
the full data shows they are not.)

| Version | Train F1 | **Test F1** | Test precision | Test recall | Test flagged / caught / false alarms |
|---|---:|---:|---:|---:|---|
| V1 | 0.1364 | **0.1370** | 0.0766 | 0.6492 | 42,802 / 3,279 / 39,523 |
| V2 (deployed) | 0.5255 | **0.5224** | 0.5896 | 0.4690 | 4,018 / 2,369 / 1,649 |
| V3 | 0.5610 | **0.5539** | 0.6651 | 0.4746 | 3,604 / 2,397 / 1,207 |

**No overfitting.** For every version, test F1 is level with train F1.
Isolation Forest never sees the labels, so the in-sample numbers in
section 1 are not inflated by memorisation. The GA did use labels to choose
hyperparameters, but those choices carry over to unseen blocks.

**V3 is better on unseen data.** One split is not enough to separate
them, so the script repeats the comparison over five random splits:

| Version | Test F1, seeds 42 / 1 / 7 / 123 / 2024 | Mean ± std | Best on |
|---|---|---:|---:|
| V1 | 0.137 / 0.136 / 0.120 / 0.118 / 0.126 | 0.128 ± 0.008 | 0 of 5 |
| V2 | 0.522 / 0.547 / 0.542 / 0.509 / 0.521 | 0.528 ± 0.014 | 0 of 5 |
| V3 | 0.554 / 0.550 / 0.572 / 0.551 / 0.548 | **0.555 ± 0.009** | 5 of 5 |

V3's in-sample lead (0.5622 vs 0.5288) holds on unseen blocks: it wins all
five splits, by 0.027 F1 on average, about twice the split-to-split spread.
On the seed-42 test set it catches 28 more anomalies (2,397 against 2,369)
with 442 fewer false alarms (1,207 against 1,649).

## 3. The V3 promotion and rollback

### What the records show

| When (IST) | What happened | Source |
|---|---|---|
| 8 Sep, 23:45 to 00:49 | The review file (`review_dataset_full_columns.log`) was uploaded 8 times | `data/uploads/` timestamps |
| 9 Sep, 00:57 | V3 promoted by `evaluate_and_promote()`: candidate F1 0.5622 beat V2's 0.5288 | `version_history.json`, 19:27:49 UTC |
| 9 Sep, 10:07 | The same review file, now scored by V3: 108 anomalies instead of the 120 documented for the review | Upload result files on Ashwin's Mac |
| 9 Sep, from 10:13 | Review file uploads back at 120 anomalies (V2) | Upload result files on Ashwin's Mac |
| 9 Sep, 10:10 | Rolled back to V2 by editing `current_version.json` by hand (note: "Manually rolled back from V3 to V2.") | `current_version.json`, `promoted_at` 04:40:48 UTC |
| 25 Oct, 10:27 | Rollback committed, just before the GCP deployment | commit `28e14ea` |

Because the rollback was a hand edit, `version_history.json` had no entry
for it, and the log jumped from "V3 promoted" straight back to V2. A
`rollback` entry has now been backfilled with the original timestamp and
marked `"backfilled": true`. `optimization.model_evolution.record_rollback()`
now logs any future rollback automatically.

### Reason

**Not recorded at the time. Awaiting confirmation from the team.**

On the review file, V3 was the better model, according to the upload result files on Ashwin's Mac:

| Model | Flagged | Real anomalies caught | False alarms | F1 on the review file |
|---|---:|---:|---:|---:|
| V2 | 120 | 75 | 45 | 0.5639 |
| V3 | 108 | 76 | 32 | 0.5984 |

The rollback was a human override of promote-if-better, not a decision
based on metrics: V3 scored higher in-sample. The timing above is
consistent with a rollback made to keep the review dataset's anomaly count
at the documented 120. Until whoever made the change confirms that, it
remains an inference and is not recorded as the reason.

### Why V2 stays deployed

V3 is the better detector on unseen data (section 2), so the rollback cost
some accuracy. V2 is still the one deployed because:

- the rollback has not been reversed, and its reason is unconfirmed; and
- Smart Storage, Early Warning and the stream processor were tuned and
  evaluated on V2's scores, so re-promoting V3 means rerunning
  `scripts/optimize_storage_policy.py` (about 1.5 minutes on all blocks)
  and checking the stream.

Re-promoting V3 is a team decision. `record_rollback(3, reason)` in
`optimization/model_evolution.py` does it and logs it.

### A bug the rollback exposed (fixed)

`evaluate_and_promote()` numbered a new version as `current + 1`. With V2
live after the rollback, the next successful retrain would have become
"V3" again. It would have overwritten `isolation_forest_v3.pkl` and its
metrics, and the history would have contained two different V3s. New
versions are now numbered one past the highest version ever issued (from
history and the model files), so the next promotion becomes V4. A test in
`tests/test_model_evolution.py` covers this.

## 4. Earlier V3 candidate (rejected 4 Sep)

Before the encoding refit, a different V3 candidate was trained only on the
later chronological half of the blocks (`retrain_and_evaluate.py
--data-slice late`). It scored F1 0.1582 on the full dataset against V2's
0.6788 at the time, and `evaluate_and_promote()` correctly rejected it.

Its contamination (0.029) had been calibrated on the later half, which has
a 2.32% anomaly rate against the earlier half's 3.54%. Applied to all
blocks, it flagged about 26% of them. A model retrained on one time window
can regress on the whole population even while its own fitness looks good.
That candidate is archived at
`data/models/archive/isolation_forest_candidate_rejected_20260904T163508218073.pkl`.
It is not the V3 in sections 1 to 3.
