# MorphGuard — 5-minute demo script and likely questions

Every number here comes from a file in `data/evaluation/` or `data/storage/`. If a file changes, update
this script before rehearsing.

## Before the review (10 minutes before)

1. Start Docker Desktop. Then `cd large-scale-log-analytics-backend && docker compose up -d`.
2. From the project folder, run `./dev.sh`. The backend is on :5000 and the frontend on :3000.
3. Sign in. The top-bar **Log source** picker should be on **HDFS**.
4. In a terminal, have these two commands ready (Live Stream, step 5):
   - `python3 -m streaming.processor --source django --from-beginning --stop-after-idle 20`
   - `python3 -m streaming.producer --source django --rate 2000`

## The demo (about 5 minutes)

| Time | Page | Say | Show |
|---|---|---|---|
| 0:00 | Dashboard | "MorphGuard reads system logs, groups them into blocks and flags unusual ones with an Isolation Forest. A genetic algorithm keeps tuning it." | KPIs, Model Status (real state), blocks-and-anomalies chart |
| 0:40 | Detection Results | "Each row is one block. Analysts can confirm or reject an alert, and the next retrain learns from it." | Click **Real** on a top row; mention Export CSV |
| 1:20 | Models | "A new model replaces the live one only if its F1 is higher. V3 was promoted, rolled back by hand with no recorded reason, and re-promoted by the team because it wins on unseen data, 0.555 vs 0.528. Every step is in the log." | Lineage, Held-out Evaluation card, Autoencoder table |
| 2:10 | Smart Storage | "The GA also learns what to keep. On held-out blocks it saves over two-thirds of the lines, keeps 95% of anomalies in full, and beats every simple rule." | KPIs, baselines chart, move one slider |
| 3:00 | Log source → OpenStack, then Django | "All five sources run through the same pipeline. Each page says what kind of labels it has: OpenStack is a case study (it catches 0 of the 4 documented faulty instances), Django is synthetic." | Source badge, OpenStack caught-0-of-4, Django dashboard |
| 3:50 | Live Stream | "The same models run live on Kafka." | Start processor + producer; alerts arrive |
| 4:40 | Close | "Limitations we are upfront about: the labels behind the extra sources, storage measured in lines not bytes, and early warning that is only truly early at k = 20." | |

## Numbers to have ready

### HDFS detection

Data: 575,061 blocks, 2.93% anomalous.

| | In-sample F1 | Held-out F1 (5 splits) |
|---|---|---|
| V1 baseline | 0.1191 | 0.128 ± 0.008 |
| V2 GA-optimized | 0.5288 | 0.528 ± 0.014 |
| **V3 GA-optimized (live)** | **0.5622** | **0.555 ± 0.009** (best on all 5 splits) |

V3 uses 2 of 7 features (`block_size`, `event_frequency`), 75 trees and contamination 0.0209. It was
rolled back by hand on 9 Sep and re-promoted on 14 Nov. Held-out: refit on 70% of all 575,061 blocks,
scored on the unseen 30% (`holdout_comparison.json`).

### Autoencoder vs Isolation Forest

Same held-out splits, same alert rate (`autoencoder_comparison.json`).

| Detector | Test F1 | ROC AUC |
|---|---|---|
| Isolation Forest (V3) | 0.555 ± 0.009 | 0.936 |
| Autoencoder, V3's 2 features | 0.346 ± 0.189 | 0.718 |
| Autoencoder, all 7 features | 0.285 ± 0.157 | 0.801 |

### Smart Storage on HDFS

Held-out half of all blocks, after the WARM fix, scored by V3:
- 68.7% of lines saved
- 0.953 of anomalies kept in full; weighted retention 0.955, which meets the 0.95 target
- WARM is a small tier: 9.5% of blocks, about 2 of 23.5 lines each
- beats every baseline

### Early warning on HDFS

At k = 20 on the 172,519-block test set:
- precision 0.849, recall 0.572, false alarms 0.31%
- on blocks still running when the alert fired: F1 0.600, median 2.9 h before the block's last line

### The other sources

Held-out test sets. The labels are not ground truth.

| Source | Labels | V1 F1 | GA F1 | Caught | Lines saved |
|---|---|---|---|---|---|
| OpenStack | by run | 0.111 | 0.138 | 12 of 59 | 22.3% |
| Apache | injected | 0.301 | 0.607 | 44 of 74 | 23.3% |
| CloudTrail | weak rules | 0.115 | 0.615 | 155 of 211 | 11.1% |
| Django | synthetic | 0.158 | 0.436 | 17 of 24 | 71.0% |

### Stream verification

A replay of 1,000,000 HDFS lines (made while V2 was live):
- **Complete blocks:** block-verdict precision 0.986, recall 0.934, false-alarm rate 1.39%.
- **Early warning:** precision 0.955 on complete blocks, 0.923 on cut-off ones.
- **Django replay:** matches batch exactly, with 161 anomalous users.

## Likely questions

**Why Isolation Forest and not a neural network?**
We tested an Autoencoder on the same five held-out splits of all blocks. With V3's two features it
averages F1 0.346 against 0.555, swings twenty times as much from split to split (±0.189 against
±0.009; it collapses to 0.09 on one split) and ranks anomalies clearly worse (AUC 0.718 against 0.936).
With all seven features it drops to 0.285. Isolation Forest also needs no labels and no feature scaling.

**Why is F1 only about 0.55?**
It is unsupervised: the model never sees the labels. At a 2.1% alert rate it catches about half the
anomalies, and two in three of its alerts are real. A supervised model would score higher, but it needs
labels that a live system does not have.

**Your published F1 is in-sample. Isn't that cheating?**
We report both. On unseen blocks test F1 matches train F1 (0.554 against 0.561 for V3), so there is no overfitting.

**Why was V3 rolled back if it scored higher?**
It was a manual override, and nobody recorded why; the history log says so plainly rather than guessing.
On unseen data V3 really is better (0.555 against 0.528, on all five splits), so the team re-promoted it on
14 Nov, re-tuned Smart Storage for it, and logged that too.

**How do you avoid promoting a lucky model?**
Promotion needs strictly higher F1, and every decision is logged. For non-HDFS sources the candidate and
the incumbent are scored on the same held-out split. Analyst feedback re-scores the incumbent on the
corrected labels before the comparison.

**How can a summary "keep" an anomaly?**
A WARM summary only counts in proportion to how much of the block it keeps, with full credit at 25% of the
block. Almost all retention now comes from fully kept anomalies (0.953); weighted retention is 0.955, which
meets the 0.95 target on unseen blocks. Savings are counted in log lines, not bytes.

**Is early warning really early?**
Only at k = 20. At smaller k, the precision comes from short blocks that had already ended. Three
quarters of HDFS blocks have 20 events or fewer, so there is nothing early to give for them. The 2.9 h is
measured to the block's last log line, not to when someone would notice.

**Why did Apache's F1 drop from 0.722 to 0.607?**
The old figure was measured on a log missing 37.9% of its real lines. The shared cleaner had treated
repeated lines inside one second as duplicates and collapsed them, which hid the bursts. With the real
lines kept, detection is harder: 44 of 74 anomalies are caught instead of 61. Early warning got better
(F1 0.497 → 0.564), and 94.4% of anomalies are now kept in full, up from 87.1%.

**Are the four extra sources real results?**
- HDFS is validated.
- OpenStack is a case study. Its labels mark the run, not the faulty session, and the model catches 0 of
  the 4 documented faulty instances.
- Apache uses injected anomalies, CloudTrail uses labels from our own rule, and Django is synthetic. They
  prove the pipeline generalises; they do not prove detection quality.

**Why not Flink?**
The stream has to reuse the batch Python features and models exactly, and a test checks that they match.
One Python consumer handles the load, and Kafka partitions keep each block's events in order. Flink would
add checkpointed state, which we list as future work.

**What happens with a new kind of log line?**
New categories get new codes in a per-source vocabulary. Existing codes never change, and a model refuses
to score against a vocabulary that renumbered its codes.

**What is still not done?**
- Durable stream state across restarts.
- Cloud connectors: CloudWatch, Azure and GCP need provider credentials.
- A longer HDFS idle timeout or completion signal. The 1-hour timeout closes most blocks too early.
