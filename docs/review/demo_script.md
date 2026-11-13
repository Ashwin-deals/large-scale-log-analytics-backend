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
| 1:20 | Models | "A new model replaces the live one only if its F1 is higher. V3 was promoted, then rolled back by hand; the log records it. On unseen data V2 and V3 are tied, 0.575 vs 0.564." | Lineage, Held-out Evaluation card, Autoencoder table |
| 2:10 | Smart Storage | "The GA also learns what to keep. On held-out blocks it saves about two-thirds of the lines and still keeps 95% of anomalies, and beats every simple rule." | KPIs, baselines chart, move one slider |
| 3:00 | Log source → OpenStack, then Django | "All five sources run through the same pipeline. Each page says what kind of labels it has: OpenStack is a case study (it catches 0 of the 4 documented faulty instances), Django is synthetic." | Source badge, OpenStack caught-0-of-4, Django dashboard |
| 3:50 | Live Stream | "The same models run live on Kafka." | Start processor + producer; alerts arrive |
| 4:40 | Close | "Limitations we are upfront about: the labels behind the extra sources, storage measured in lines not bytes, and early warning that is only truly early at k = 20." | |

## Numbers to have ready

### HDFS detection

Data: 575,061 blocks, 2.93% anomalous.

| | In-sample F1 | Held-out F1 (5 splits) |
|---|---|---|
| V1 baseline | 0.1191 | 0.169 |
| **V2 GA-optimized** | **0.5288** | **0.575 ± 0.013** |
| V3 (rolled back) | 0.5622 | 0.564 ± 0.007 |

V2 uses 3 of 7 features, 150 trees and contamination 0.0235. Held-out numbers come from a 150,000-block
stratified sample (`holdout_comparison.json`).

### Autoencoder vs Isolation Forest

Same held-out splits, same alert rate (`autoencoder_comparison.json`).

| Detector | Test F1 | ROC AUC |
|---|---|---|
| Isolation Forest | 0.575 | 0.941 |
| Autoencoder, V2's 3 features | 0.509 | 0.750 |
| Autoencoder, all 7 features | 0.400 | 0.800 |

### Smart Storage on HDFS

The figures are held-out, after the WARM fix, on the 150k sample:
- 68.4% of lines saved
- full retention 0.950, weighted 0.952
- beats every baseline

### Early warning on HDFS

At k = 20 on the 172,519-block test set:
- precision 0.849, recall 0.572, false alarms 0.31%
- on blocks still running when the alert fired: F1 0.600, median 2.9 h before the block's last line

### The other sources

Held-out test sets. The labels are not ground truth.

| Source | Labels | V1 F1 | GA F1 | Caught | Lines saved |
|---|---|---|---|---|---|
| OpenStack | by run | 0.111 | 0.138 | 12 of 59 | 31.4% |
| Apache | injected | 0.301 | 0.607 | 44 of 74 | 23.3% |
| CloudTrail | weak rules | 0.115 | 0.615 | 155 of 211 | 11.8% |
| Django | synthetic | 0.158 | 0.436 | 17 of 24 | 71.0% |

### Stream verification

A replay of 1,000,000 HDFS lines:
- **Complete blocks:** block-verdict precision 0.986, recall 0.934, false-alarm rate 1.39%.
- **Early warning:** precision 0.955 on complete blocks, 0.923 on cut-off ones.
- **Django replay:** matches batch exactly, with 161 anomalous users.

## Likely questions

**Why Isolation Forest and not a neural network?**
We tested an Autoencoder on the same held-out splits. Isolation Forest scores F1 0.575 against the
Autoencoder's best of 0.509, with AUC 0.94 against 0.75–0.80. It is far more stable across splits
(±0.013 against ±0.100) and fits faster. It also needs no labels, which matters with only 2.9% anomalies.

**Why is F1 only about 0.58?**
It is unsupervised: the model never sees the labels. At a 2.35% alert rate it catches about half the
anomalies, and two in three of its alerts are real. A supervised model would score higher, but it needs
labels that a live system does not have.

**Your published F1 is in-sample. Isn't that cheating?**
We report both. On unseen blocks, test F1 equals train F1 (0.575), so there is no overfitting.

**Why was V3 rolled back if it scored higher?**
It was a manual override, recorded in the history log. On unseen data, V2 and V3 are tied. Smart Storage,
Early Warning and streaming are tuned on V2. *The reason for the rollback itself is still being confirmed
by the team.*

**How do you avoid promoting a lucky model?**
Promotion needs strictly higher F1, and every decision is logged. For non-HDFS sources the candidate and
the incumbent are scored on the same held-out split. Analyst feedback re-scores the incumbent on the
corrected labels before the comparison.

**Smart Storage says 95% of anomalies are kept. How can a summary keep an anomaly?**
A WARM summary only counts in proportion to how much of the block it keeps, with full credit at 25% of the
block. The 95% target is met by fully kept anomalies: full retention is 0.950. Savings are counted in log
lines, not bytes.

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
