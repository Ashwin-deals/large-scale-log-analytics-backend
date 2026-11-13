# MorphGuard API Reference

Flask server, `python3 app.py`, port 5000. Every route except register, login and health needs
`Authorization: Bearer <token>` (a JWT from login, valid 24 h). Errors are JSON `{"error": "..."}`.

## Log sources: `?source=`

Every data route takes `?source=hdfs|openstack|apache|cloudtrail|django` (default `hdfs`).

| Status | Meaning |
|---|---|
| 400 | Unknown source |
| 404 with `"not_ready": true` | That source's data is not on this server; `error` gives the command that fixes it |

Responses carry `source: {name, title, label_kind, block_noun}`. `label_kind` is `ground truth`, `by run`,
`INJECTED`, `WEAK rule-based` or `SYNTHETIC`. Show it next to any metric for that source.

## Auth and account

| Route | Body / params | Returns |
|---|---|---|
| `POST /api/auth/register` | `{email, password (8+ chars), name}` | `201 {token, user}`; `409` if the email exists |
| `POST /api/auth/login` | `{email, password}` | `{token, user}`; `401` on a bad pair |
| `GET /api/auth/me` | | `{email, name, created_at, registered}`, never the hash |
| `POST /api/auth/change-password` | `{current_password, new_password}` | `{ok: true}`; `403` wrong current password; `400` new one too short or unchanged |
| `GET /api/health` | | `{status: "ok"}`, no auth |

## Dashboard, detections, analytics

| Route | Params | Returns |
|---|---|---|
| `GET /api/dashboard` | `source`, `scope=dataset` (ignore the latest upload) | KPIs (`total_logs_processed`, `total_blocks_analyzed`, `anomalies_detected`, `anomaly_rate_pct`, `current_version`, `false_positive_rate_pct`, `model_status{state, tone, detail}`), `volume_by_day` (logs, blocks, anomalies), `severity_mix`, `model_evolution`, `recent_activity`, `view` (`dataset` or `upload`) |
| `GET /api/detections` | `source`, `search`, `severity=all\|critical\|high\|medium\|normal`, `page`, `limit` (max 200) | `{items, total, page, limit}`, highest score first; `true_label` is null for uploaded blocks |
| `GET /api/detections/summary` | `source` | `total_analyzed`, `normal_count`, `anomalous_count`, `avg_anomaly_score`, `contamination_threshold` |
| `GET /api/detections/export` | same filters as `/api/detections` | CSV of every matching row |
| `GET /api/analytics` | `source` | component / event type / log level / hour-of-day distributions, anomalies per day and per component |

`model_status.state` is one of four values:
- `healthy`: F1 ≥ 0.40
- `degraded`: F1 below 0.40
- `unverified`: no metrics recorded
- `unavailable`: the model file is missing

## Models and retraining

| Route | Params / body | Returns |
|---|---|---|
| `GET /api/models` | `source` | `current`, `history_table` (every issued version, `status` = `active`, `retired` or `rolled_back`), `timeline` (promotions, rejections, refits, manual rollbacks), `ga_best_config`, `job`, `evaluation` (see below) |
| `POST /api/models/retrain` | `?source=` | `202 {job_id, source}`; `409` if a job is running |
| `GET /api/models/retrain/<job_id>` | | `{state: queued\|running\|done\|error, result, feedback, evaluated_on, random_seed}` |

`evaluation` holds held-out test F1, the test set size and label caveats.
- HDFS adds the five-split stability check and `model_comparison` (Autoencoder vs Isolation Forest).
- OpenStack adds `ground_truth_check`, which reports how many of the 4 documented faulty instances the model catches.

**How a retrain decides.** The GA tunes a candidate. It replaces the live model only if its F1 is strictly
higher, a rule called promote-if-better.
- **HDFS:** candidates are scored on every block, as V1 and V2 were.
- **Other sources:** the candidate is fitted on the source's 70% split and scored on the same 30% held-out
  split its incumbent was scored on.
- **Analyst feedback:** verdicts override the labels first. The incumbent is then re-scored on the same
  corrected rows, so the comparison stays like for like.
- **Rejected candidates** are archived.
- **Numbering:** new versions are numbered past the highest version ever issued, so a rollback can never
  make a new model overwrite an old one.

## Analyst feedback

| Route | Body / params | Returns |
|---|---|---|
| `POST /api/feedback` | `?source=`, `{block_id, verdict: "anomaly"\|"normal", note?}` | `201`, the stored verdict (latest wins) |
| `GET /api/feedback` | `source` | `{counts: {anomaly, normal}, items, by_block}` |
| `DELETE /api/feedback` | `source`, `block_id` | `{deleted}` |

The next retrain of that source applies the verdicts (see above).

## Smart Storage

| Route | Params / body | Returns |
|---|---|---|
| `GET /api/storage/summary` | `source` | Held-out savings and retention, tiers, `warm_tier` diagnostics, every baseline, `stale` |
| `GET /api/storage/policy` | `source` | The tuned weights, thresholds and summary fraction |
| `POST /api/storage/simulate` | `?source=`; body below | Re-costs the policy and returns `outcome`, `meets_retention_target` and `fitness` |

The `simulate` body accepts any of these fields: `weights{anomaly_score, event_frequency, block_size}`,
`t_hot`, `t_warm`, `summary_fraction`, `retention_target`, `warm_credit`, `warm_ref_fraction`, and
`split=holdout|tune|all`.

**How storage is costed:**
- HOT keeps every line.
- WARM keeps 1 metadata line plus `round(f × lines)` summary lines.
- COLD keeps 1 metadata line.

**How a WARM anomaly earns retention credit:** `warm_credit × min(1, summary / (warm_ref_fraction × lines))`.

## Early warning

`GET /api/early-warning/summary?source=` returns held-out precision, recall, F1, false-alarm rate and lead
time for each prefix length k. It also returns the precision-vs-lead trade-off, along with the same
metrics for blocks that were still running when the alert fired.

## Uploads and connectors

| Route | Body / params | Returns |
|---|---|---|
| `POST /api/sources/upload` | multipart `file` (.log .json .gz .csv .txt, up to 500 MB), `source` | `201`; scored in the background by that source's model |
| `GET /api/sources/uploads` | | Upload history with blocks and anomalies per file |
| `GET /api/sources/uploads/<id>/detections` | `page`, `limit` | Per-block results for one upload |
| `GET /api/sources/connectors` | | Each connector with a `status` (`active`, `live`, `idle`, `available` or `future`) and a `detail` line |

## Live stream

| Route | Params | Returns |
|---|---|---|
| `GET /api/stream/status` | `source`, `run_id` | Newest processor run: events/s, blocks in flight, alert counts, storage saved, tiers, `live` |
| `GET /api/stream/alerts` | `source`, `run_id`, `type=early_warning\|block_anomaly`, `page`, `limit` | Alerts of that run. An early warning gains `lead_fraction` once its block is finalized. |

## Example

```bash
TOKEN=$(curl -s -X POST localhost:5000/api/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"..."}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])')
curl -H "Authorization: Bearer $TOKEN" "localhost:5000/api/detections?source=django&severity=critical&limit=5"
curl -H "Authorization: Bearer $TOKEN" -X POST "localhost:5000/api/models/retrain?source=django"
```
