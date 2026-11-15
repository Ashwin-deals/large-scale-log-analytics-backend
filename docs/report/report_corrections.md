# case_3_software_project.pdf: corrections

I checked every number in the draft against the repo (backend `main` at 276cc1b, 15 Nov 2026).
Most of the draft is still right. The stale parts come from three changes:

- **V3 re-promotion (14 Nov).** Every HDFS number that depended on the live model has changed.
- **The other sources were rerun (12–14 Nov).** This changed Apache's detection numbers and the storage numbers for all four other sources.
- **The test suite grew** from 163 tests in 9 files to 207 tests in 13 files.

## A. Wrong numbers or statements (must fix)

| # | Where | Draft says | Change to |
|---|---|---|---|
| 1 | Abstract; §5.1 Table 4; §6 "Error handling"; §7 Conclusion; Table 9 | 163 automated tests (1 skipped), 9 test files | **207 automated tests in 13 files** (1 is skipped only when the CloudTrail data is not downloaded) |
| 2 | Abstract | "raises F1 from 0.119 (untuned) to 0.529" | "raises F1 from 0.119 (untuned) to **0.562** (deployed model V3; **0.555** on held-out data)" |
| 3 | Abstract | "Smart Storage saves 68.76% of lines at a weighted retention of 0.9676" | "Smart Storage saves **68.71%** of lines at a weighted retention of **0.955** of anomalous blocks on held-out data" |
| 4 | Table 5, V2 row | "In-sample; held-out F1 0.528" | Same text, plus "**retired**" |
| 5 | Table 5, V3 row | precision/recall "n/a" | Precision **0.7069**, recall **0.4666**; Notes: "**Deployed** (re-promoted 14 Nov); wins all 5 held-out splits" |
| 6 | Sentence under Table 5 | "As of 14 November, V3 had been rolled back earlier ... the team was deciding whether to re-promote it." | "V3 was promoted automatically on 9 Sep 2026 and rolled back to V2 by hand the same day; no reason for the rollback was recorded. On 14 Nov 2026 the team re-promoted V3 because it beats V2 on all five held-out splits (0.555 ± 0.009 against 0.528 ± 0.014)." |
| 7 | **Figure 5** (image and caption) | V2's results: 4,809 / 45 / 71 / 75; accuracy 97.68%, precision 62.50%, recall 51.37%, F1 0.5639 | **Replace the image** with `docs/report/fig5_v3_vs_ground_truth.png` (rerun on 15 Nov against V3). Caption: "Deployed model (V3) versus ground truth on 5,000 sampled HDFS blocks (seed 42): 4,822 true negatives, 32 false positives, 70 false negatives and 76 true positives; accuracy 97.96%, precision 70.37%, recall 52.05%, F1 0.5984" |
| 8 | Table 6, Apache row | V1 F1 0.336, GA F1 0.722, caught 61 of 74 | V1 F1 **0.301**, GA F1 **0.607**, caught **44 of 74** (from the 13 Nov rebuild, which kept Apache's repeated lines) |
| 9 | Table 7, GA policy row | 68.76%; weighted 0.9676, full 0.9356, coverage 0.9996; f = 0.0565, hot 0.172, warm 0.049 | **68.71%**; weighted **0.955**, full **0.953**, coverage **0.978**; f = **0.049**, hot threshold **0.126**, warm **0.118**. Add: "held-out half, policy tuned on V3's scores" |
| 10 | Table 7, Top-k row | retention 0.9425 | **0.8876** at 68.71% saved |
| 11 | Table 7, Random row | n/a / 0.2745 | **68.71%** / **0.2749** |
| 12 | Table 7, Keep-only-flagged row | 93.00% / 0.4559 | **93.44%** / **0.4667** |
| 13 | Text after Table 7 | OpenStack 31.4% (0.929), Apache 39.5% (0.935), CloudTrail 11.8% (0.943), Django 82.1% (0.950). "Only Django meets the target among the four." | OpenStack **22.3% (0.990)**, Apache **23.3% (0.945)**, CloudTrail **11.1% (0.958)**, Django **71.0% (0.969)**. "**Three of the four meet the 0.95 target; only Apache (0.945) misses it.**" |
| 14 | §7 Conclusion | "lifts F1 from 0.119 to about 0.53 (0.555 held-out for V3), Smart Storage saves 68.76% of lines with a weighted retention of 0.9676 (above ... the top-k baseline of 0.9425)" | "lifts F1 from 0.119 to 0.562 with the deployed model V3 (0.555 held-out), Smart Storage saves 68.71% of lines with a weighted retention of 0.955 on held-out data (above the 0.95 target and the top-k baseline of 0.888)" |
| 15 | §7 Limitations | "Among the other four sources only Django meets the 0.95 retention target." / "The decision on re-promoting HDFS model V3 is open." | "Among the other four sources only Apache misses the 0.95 retention target (0.945)." Delete the V3 sentence; the decision has been made. |
| 16 | §7 Future roadmap | "improve retention on OpenStack, Apache and CloudTrail" | "improve retention on Apache" |
| 17 | §4.1 Test setup note; §7 Limitations | "suite was run with Python 3.10, scikit-learn 1.7.2" / "test machine used 1.7.2 while the repository pins 1.8.0" | CI now runs the suite on **Python 3.12 with scikit-learn 1.8.0**, the pinned versions. Drop both the note and the limitation, or say "CI runs Python 3.12 with scikit-learn 1.8.0". |
| 18 | §4.1 Build & Deployment; §6 Security | secrets include "Hugging Face token" | Remove it: the dataset is public, so downloads need no token. Secrets are the database address, JWT secret and CORS origin. |
| 19 | NFR-04; §2.1 Scope; §3.1; §4.1 | "datasets are hosted on Hugging Face" (all of them) | "HDFS, OpenStack, Apache and Django are hosted on Hugging Face; CloudTrail is downloaded from the original flAWS archive [7], with each file's SHA-256 checksum checked" |
| 20 | §3.2; §6 | version history holds "promote, reject, refit and rollback events" | "promote, reject, refit, rollback and **manual re-promotion** events" |
| 21 | Title page | Two team members are listed with the same roll number | One of them is wrong. Fix it before submission. |
| 22 | §4.1 and Table 9 | Frontend "[add the frontend repository URL]" | https://github.com/Akilan-J/large-scale-log-analytics-frontend |
| 23 | Table 8, first row | "F1 at k=20, all blocks: 0.684" | "All blocks at k=20: precision / recall / F1 **0.849 / 0.572 / 0.684**". Keep the second row (still-running blocks, 0.696 / 0.527 / 0.600): both are correct, they count different blocks. Figure E1's "precision 0.85" is the all-blocks figure. |
| 24 | Figure E1 (`architecture_diagram.png`) | "k = 20: precision 0.85 on HDFS"; "raw data on Hugging Face" | Replace the image with `docs/report/architecture_diagram.png`, which reads "k = 20, all blocks: precision 0.85" and "raw logs: Hugging Face, flAWS" |

## B. Check these (I can't verify them from the repo)

- **Figure 6 and the model-selection paragraph** (IF about 0.9 accuracy and about 0.5 s; autoencoder 4 s; HBOS 3.7 s; One-Class SVM 0.55). No script or result file in the repo produces these numbers. The repo's own comparison (`data/evaluation/autoencoder_comparison.json`, 5 seeds, held-out) gives:

  | Model | F1 | AUC | Fit time |
  |---|---:|---:|---:|
  | Isolation Forest (V3 settings) | 0.555 | 0.936 | 1.9 s |
  | Autoencoder (all 7 features) | 0.285 | 0.801 | 8.6 s |

  The paragraph's "0.555 against 0.346" is right, but 0.346 is the autoencoder on V3's two features. Either say where Figure 6 comes from, or cut the time claims that conflict with 1.9 s.
- **Abstract wording.** It says "three-tier architecture" and then "All four tiers are implemented". Pick one. Three tiers plus a streaming side path is consistent with §3.1.
- **Kafka replay figures** (50,000 lines, about 5,000 lines/s, 4,087 blocks, 625 alerts) were measured while V2 was live. They are still valid; add "(run with V2 live)" if you want to be exact.
- **Table 9 placeholders.** The certifications and TEA deliverables 1–3 are still "[Fill in]".

## D. Against the template (3_report.pdf)

All section and subsection headings (1 to 8, 2.1 to 7) match the template, in the same order. These parts differ from what the template asks for:

| # | Where | Template asks for | Draft has | Fix |
|---|---|---|---|---|
| D1 | §5.2, Figure 7 | "Screenshots: UI Workflows, **API Execution & Test Outputs**" | UI screenshots only, panels (a)–(f) | Add the API and test outputs. Panels (g) and (h) of `fig3_system_demonstration.png` (also in `fig3_panels/`) show curl calls (401/200/404/400) and the pytest/CI run, or use that whole figure. Update the §5.2 sentence to mention them. |
| D2 | Figure numbers | Fig. 1 Architecture, **Fig. 2 Database Schema**, **Fig. 3 Demonstration** | Schema is Fig. 3 and the demonstration is Fig. 7; extra Figs. 2, 4, 5, 6 | Extra figures are fine, but put the template's three at their expected places. Move the functional view (now Fig. 2) after the schema, or to the end of §3.1 as Fig. 1b. Otherwise make sure the captions keep the template's exact wording, which they currently do. |
| D3 | §3.2 | Relationships, **constraints**, indexing strategy; figure with keys and **cardinalities** | Relationships and indexes, but no constraints paragraph | Add the constraints: unique email (duplicate sign-up returns 409), one verdict per (source, block_id) by upsert, alert `_id` = `run_id:type:block_id` so a replay never duplicates, enumerated values and lengths validated by the API, passwords stored only as bcrypt hashes. Ready-made text is in `section3_and_5_2.tex`, §3.2 "Constraints". |
| D4 | §4.3 Database & Persistence | "**transaction handling, query optimization**, and data integrity" | Data integrity only | Add, for example: "Every write is a single-document operation, which MongoDB applies atomically, so no multi-document transactions are needed. Feedback is an upsert and stream status is a whole-document replace. Queries use indexes on `user_details.email` (unique), `stream_status.updated_at` and `stream_alerts (run_id, detected_at)`." |
| D5 | §6 Concurrency, Data Integrity & Security | "**transaction isolation**, concurrency handling, authN/authZ, **input sanitization**" | No isolation or sanitization detail | Add, for example: "Writes are atomic per document, and the live-model pointer (`current_version.json`) changes in one step, so a reader never sees a half-promoted model. Inputs are validated before use: upload filenames are cleaned with `secure_filename`, upload size is capped by `MAX_CONTENT_LENGTH`, and sources, verdicts, password length and note length are checked against fixed rules." |
| D6 | §6 Error Handling, Logging | "structured exception handling, **logging**" | No mention of logging or the global handler | Add, for example: "A global Flask error handler turns every unhandled exception into a JSON error, logs the stack trace with `app.logger`, and keeps the CORS headers so the dashboard can show the error. Missing-source and not-built errors return 400 and 404 through blueprint handlers." |
| D7 | Abstract | 150–250 words | 243 words | After fixes 2–3 in section A it goes just over 250. Write "0.562 (0.555 held-out)" instead of the longer wording, and drop "(1 skipped)". |
| D8 | Table 9 | Every row filled, status [Completed] | Certifications and TEA 1–3 still "[Fill in]" but marked Completed | Fill them in before submission. |

## E. Second check: G045_Report.pdf (15 Nov)

Every item in sections A and D has been applied, and the abstract is 249 words, so there is no room to add to it. What is still open:

| # | Where | Report has | Fix |
|---|---|---|---|
| E1 | Table 10, TEA 2 "Project Website"; §2.1 | The live site described as running the current system | https://morphguard.duckdns.org still serves the 25 Oct build (no V3, no held-out results, no Autoencoder comparison). Redeploy (`deploy/vm_deploy.sh`) before the review, so what examiners see matches the report. |
| E2 | Figure E1 | The old diagram: "k = 20: precision 0.85 on HDFS", "raw data on Hugging Face" | Replace the image with `docs/report/architecture_diagram.png` (row 24) |
| E3 | Figure 2 | An older render of the schema: "uploaded_by" overlaps "stored_path", "deployed model version" overlaps the stream_alerts header, and the last rows of uploads ("error") and alert_feedback ("at") are cut off | Use `docs/report/fig2_database_schema.pdf` |
| E4 | Figures 1 and 2 | 1440 px PNG copies | Use the vector PDFs in `docs/report/`; they stay sharp in print |
| E5 | Table 8, lead-time rows | 0.339, 10,466 s and 14,110 s with no label | Label them "still-running blocks". Over all test blocks, the median lead time is 0 s and the mean is 5,165 s. |
| E6 | Table 9, "Autoencoder (V3's two features)" | AUC n/a, fit time n/a | AUC **0.718**, fit time **5.2 s** (`data/evaluation/autoencoder_comparison.json`) |
| E7 | Table 5, V1 row | Precision n/a, recall n/a | Precision **0.0656**, recall **0.6514** |
| E8 | Table 4, System / E2E | "the full pipeline re-run from the Hugging Face datasets" | "from the downloaded datasets (Hugging Face and flAWS)" |
| E9 | §3.1 Persistence tier | MongoDB holds "user accounts, uploads and streaming alerts" | Add analyst feedback, as §3.2 does |
| E10 | Table 4, System / E2E | Walk-through "with a headless browser and by hand", including upload | The headless walk-through covered every page for all five sources. The steps that write to the database (upload, change password, Real / False alarm, Run Retrain, sign out and back in) still have to be done by hand. Do them before the review (`docs/review/walkthrough_checklist.md`). |

## C. Still correct (no change)

- V1/V2/V3 in-sample and held-out F1 and the dataset sizes.
- OpenStack, CloudTrail and Django detection rows in Table 6.
- Table 8 (Early Warning; it has its own models, so the V3 change does not affect it).
- The API table.
- The SRS and the references.

## Files

- `fig5_v3_vs_ground_truth.png`: the replacement Figure 5.
- `fig1_software_architecture.*`, `fig2_database_schema.*`, `fig3_system_demonstration.*`, `section3_and_5_2.tex`: optional replacements for Figures 1, 3 and 7. These have the current numbers, and the demo figure includes API and test outputs, which the §5.2 heading asks for.
