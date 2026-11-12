# Walk-through checklist (status doc E5)

A person clicks through every page for every source before the review, because headless checks miss
layout and wording problems. Tick each box and note anything odd next to it.

For a source whose data is not on the machine, the page should show **"Not available on this server
yet"** with the command to fix it. That counts as a pass. A red error or a blank page is a fail.

| Page | HDFS | OpenStack | Apache | CloudTrail | Django |
|---|---|---|---|---|---|
| Dashboard: KPIs, Model Status state, blocks chart, recent activity | ☐ | ☐ | ☐ | ☐ | ☐ |
| Detection Results: rows, search, severity filter, paging, Export CSV, Real / False alarm | ☐ | ☐ | ☐ | ☐ | ☐ |
| Analytics: every chart has data and the right source name | ☐ | ☐ | ☐ | ☐ | ☐ |
| Models: lineage, held-out card (OpenStack: caught 0 of 4), timeline, Run Retrain Cycle | ☐ | ☐ | ☐ | ☐ | ☐ |
| Smart Storage: KPIs, tier chart, baselines, simulator sliders respond | ☐ | ☐ | ☐ | ☐ | ☐ |
| Live Stream: status and alerts for a run of that source | ☐ | ☐ | ☐ | ☐ | ☐ |

Pages that do not depend on the source:

- ☐ Log Sources: upload a small file for each source type; history updates; per-upload results open
- ☐ Log Sources: connector cards show Active / Live or Idle / Available / Future work with a reason
- ☐ Settings: account details; change password (then sign in with the new one); default source; retention target
- ☐ Top bar: "Stream live" while a processor runs, "Stream idle" otherwise
- ☐ Light and dark system theme both readable
- ☐ Sign out, then sign in again

Before the review:

- ☐ Rehearsal 1 of `docs/review/demo_script.md`, timed: ______
- ☐ Rehearsal 2, timed: ______
