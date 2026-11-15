# Report figures and corrections (Phase II external review)

Material for the 23CSE498 Phase II report. It follows the department template, `3_report.pdf`.

| File | What it is |
|---|---|
| `report_corrections.md` | Numbers and statements in the 14 Nov draft that are out of date since V3 was re-promoted, plus gaps against the template, each with replacement text |
| `section3_and_5_2.tex` | LaTeX text and figure environments for §3.1, §3.2 and §5.2 |
| `fig1_software_architecture.{pdf,png,svg}` | Figure 1: high-level architecture (UML component view) |
| `fig2_database_schema.{pdf,png,svg}` | Figure 2: MongoDB collections, file-based model registry, keys, cardinalities and indexes |
| `fig3_system_demonstration.{pdf,png}` | Figure 3: running system (a–f), REST API calls (g), test suite and CI (h) |
| `fig3_panels/` | Each Figure 3 panel as its own image |
| `architecture_diagram.png` | Figure E1 (functional view): `docs/architecture_diagram.svg` rendered at 2x |
| `fig5_v3_vs_ground_truth.png` | Replacement for the draft's Figure 5: the deployed model (V3) on 5,000 sampled HDFS blocks |

Every number comes from files committed under `data/evaluation/`, `data/storage/` and `data/models/`.

To regenerate Figure 5:

```bash
python3 scripts/generate_verification_sample.py
python3 scripts/plot_verification_results.py --out docs/report/fig5_v3_vs_ground_truth.png
```
