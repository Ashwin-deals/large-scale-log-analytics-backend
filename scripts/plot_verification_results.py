"""Plot the model-vs-ground-truth comparison for a verification sample.

Reads the metrics JSON written by scripts/generate_verification_sample.py
(run that first) and renders a single PNG with:
  - the confusion matrix as an annotated heatmap
  - predicted vs. actual block counts, split into two panels since Normal
    and Anomaly counts differ by ~30x on the same sample
  - the headline metrics (accuracy/precision/recall/F1/error rate)

Usage:
    python3 scripts/generate_verification_sample.py   # produces the metrics first
    python3 scripts/plot_verification_results.py
    python3 scripts/plot_verification_results.py --metrics data/evaluation/verification_sample_metrics.json --out data/testing/verification_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_METRICS_PATH = Path("data/evaluation/verification_sample_metrics.json")
DEFAULT_OUT_PATH = Path("data/testing/verification_comparison.png")

NORMAL = "#1baf7a"
PREDICTED = "#2a78d6"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def plot(metrics: dict, out_path: Path) -> None:
    labels = metrics["confusion_matrix"]["labels"]
    matrix = np.array(metrics["confusion_matrix"]["matrix"])
    (tn, fp), (fn, tp) = matrix

    actual = {"Normal": metrics["support"]["normal"], "Anomaly": metrics["support"]["anomaly"]}
    predicted = {"Normal": tn + fn, "Anomaly": fp + tp}

    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle(
        f"Deployed model vs. ground truth  —  {metrics['support']['total']:,} sampled blocks "
        f"(seed {metrics.get('sample_seed', '?')})",
        fontsize=13, fontweight="bold",
    )

    # --- confusion matrix -----------------------------------------------
    ax1 = fig.add_subplot(2, 2, 1)
    correctness = np.array([[1, 0], [0, 1]])  # diagonal = correct
    cmap = plt.matplotlib.colors.ListedColormap(["#fbe9e9", "#e5f5e5"])
    ax1.imshow(correctness, cmap=cmap, vmin=0, vmax=1)
    row_totals = matrix.sum(axis=1, keepdims=True)
    pct = matrix / row_totals * 100
    for i in range(2):
        for j in range(2):
            color = GOOD if i == j else CRITICAL
            ax1.text(j, i - 0.12, f"{matrix[i, j]:,}", ha="center", va="center",
                      fontsize=15, fontweight="bold", color=color)
            ax1.text(j, i + 0.16, f"{pct[i, j]:.2f}% of actual {labels[i]}", ha="center", va="center",
                      fontsize=8.5, color="#52514e")
    ax1.set_xticks([0, 1], [f"Predicted\n{l}" for l in labels])
    ax1.set_yticks([0, 1], [f"Actual\n{l}" for l in labels])
    ax1.set_title("Confusion matrix", fontsize=11, loc="left")
    for spine in ax1.spines.values():
        spine.set_visible(False)

    # --- predicted vs actual, two panels (scales differ ~30x) -----------
    for idx, cls in enumerate(labels):
        ax = fig.add_subplot(2, 2, 2 + idx)
        values = [actual[cls], predicted[cls]]
        bars = ax.bar(["Actual", "Predicted"], values, color=[NORMAL, PREDICTED], width=0.55)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}",
                     ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(f"{cls} blocks", fontsize=11, loc="left")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.2)

    # --- metrics summary strip -------------------------------------------
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    rows = [
        ("Accuracy", f"{metrics['accuracy']*100:.2f}%"),
        ("Precision (Anomaly)", f"{metrics['precision']*100:.2f}%"),
        ("Recall (Anomaly)", f"{metrics['recall']*100:.2f}%"),
        ("F1 (Anomaly)", f"{metrics['f1']:.4f}"),
        ("Error rate", f"{metrics['error_rate']*100:.2f}%"),
        ("False positive rate", f"{metrics['false_positive_rate']*100:.2f}%"),
        ("False negative rate", f"{metrics['false_negative_rate']*100:.2f}%"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.92 - i * 0.135
        ax4.text(0.0, y, k, fontsize=10.5, color="#52514e", transform=ax4.transAxes)
        ax4.text(1.0, y, v, fontsize=10.5, fontweight="bold", ha="right", transform=ax4.transAxes)
    ax4.set_title("Metrics", fontsize=11, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    if not args.metrics.exists():
        raise SystemExit(
            f"{args.metrics} not found — run scripts/generate_verification_sample.py first."
        )
    metrics = json.loads(args.metrics.read_text())
    plot(metrics, args.out)


if __name__ == "__main__":
    main()
