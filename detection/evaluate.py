import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DEFAULT_METRICS_PATH = Path("data/evaluation/baseline_isolation_forest_metrics.json")
LABELS = ["Normal", "Anomaly"]
POSITIVE_LABEL = "Anomaly"


def evaluate(
    predictions: pd.DataFrame,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
) -> dict:
    """
    Scores predicted_label against true_label. Anomaly is treated as the
    positive class for precision/recall/F1 since it's the class of interest
    and the minority class (~2.9% of blocks).
    """
    y_true = predictions["true_label"]
    y_pred = predictions["predicted_label"]

    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0),
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": matrix.tolist(),
        },
        "support": {
            "total": int(len(y_true)),
            "normal": int((y_true == "Normal").sum()),
            "anomaly": int((y_true == "Anomaly").sum()),
        },
    }

    metrics_path = Path(metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)

    return metrics


def print_metrics_table(metrics: dict) -> None:
    print(f"{'Metric':<12}{'Value':>10}")
    for key in ["accuracy", "precision", "recall", "f1"]:
        print(f"{key:<12}{metrics[key]:>10.4f}")

    labels = metrics["confusion_matrix"]["labels"]
    print("\nConfusion Matrix (rows=true, cols=predicted):")
    print(f"{'':<10}" + "".join(f"{label:>10}" for label in labels))
    for label, row in zip(labels, metrics["confusion_matrix"]["matrix"]):
        print(f"{label:<10}" + "".join(f"{value:>10}" for value in row))

    support = metrics["support"]
    print(f"\nSupport: total={support['total']}, normal={support['normal']}, anomaly={support['anomaly']}")
