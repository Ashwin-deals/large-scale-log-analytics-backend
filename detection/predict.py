from pathlib import Path

import pandas as pd

from detection.isolation_forest import TRAINING_FEATURE_COLUMNS

DEFAULT_PREDICTIONS_PATH = Path("data/predictions/isolation_forest_v1_predictions.csv")

RAW_PREDICTION_LABELS = {1: "Normal", -1: "Anomaly"}


def predict(
    model,
    merged: pd.DataFrame,
    output_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    feature_columns: list[str] = TRAINING_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """
    Generates a Normal/Anomaly prediction and an anomaly score for every
    row in `merged`, using `feature_columns` as model input (defaults to
    TRAINING_FEATURE_COLUMNS, i.e. the V1 baseline's full feature set; a GA
    candidate with a different selected feature subset can pass its own).

    `merged["Label"]` (ground truth) is optional: training/evaluation always
    has it (via load_labeled_features's join against anomaly_label.csv), but
    real inference on a freshly uploaded, unlabeled log does not.
    """
    inference_matrix = merged[feature_columns]
    raw_predictions = model.predict(inference_matrix)
    anomaly_scores = -model.decision_function(inference_matrix)

    predictions = pd.DataFrame(
        {
            "block_id": merged["block_id"].to_numpy(),
            "predicted_label": pd.Series(raw_predictions).map(RAW_PREDICTION_LABELS).to_numpy(),
            "anomaly_score": anomaly_scores,
        }
    )
    if "Label" in merged.columns:
        predictions.insert(1, "true_label", merged["Label"].to_numpy())

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions
