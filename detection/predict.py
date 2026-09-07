import warnings
from pathlib import Path

import pandas as pd

from detection.isolation_forest import TRAINING_FEATURE_COLUMNS
from feature_engineering.category_encoder import CategoryEncoder

DEFAULT_PREDICTIONS_PATH = Path("data/predictions/isolation_forest_v1_predictions.csv")

RAW_PREDICTION_LABELS = {1: "Normal", -1: "Anomaly"}


def check_encoder_compatibility(model) -> None:
    """Refuse to score against a vocabulary the model wasn't trained on.

    Categorical codes are only meaningful relative to the vocabulary that
    produced them. Scoring a model against features encoded differently
    returns confident nonsense rather than an error, which is exactly how the
    dashboard came to report half of all blocks as anomalous.
    """
    trained_with = getattr(model, "encoder_fingerprint_", None)
    if trained_with is None:
        warnings.warn(
            "Model has no encoder fingerprint — it predates stable category "
            "encoding, so its predictions cannot be verified against the "
            "current features. Retrain it to restore the guarantee.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    current = CategoryEncoder.load().fingerprint()
    if trained_with != current:
        raise ValueError(
            f"Category encoding mismatch: model was trained with vocabulary "
            f"{trained_with} but the current features use {current}. The codes "
            f"no longer refer to the same components/IPs, so any prediction "
            f"would be meaningless. Rebuild features or retrain the model."
        )


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
    """
    check_encoder_compatibility(model)

    inference_matrix = merged[feature_columns]
    raw_predictions = model.predict(inference_matrix)
    anomaly_scores = -model.decision_function(inference_matrix)

    predictions = pd.DataFrame(
        {
            "block_id": merged["block_id"].to_numpy(),
            "true_label": merged["Label"].to_numpy(),
            "predicted_label": pd.Series(raw_predictions).map(RAW_PREDICTION_LABELS).to_numpy(),
            "anomaly_score": anomaly_scores,
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_path, index=False)
    return predictions
