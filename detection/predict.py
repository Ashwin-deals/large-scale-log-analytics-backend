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
    trained_vocab = getattr(model, "encoder_vocab_", None)
    if trained_vocab is None:
        warnings.warn(
            "Model has no encoder vocabulary — it predates stable category "
            "encoding, so its predictions cannot be verified against the "
            "current features. Retrain it to restore the guarantee.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    # Only renumbering invalidates a model. New values appended by, say, an
    # uploaded log carrying an unseen host are harmless and must not block
    # inference, so compare code-by-code rather than by whole-vocabulary hash.
    conflicts = CategoryEncoder.load().conflicts_with(trained_vocab)
    if conflicts:
        shown = "; ".join(conflicts[:3])
        more = f" (+{len(conflicts) - 3} more)" if len(conflicts) > 3 else ""
        raise ValueError(
            f"Category encoding mismatch — {len(conflicts)} value(s) were renumbered "
            f"since this model was trained: {shown}{more}. Its splits no longer refer "
            f"to the same categories, so any prediction would be meaningless. "
            f"Retrain the model against the current vocabulary."
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

    `merged["Label"]` (ground truth) is optional: training/evaluation always
    has it (via load_labeled_features's join against anomaly_label.csv), but
    real inference on a freshly uploaded, unlabeled log does not.
    """
    check_encoder_compatibility(model)

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
