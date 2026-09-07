from pathlib import Path

import joblib
import pandas as pd

from detection.isolation_forest import TRAINING_FEATURE_COLUMNS, build_isolation_forest

DEFAULT_FEATURES_PATH = Path("data/features/features.csv")
DEFAULT_LABELS_PATH = Path("data/raw/hdfs/anomaly_label.csv")
DEFAULT_MODEL_PATH = Path("data/models/isolation_forest_v1.pkl")


def load_labeled_features(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
) -> pd.DataFrame:
    """
    Loads features.csv and anomaly_label.csv and inner-joins them on
    block_id. This join has already been verified clean (0% row loss, no
    duplicate IDs on either side — see scripts/verify_features_labels.py)
    and is not re-validated here.
    """
    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path)
    return features.merge(labels, left_on="block_id", right_on="BlockId", how="inner")


def train_isolation_forest(
    merged: pd.DataFrame,
    model_path: str | Path = DEFAULT_MODEL_PATH,
):
    """
    Fits an unsupervised Isolation Forest on TRAINING_FEATURE_COLUMNS only
    (block_id and labels are never passed to fit) and saves it to disk.
    """
    model = build_isolation_forest()
    training_matrix = merged[TRAINING_FEATURE_COLUMNS]
    model.fit(training_matrix)

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model


def train(
    features_path: str | Path = DEFAULT_FEATURES_PATH,
    labels_path: str | Path = DEFAULT_LABELS_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
):
    merged = load_labeled_features(features_path, labels_path)
    model = train_isolation_forest(merged, model_path)
    return model, merged


if __name__ == "__main__":
    train()
