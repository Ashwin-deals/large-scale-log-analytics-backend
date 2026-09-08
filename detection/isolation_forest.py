from pathlib import Path

import joblib
from sklearn.ensemble import IsolationForest

from feature_engineering.category_encoder import CategoryEncoder
from feature_engineering.feature_extractor import FEATURE_COLUMNS

EXCLUDED_FEATURE_COLUMNS = ["is_error"]
TRAINING_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS if column not in EXCLUDED_FEATURE_COLUMNS
]

RANDOM_STATE = 42


def build_isolation_forest() -> IsolationForest:
    """
    Baseline unsupervised Isolation Forest: default contamination and tree
    settings, all cores, fixed random_state so the saved model is
    reproducible across re-runs.
    """
    return IsolationForest(n_jobs=-1, random_state=RANDOM_STATE)


def save_model(model, model_path: str | Path):
    """Persist a fitted model together with the vocabulary it was trained on.

    Every save goes through here so a model can never reach disk without the
    fingerprint that lets inference verify the feature codes still mean the
    same thing.
    """
    encoder = CategoryEncoder.load()
    # The vocabulary itself, not just its hash: inference needs the individual
    # codes to tell a harmless append from a breaking renumber.
    model.encoder_vocab_ = encoder.vocab
    model.encoder_fingerprint_ = encoder.fingerprint()
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    return model_path
