from sklearn.ensemble import IsolationForest

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
