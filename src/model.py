"""Evolutionary Isolation Forest Model.

Combines scikit-learn's Isolation Forest with Genetic Algorithm
hyperparameters: optimal feature subset selection and dynamic thresholding.
"""

from typing import Optional, Tuple, List, Union
import numpy as np


def c_factor(n: int) -> float:
    """Average path length of unsuccessful search in Binary Search Tree (BST).
    c(n) = 2 * (ln(n - 1) + 0.5772156649) - (2 * (n - 1) / n)
    """
    if n <= 1:
        return 1.0
    if n == 2:
        return 1.0
    euler_gamma = 0.5772156649
    return 2.0 * (np.log(n - 1.0) + euler_gamma) - (2.0 * (n - 1.0) / n)


class EvoIsolationForest:
    """Isolation Forest anomaly detector with GA-optimized feature mask and threshold."""

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: Union[int, float] = 256,
        feature_indices: Optional[np.ndarray] = None,
        threshold: float = 0.60,
        random_state: Optional[int] = 42
    ):
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.feature_indices = feature_indices
        self.threshold = threshold
        self.random_state = random_state
        self._model = None
        self._is_fitted = False
        self._score_min: Optional[float] = None
        self._score_max: Optional[float] = None

    def fit(self, X: np.ndarray) -> "EvoIsolationForest":
        """Fits the Isolation Forest on the selected feature subspace."""
        if self.feature_indices is not None:
            X_sub = X[:, self.feature_indices]
        else:
            X_sub = X

        try:
            from sklearn.ensemble import IsolationForest
            self._model = IsolationForest(
                n_estimators=self.n_estimators,
                max_samples=min(len(X_sub), self.max_samples if isinstance(self.max_samples, int) else int(self.max_samples * len(X_sub))),
                random_state=self.random_state,
                n_jobs=-1
            )
            self._model.fit(X_sub)
            raw_scores = -self._model.score_samples(X_sub)
            self._score_min = float(raw_scores.min())
            self._score_max = float(raw_scores.max())
        except ImportError:
            # Fallback lightweight isolation estimator if sklearn is unavailable
            self._fit_fallback(X_sub)

        self._is_fitted = True
        return self

    def _fit_fallback(self, X: np.ndarray) -> None:
        """Standalone fallback isolation tree ensemble if sklearn is not installed."""
        np.random.seed(self.random_state or 42)
        n_samples, n_feats = X.shape
        sample_size = min(n_samples, 256)
        
        self._trees = []
        for _ in range(self.n_estimators):
            idx = np.random.choice(n_samples, size=sample_size, replace=False)
            sub_X = X[idx]
            tree = self._build_itree(sub_X, curr_depth=0, max_depth=int(np.ceil(np.log2(max(sample_size, 2)))) )
            self._trees.append(tree)

    def _build_itree(self, X: np.ndarray, curr_depth: int, max_depth: int) -> dict:
        n_samples, n_feats = X.shape
        if curr_depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        feat = np.random.randint(0, n_feats)
        min_v = X[:, feat].min()
        max_v = X[:, feat].max()
        if min_v == max_v:
            return {"type": "leaf", "size": n_samples}

        split_v = np.random.uniform(min_v, max_v)
        left_mask = X[:, feat] < split_v
        right_mask = ~left_mask

        return {
            "type": "split",
            "feature": feat,
            "split_val": split_v,
            "left": self._build_itree(X[left_mask], curr_depth + 1, max_depth),
            "right": self._build_itree(X[right_mask], curr_depth + 1, max_depth)
        }

    def _path_length(self, x: np.ndarray, tree: dict, depth: int) -> float:
        if tree["type"] == "leaf":
            return depth + c_factor(tree["size"])
        if x[tree["feature"]] < tree["split_val"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def compute_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes continuous anomaly score s(x) in [0, 1].
        
        s(x) -> 1 implies high anomaly likelihood.
        s(x) -> 0 implies normal behavior.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before computing anomaly scores.")

        if self.feature_indices is not None:
            X_sub = X[:, self.feature_indices]
        else:
            X_sub = X

        if hasattr(self, "_model") and self._model is not None:
            # sklearn score_samples returns opposite of anomaly score: lower is more anomalous
            # scikit-learn standard: anomaly_score = -score_samples
            raw_scores = -self._model.score_samples(X_sub)
            if self._score_min is not None and self._score_max is not None and self._score_max > self._score_min:
                scores = (raw_scores - self._score_min) / (self._score_max - self._score_min)
                return np.clip(scores, 0.0, 1.0)
            elif len(X_sub) > 1:
                min_s, max_s = raw_scores.min(), raw_scores.max()
                if max_s > min_s:
                    return (raw_scores - min_s) / (max_s - min_s)
            return np.clip(raw_scores, 0.0, 1.0)
        else:
            # Fallback path length scoring: s(x) = 2^(-E(h(x))/c(n))
            n_samples = len(X_sub)
            scores = np.zeros(n_samples)
            c_val = c_factor(256)
            for i in range(n_samples):
                avg_path = np.mean([self._path_length(X_sub[i], t, 0) for t in self._trees])
                scores[i] = 2.0 ** (-avg_path / c_val)
            return scores

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary anomaly labels (1 for anomaly, 0 for normal) based on GA threshold."""
        scores = self.compute_anomaly_scores(X)
        return (scores >= self.threshold).astype(int)
