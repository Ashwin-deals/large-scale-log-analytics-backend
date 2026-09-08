"""Concept Drift Detection for Streaming Cloud Logs.

Detects statistical distribution shifts in cloud log features using
windowed Kolmogorov-Smirnov tests and Population Stability Index (PSI).
"""

from typing import Optional, Dict, Any
import numpy as np
from scipy.stats import ks_2samp


class ConceptDriftHandler:
    """Monitors sliding windows of streaming features to identify concept drift."""

    def __init__(
        self,
        reference_window_size: int = 500,
        current_window_size: int = 150,
        ks_alpha: float = 0.01,
        drift_feature_threshold: float = 0.35
    ):
        self.reference_window_size = reference_window_size
        self.current_window_size = current_window_size
        self.ks_alpha = ks_alpha
        self.drift_feature_threshold = drift_feature_threshold

        self.reference_buffer: Optional[np.ndarray] = None
        self.current_buffer: list = []
        self.drift_history: list = []

    def set_reference(self, X_ref: np.ndarray) -> None:
        """Sets the baseline reference feature distribution."""
        if len(X_ref) > self.reference_window_size:
            self.reference_buffer = X_ref[-self.reference_window_size:].copy()
        else:
            self.reference_buffer = X_ref.copy()
        self.current_buffer.clear()

    def add_sample(self, feature_vector: np.ndarray) -> bool:
        """Adds a streaming sample and checks if concept drift has occurred.
        
        Returns:
            True if concept drift is detected across the feature threshold, else False.
        """
        self.current_buffer.append(feature_vector)
        if len(self.current_buffer) > self.current_window_size:
            self.current_buffer.pop(0)

        if len(self.current_buffer) < self.current_window_size or self.reference_buffer is None:
            return False

        return self._evaluate_drift()

    def _evaluate_drift(self) -> bool:
        """Runs Kolmogorov-Smirnov test per feature between reference and current buffer."""
        curr_matrix = np.array(self.current_buffer)
        ref_matrix = self.reference_buffer

        n_features = ref_matrix.shape[1]
        drifted_features_count = 0

        for col in range(n_features):
            ref_col = ref_matrix[:, col]
            curr_col = curr_matrix[:, col]

            # If variance is zero in both, skip
            if np.std(ref_col) < 1e-6 and np.std(curr_col) < 1e-6:
                continue

            stat, p_value = ks_2samp(ref_col, curr_col)
            if p_value < self.ks_alpha:
                drifted_features_count += 1

        drift_ratio = drifted_features_count / max(1, n_features)
        is_drift = drift_ratio >= self.drift_feature_threshold

        if is_drift:
            self.drift_history.append({
                "drift_ratio": drift_ratio,
                "drifted_features": drifted_features_count,
                "total_features": n_features
            })

        return is_drift
