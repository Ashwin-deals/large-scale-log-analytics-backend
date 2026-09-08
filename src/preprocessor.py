"""Cloud Security Log Preprocessor & Feature Extraction.

Parses raw log events, generates rolling temporal statistics,
and produces standardized feature vectors for anomaly detection.
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class LogFeatureExtractor:
    """Extracts behavioral and statistical security features from cloud logs."""

    SENSITIVE_ACTIONS = {
        "AttachUserPolicy", "CreateAccessKey", "PutRolePolicy",
        "CreateUser", "DownloadArchive", "ExportSnapshot", "AssumeRoleWithPassword"
    }

    FEATURE_NAMES = [
        "log_bytes",
        "request_duration_ms",
        "is_error",
        "is_external_ip",
        "is_auth_service",
        "sensitive_action_flag",
        "user_event_count_window",
        "ip_event_count_window",
        "user_error_rate_window",
        "window_byte_velocity",
        "hour_sin",
        "hour_cos"
    ]

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self._feature_means: Optional[np.ndarray] = None
        self._feature_stds: Optional[np.ndarray] = None

    def _is_private_ip(self, ip: str) -> bool:
        """Determines if an IP address belongs to RFC 1918 private ranges."""
        return ip.startswith("10.") or ip.startswith("172.16.") or ip.startswith("192.168.")

    def extract_features(self, df: pd.DataFrame, fit_scaler: bool = False) -> Tuple[np.ndarray, List[str]]:
        """Extracts engineered numeric features from a DataFrame of cloud logs.
        
        Args:
            df: DataFrame of raw cloud security logs.
            fit_scaler: If True, fits the standardizer on this batch.
            
        Returns:
            Tuple of (Feature Matrix (N, D), List of feature names).
        """
        if df.empty:
            return np.empty((0, len(self.FEATURE_NAMES))), self.FEATURE_NAMES

        df = df.sort_values("timestamp").copy()
        
        # 1. Base log level features
        log_bytes = np.log1p(df["bytes_transferred"].values.astype(float))
        duration = df["request_duration_ms"].values.astype(float)
        is_error = ((df["status_code"] >= 400) | (df["error_code"] != "None")).astype(float).values
        is_external = (~df["source_ip"].apply(self._is_private_ip)).astype(float).values
        is_auth = (df["event_source"] == "iam.amazonaws.com").astype(float).values
        is_sensitive = df["event_name"].isin(self.SENSITIVE_ACTIONS).astype(float).values

        # 2. Cyclic time features
        timestamps = pd.to_datetime(df["timestamp"])
        hours = timestamps.dt.hour + (timestamps.dt.minute / 60.0)
        hour_sin = np.sin(2 * np.pi * hours / 24.0).values
        hour_cos = np.cos(2 * np.pi * hours / 24.0).values

        # 3. Sliding window behavioral aggregations
        n = len(df)
        user_event_count = np.zeros(n, dtype=float)
        ip_event_count = np.zeros(n, dtype=float)
        user_error_rate = np.zeros(n, dtype=float)
        byte_velocity = np.zeros(n, dtype=float)

        user_window_history: Dict[str, List[Dict[str, Any]]] = {}
        ip_window_history: Dict[str, List[Dict[str, Any]]] = {}

        for i, row in enumerate(df.itertuples()):
            u = row.user_name
            ip = row.source_ip
            err = 1.0 if (row.status_code >= 400 or row.error_code != "None") else 0.0
            bytes_val = float(row.bytes_transferred)

            # Update user history
            if u not in user_window_history:
                user_window_history[u] = []
            user_window_history[u].append({"error": err, "bytes": bytes_val})
            if len(user_window_history[u]) > self.window_size:
                user_window_history[u].pop(0)

            # Update IP history
            if ip not in ip_window_history:
                ip_window_history[ip] = []
            ip_window_history[ip].append({"error": err, "bytes": bytes_val})
            if len(ip_window_history[ip]) > self.window_size:
                ip_window_history[ip].pop(0)

            u_hist = user_window_history[u]
            ip_hist = ip_window_history[ip]

            user_event_count[i] = float(len(u_hist))
            ip_event_count[i] = float(len(ip_hist))
            user_error_rate[i] = sum(item["error"] for item in u_hist) / len(u_hist)
            byte_velocity[i] = np.log1p(sum(item["bytes"] for item in ip_hist))

        # Assemble feature matrix
        raw_matrix = np.column_stack([
            log_bytes,
            duration,
            is_error,
            is_external,
            is_auth,
            is_sensitive,
            user_event_count,
            ip_event_count,
            user_error_rate,
            byte_velocity,
            hour_sin,
            hour_cos
        ])

        if fit_scaler or self._feature_means is None:
            self._feature_means = np.mean(raw_matrix, axis=0)
            self._feature_stds = np.std(raw_matrix, axis=0)
            # Avoid division by zero
            self._feature_stds[self._feature_stds < 1e-6] = 1.0

        normalized_matrix = (raw_matrix - self._feature_means) / self._feature_stds
        return normalized_matrix, self.FEATURE_NAMES
