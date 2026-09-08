"""Real-Time Streaming Anomaly Detector & Security Alert Engine.

Processes streaming cloud security logs, scores events against the
GA-optimized Isolation Forest model, and triggers alert notifications.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from .model import EvoIsolationForest
from .preprocessor import LogFeatureExtractor
from .drift_handler import ConceptDriftHandler


class StreamingAnomalyDetector:
    """End-to-end streaming security detector for cloud logs."""

    def __init__(
        self,
        model: EvoIsolationForest,
        preprocessor: LogFeatureExtractor,
        drift_handler: Optional[ConceptDriftHandler] = None
    ):
        self.model = model
        self.preprocessor = preprocessor
        self.drift_handler = drift_handler
        self.recent_logs: List[Dict[str, Any]] = []
        self.alerts: List[Dict[str, Any]] = []
        self.total_processed: int = 0
        self.total_anomalies: int = 0

    def process_event(self, log_event: Dict[str, Any]) -> Dict[str, Any]:
        """Processes a single real-time cloud log event and generates detection outputs."""
        self.total_processed += 1
        self.recent_logs.append(log_event)

        # Use recent window for contextual sliding-window feature extraction
        window_df = pd.DataFrame(self.recent_logs[-self.preprocessor.window_size:])
        features, _ = self.preprocessor.extract_features(window_df, fit_scaler=False)
        current_feature_vec = features[-1]

        # Check for concept drift
        drift_detected = False
        if self.drift_handler is not None:
            drift_detected = self.drift_handler.add_sample(current_feature_vec)

        # Anomaly scoring
        score = float(self.model.compute_anomaly_scores(current_feature_vec.reshape(1, -1))[0])
        is_anomaly = bool(score >= self.model.threshold)

        alert_info = None
        if is_anomaly:
            self.total_anomalies += 1
            severity = "CRITICAL" if score >= 0.85 else ("HIGH" if score >= 0.75 else "MEDIUM")
            alert_info = {
                "alert_id": f"ALT-{self.total_processed:06d}",
                "timestamp": log_event.get("timestamp"),
                "severity": severity,
                "anomaly_score": round(score, 4),
                "threshold": round(self.model.threshold, 4),
                "source_ip": log_event.get("source_ip"),
                "user_name": log_event.get("user_name"),
                "event_source": log_event.get("event_source"),
                "event_name": log_event.get("event_name"),
                "attack_type": log_event.get("attack_type", "UNKNOWN"),
                "drift_flag": drift_detected
            }
            self.alerts.append(alert_info)

        return {
            "processed_id": self.total_processed,
            "anomaly_score": round(score, 4),
            "is_anomaly": is_anomaly,
            "drift_detected": drift_detected,
            "alert": alert_info
        }

    def process_batch(self, df_batch: pd.DataFrame) -> List[Dict[str, Any]]:
        """Processes a batch of streaming events sequentially."""
        results = []
        for row in df_batch.to_dict(orient="records"):
            results.append(self.process_event(row))
        return results

    def get_summary(self) -> Dict[str, Any]:
        """Returns detection statistics."""
        return {
            "total_processed": self.total_processed,
            "total_anomalies_detected": self.total_anomalies,
            "anomaly_rate": round(self.total_anomalies / max(1, self.total_processed), 4),
            "total_alerts": len(self.alerts),
            "model_threshold": self.model.threshold,
            "features_used": len(self.model.feature_indices) if self.model.feature_indices is not None else "all"
        }
