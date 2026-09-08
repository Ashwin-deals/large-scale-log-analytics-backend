"""Cloud Security Log Generator.

Simulates cloud audit/access logs (similar to AWS CloudTrail and VPC flows)
containing both legitimate enterprise traffic and synthetic attack patterns.
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


class CloudLogGenerator:
    """Generates synthetic cloud access and security audit logs."""

    SERVICES = ["iam.amazonaws.com", "s3.amazonaws.com", "ec2.amazonaws.com", "lambda.amazonaws.com"]
    NORMAL_ACTIONS = {
        "iam.amazonaws.com": ["GetUser", "ListRoles", "GetAccountSummary"],
        "s3.amazonaws.com": ["GetObject", "ListBucket", "HeadObject", "PutObject"],
        "ec2.amazonaws.com": ["DescribeInstances", "DescribeSecurityGroups", "DescribeVpcs"],
        "lambda.amazonaws.com": ["InvokeFunction", "GetFunctionConfiguration"]
    }
    PRIVILEGE_ACTIONS = ["AttachUserPolicy", "CreateAccessKey", "PutRolePolicy", "CreateUser"]
    EXFILTRATION_ACTIONS = ["GetObject", "DownloadArchive", "ExportSnapshot"]

    NORMAL_USERS = ["alice_dev", "bob_ops", "carol_data", "dave_sec", "service_app_worker"]
    COMPROMISED_USERS = ["temp_guest", "extern_contractor", "service_legacy"]

    INTERNAL_IPS = [f"10.0.{i}.{j}" for i in range(1, 4) for j in range(10, 30)]
    EXTERNAL_IPS = [f"198.51.100.{i}" for i in range(1, 20)]
    ATTACKER_IPS = ["203.0.113.45", "198.51.100.99", "185.220.101.5"]

    def __init__(self, random_seed: Optional[int] = 42):
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _generate_normal_event(self, timestamp: datetime) -> Dict[str, Any]:
        """Generates a benign cloud activity log entry."""
        service = random.choice(self.SERVICES)
        action = random.choice(self.NORMAL_ACTIONS[service])
        user = random.choice(self.NORMAL_USERS)
        source_ip = random.choice(self.INTERNAL_IPS)
        
        # Benign parameters
        duration = float(np.random.exponential(scale=45.0) + 5.0)
        bytes_tx = int(np.random.lognormal(mean=7.0, sigma=1.2)) if "s3" in service else random.randint(100, 800)
        error_code = "None" if random.random() > 0.03 else "ClientRequestTimeout"
        status_code = 200 if error_code == "None" else 408

        return {
            "timestamp": timestamp.isoformat(),
            "user_name": user,
            "source_ip": source_ip,
            "event_source": service,
            "event_name": action,
            "bytes_transferred": bytes_tx,
            "request_duration_ms": round(duration, 2),
            "error_code": error_code,
            "status_code": status_code,
            "is_anomaly": 0,
            "attack_type": "BENIGN"
        }

    def _generate_attack_event(self, timestamp: datetime, attack_type: Optional[str] = None) -> Dict[str, Any]:
        """Generates an anomalous cloud security event."""
        if attack_type is None:
            attack_type = random.choice(["BRUTE_FORCE", "PRIVILEGE_ESCALATION", "DATA_EXFILTRATION", "API_FLOOD"])

        if attack_type == "BRUTE_FORCE":
            return {
                "timestamp": timestamp.isoformat(),
                "user_name": random.choice(self.COMPROMISED_USERS),
                "source_ip": random.choice(self.ATTACKER_IPS),
                "event_source": "iam.amazonaws.com",
                "event_name": "AssumeRoleWithPassword",
                "bytes_transferred": random.randint(50, 200),
                "request_duration_ms": round(float(np.random.exponential(scale=10.0) + 2.0), 2),
                "error_code": "AccessDenied",
                "status_code": 403,
                "is_anomaly": 1,
                "attack_type": "BRUTE_FORCE"
            }
        elif attack_type == "PRIVILEGE_ESCALATION":
            return {
                "timestamp": timestamp.isoformat(),
                "user_name": random.choice(self.COMPROMISED_USERS),
                "source_ip": random.choice(self.EXTERNAL_IPS),
                "event_source": "iam.amazonaws.com",
                "event_name": random.choice(self.PRIVILEGE_ACTIONS),
                "bytes_transferred": random.randint(300, 1500),
                "request_duration_ms": round(float(np.random.exponential(scale=120.0) + 40.0), 2),
                "error_code": "None" if random.random() > 0.4 else "UnauthorizedOperation",
                "status_code": 200 if random.random() > 0.4 else 403,
                "is_anomaly": 1,
                "attack_type": "PRIVILEGE_ESCALATION"
            }
        elif attack_type == "DATA_EXFILTRATION":
            # Very high bytes transferred in a short window
            return {
                "timestamp": timestamp.isoformat(),
                "user_name": random.choice(self.COMPROMISED_USERS),
                "source_ip": random.choice(self.ATTACKER_IPS),
                "event_source": "s3.amazonaws.com",
                "event_name": random.choice(self.EXFILTRATION_ACTIONS),
                "bytes_transferred": int(np.random.uniform(50_000_000, 250_000_000)),
                "request_duration_ms": round(float(np.random.uniform(4000.0, 15000.0)), 2),
                "error_code": "None",
                "status_code": 200,
                "is_anomaly": 1,
                "attack_type": "DATA_EXFILTRATION"
            }
        else: # API_FLOOD
            return {
                "timestamp": timestamp.isoformat(),
                "user_name": random.choice(self.COMPROMISED_USERS),
                "source_ip": random.choice(self.ATTACKER_IPS),
                "event_source": random.choice(self.SERVICES),
                "event_name": "DescribeInstances",
                "bytes_transferred": random.randint(50, 150),
                "request_duration_ms": round(float(np.random.uniform(1.0, 15.0)), 2),
                "error_code": "RequestLimitExceeded",
                "status_code": 429,
                "is_anomaly": 1,
                "attack_type": "API_FLOOD"
            }

    def generate_log_batch(
        self,
        n_records: int = 1000,
        anomaly_ratio: float = 0.08,
        start_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Generates a batch of cloud security logs with specified anomaly ratio."""
        if start_time is None:
            from datetime import timezone
            current_time = datetime.now(timezone.utc)
        else:
            current_time = start_time

        logs: List[Dict[str, Any]] = []
        n_anomalies = int(n_records * anomaly_ratio)
        n_normal = n_records - n_anomalies

        # Generate normal records
        for _ in range(n_normal):
            current_time += timedelta(milliseconds=random.randint(50, 500))
            logs.append(self._generate_normal_event(current_time))

        # Inject attack patterns
        for _ in range(n_anomalies):
            current_time += timedelta(milliseconds=random.randint(10, 100))
            logs.append(self._generate_attack_event(current_time))

        df = pd.DataFrame(logs)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
