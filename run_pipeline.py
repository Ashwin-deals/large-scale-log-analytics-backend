"""End-to-End Execution Pipeline for Cloud Security Log Analytics.

Demonstrates:
1. Synthetic Cloud Security Log Generation
2. Preprocessing & Temporal Behavioral Feature Extraction
3. Genetic Algorithm (Evolutionary Optimization) of Features & Threshold
4. Isolation Forest Model Training
5. Real-Time Streaming Detection & Concept Drift Handling
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from src.data_generator import CloudLogGenerator
from src.preprocessor import LogFeatureExtractor
from src.genetic_optimizer import GeneticOptimizer
from src.model import EvoIsolationForest
from src.drift_handler import ConceptDriftHandler
from src.detector import StreamingAnomalyDetector


def main():
    print("=" * 70)
    print(" Large-Scale Log Analytics for Cloud Security (Evolutionary Framework)")
    print("=" * 70)

    # 1. Generate Historical Training Logs
    print("\n[Step 1/5] Generating historical cloud security logs...")
    generator = CloudLogGenerator(random_seed=42)
    train_df = generator.generate_log_batch(n_records=1200, anomaly_ratio=0.08)
    print(f" -> Generated {len(train_df)} logs ({train_df['is_anomaly'].sum()} anomalies, "
          f"{(train_df['is_anomaly'] == 0).sum()} normal)")

    # 2. Feature Extraction
    print("\n[Step 2/5] Extracting behavioral & sliding-window features...")
    preprocessor = LogFeatureExtractor(window_size=50)
    X_train, feature_names = preprocessor.extract_features(train_df, fit_scaler=True)
    y_train = train_df["is_anomaly"].values
    print(f" -> Extracted {X_train.shape[1]} features across {X_train.shape[0]} samples.")
    for idx, name in enumerate(feature_names):
        print(f"    [{idx:02d}] {name}")

    # 3. Genetic Algorithm Optimization
    print("\n[Step 3/5] Running Genetic Algorithm (Feature Selection & Threshold Calibration)...")
    
    # Split train into train / validation for GA fitness evaluation
    val_size = 400
    X_ga_train, y_ga_train = X_train[:-val_size], y_train[:-val_size]
    X_ga_val, y_ga_val = X_train[-val_size:], y_train[-val_size:]

    def ga_evaluator(feature_indices: np.ndarray, threshold: float):
        # Quick surrogate model fit
        clf = EvoIsolationForest(
            n_estimators=40,
            feature_indices=feature_indices,
            threshold=threshold,
            random_state=42
        )
        clf.fit(X_ga_train)
        preds = clf.predict(X_ga_val)
        
        # Calculate accuracy and FPR
        acc = float(accuracy_score(y_ga_val, preds))
        neg_mask = (y_ga_val == 0)
        fpr = float(np.mean(preds[neg_mask] == 1)) if np.any(neg_mask) else 0.0
        pos_mask = (y_ga_val == 1)
        tpr = float(np.mean(preds[pos_mask] == 1)) if np.any(pos_mask) else 0.0
        return acc, fpr, tpr

    optimizer = GeneticOptimizer(
        n_features=X_train.shape[1],
        population_size=20,
        n_generations=10,
        alpha=1.0,
        beta=1.8,
        gamma=0.05,
        random_seed=42
    )
    best_chrom = optimizer.evolve(ga_evaluator, verbose=True)

    print("\n" + "-" * 50)
    print("GA Optimization Results:")
    print(f" -> Optimal Features: {[feature_names[i] for i in best_chrom.selected_indices]}")
    print(f" -> Selected Count: {len(best_chrom.selected_indices)} / {len(feature_names)}")
    print(f" -> Dynamic Threshold (T*): {best_chrom.threshold:.4f}")
    print(f" -> Best Fitness: {best_chrom.fitness:.4f}")
    print("-" * 50)

    # 4. Train Final Model with Evolved Parameters
    print("\n[Step 4/5] Training Production Isolation Forest with Evolved Parameters...")
    model = EvoIsolationForest(
        n_estimators=100,
        feature_indices=best_chrom.selected_indices,
        threshold=best_chrom.threshold,
        random_state=42
    )
    model.fit(X_train)
    print(" -> Model successfully trained on selected subspace.")

    # 5. Real-Time Streaming Detection Simulation
    print("\n[Step 5/5] Initializing Streaming Detection & Concept Drift Handler...")
    drift_handler = ConceptDriftHandler(
        reference_window_size=400,
        current_window_size=100,
        ks_alpha=0.01,
        drift_feature_threshold=0.30
    )
    drift_handler.set_reference(X_train)

    detector = StreamingAnomalyDetector(
        model=model,
        preprocessor=preprocessor,
        drift_handler=drift_handler
    )

    # Generate streaming test events
    print(" -> Generating incoming real-time streaming traffic...")
    test_df = generator.generate_log_batch(n_records=350, anomaly_ratio=0.10)
    
    stream_results = detector.process_batch(test_df)
    
    # Calculate performance on stream
    y_test_true = test_df["is_anomaly"].values
    y_test_pred = np.array([r["is_anomaly"] for r in stream_results], dtype=int)

    acc = accuracy_score(y_test_true, y_test_pred)
    prec = precision_score(y_test_true, y_test_pred, zero_division=0)
    rec = recall_score(y_test_true, y_test_pred, zero_division=0)
    f1 = f1_score(y_test_true, y_test_pred, zero_division=0)

    print("\n" + "=" * 70)
    print(" Streaming Detection Performance Summary")
    print("=" * 70)
    print(f" Total Processed Events: {len(test_df)}")
    print(f" Total Security Alerts Dispatched: {len(detector.alerts)}")
    print(f" Detection Accuracy:   {acc * 100:.2f}%")
    print(f" Precision:            {prec * 100:.2f}%")
    print(f" Recall:               {rec * 100:.2f}%")
    print(f" F1-Score:             {f1 * 100:.2f}%")

    # Sample alerts
    print("\nSample Security Alerts Dispatched:")
    for alert in detector.alerts[:3]:
        print(f" [{alert['severity']}] ID: {alert['alert_id']} | User: {alert['user_name']} "
              f"| IP: {alert['source_ip']} | Type: {alert['attack_type']} "
              f"| Score: {alert['anomaly_score']} (Threshold: {alert['threshold']})")

    print("\nPipeline executed successfully.")


if __name__ == "__main__":
    main()
