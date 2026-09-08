"""Comprehensive Test Suite for Cloud Security Log Analytics."""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import CloudLogGenerator
from src.preprocessor import LogFeatureExtractor
from src.genetic_optimizer import GeneticOptimizer, Chromosome
from src.model import EvoIsolationForest, c_factor
from src.drift_handler import ConceptDriftHandler
from src.detector import StreamingAnomalyDetector


def test_c_factor():
    """Verify average BST path length calculation."""
    assert c_factor(1) == 1.0
    assert c_factor(2) == 1.0
    val = c_factor(256)
    assert 9.0 < val < 11.0


def test_cloud_log_generator():
    """Verify log generator output and schema."""
    gen = CloudLogGenerator(random_seed=123)
    df = gen.generate_log_batch(n_records=100, anomaly_ratio=0.15)

    assert len(df) == 100
    assert df["is_anomaly"].sum() == 15
    required_cols = {"timestamp", "user_name", "source_ip", "event_source", "event_name", "bytes_transferred", "is_anomaly", "attack_type"}
    assert required_cols.issubset(set(df.columns))


def test_log_feature_extractor():
    """Verify feature extractor output matrix and feature names."""
    gen = CloudLogGenerator(random_seed=42)
    df = gen.generate_log_batch(n_records=60, anomaly_ratio=0.10)

    extractor = LogFeatureExtractor(window_size=20)
    X, names = extractor.extract_features(df, fit_scaler=True)

    assert X.shape == (60, len(names))
    assert len(names) == 12
    assert not np.isnan(X).any()


def test_chromosome_and_optimizer():
    """Verify GA chromosome mutations, crossover, and optimization cycle."""
    n_features = 8
    chrom = Chromosome(n_features=n_features)
    assert len(chrom.features) == n_features
    assert 0.40 <= chrom.threshold <= 0.90

    # Test mutation
    opt = GeneticOptimizer(n_features=n_features, population_size=10, n_generations=2, random_seed=42)
    c1, c2 = opt.crossover(opt.population[0], opt.population[1])
    assert len(c1.features) == n_features
    assert len(c2.features) == n_features

    opt.mutate(c1)
    assert np.any(c1.features)  # At least one feature preserved

    # Test evolution step with dummy evaluation
    def mock_eval(indices, th):
        # Prefer fewer features and threshold close to 0.65
        acc = 0.90
        fpr = 0.05
        tpr = 0.85
        return acc, fpr, tpr

    best = opt.evolve(mock_eval, verbose=False)
    assert best.fitness > -np.inf
    assert len(best.selected_indices) >= 1


def test_evo_isolation_forest():
    """Verify Isolation Forest training, anomaly scoring, and thresholding."""
    np.random.seed(42)
    X_normal = np.random.randn(80, 6)
    X_anomaly = np.random.uniform(5.0, 10.0, size=(10, 6))
    X = np.vstack([X_normal, X_anomaly])

    model = EvoIsolationForest(n_estimators=30, threshold=0.60, random_state=42)
    model.fit(X)

    scores = model.compute_anomaly_scores(X)
    assert len(scores) == 90
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    preds = model.predict(X)
    assert len(preds) == 90
    # Anomalies should have higher scores than normal samples on average
    assert scores[-10:].mean() > scores[:80].mean()


def test_concept_drift_handler():
    """Verify drift detection when distribution shifts."""
    np.random.seed(42)
    ref_data = np.random.normal(loc=0.0, scale=1.0, size=(200, 4))
    
    handler = ConceptDriftHandler(reference_window_size=150, current_window_size=50, ks_alpha=0.05, drift_feature_threshold=0.5)
    handler.set_reference(ref_data)

    # Stream normal samples - no drift
    drift_seen = False
    for _ in range(55):
        sample = np.random.normal(loc=0.0, scale=1.0, size=4)
        if handler.add_sample(sample):
            drift_seen = True
    assert not drift_seen

    # Stream shifted samples - drift should trigger
    drift_detected = False
    for _ in range(55):
        drifted_sample = np.random.normal(loc=5.0, scale=3.0, size=4)
        if handler.add_sample(drifted_sample):
            drift_detected = True
            break
    assert drift_detected


def test_streaming_anomaly_detector_integration():
    """Verify full end-to-end streaming detection and alert dispatching."""
    gen = CloudLogGenerator(random_seed=42)
    train_df = gen.generate_log_batch(n_records=100, anomaly_ratio=0.05)

    extractor = LogFeatureExtractor(window_size=20)
    X_train, _ = extractor.extract_features(train_df, fit_scaler=True)

    model = EvoIsolationForest(n_estimators=30, threshold=0.55, random_state=42)
    model.fit(X_train)

    drift_handler = ConceptDriftHandler(reference_window_size=100, current_window_size=30)
    drift_handler.set_reference(X_train)

    detector = StreamingAnomalyDetector(model=model, preprocessor=extractor, drift_handler=drift_handler)

    test_df = gen.generate_log_batch(n_records=30, anomaly_ratio=0.20)
    results = detector.process_batch(test_df)

    assert len(results) == 30
    summary = detector.get_summary()
    assert summary["total_processed"] == 30
    assert summary["total_anomalies_detected"] >= 1
    assert len(detector.alerts) == summary["total_anomalies_detected"]
