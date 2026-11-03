"""Tests for cross-domain calibration breakdown, vector temperature scaling, and oracle recalibration."""

import numpy as np
import pytest

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.evaluation.calibration import (
    TemperatureScaling,
    compute_reliability_diagram_data,
    evaluate_calibration_breakdown,
)


def test_temperature_scaling_vector_multilabel():
    np.random.seed(42)
    n_samples = 150
    n_classes = len(SHARED_CLASSES)

    # Class 0 overconfident, Class 1 underconfident, others neutral
    logits = np.random.randn(n_samples, n_classes) * 2.0
    logits[:, 0] *= 3.0  # severely overconfident
    labels = (np.random.rand(n_samples, n_classes) > 0.7).astype(np.float32)

    scaler = TemperatureScaling()
    scaler.fit(logits, labels)

    assert scaler.temperatures is not None
    assert len(scaler.temperatures) == n_classes
    assert all(t > 0.01 for t in scaler.temperatures)
    assert scaler.temperature > 0.0

    cal_probs = scaler.calibrate_probs(logits)
    assert cal_probs.shape == (n_samples, n_classes)
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)


def test_compute_reliability_diagram_data_uniform_and_adaptive():
    np.random.seed(42)
    y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0] * 10)
    y_pred = np.linspace(0.05, 0.95, len(y_true))

    # Uniform strategy
    diag_uniform = compute_reliability_diagram_data(y_true, y_pred, n_bins=5, strategy="uniform")
    assert "ece" in diag_uniform
    assert "bin_accs" in diag_uniform
    assert "bin_confs" in diag_uniform
    assert "bin_counts" in diag_uniform
    assert "bin_edges" in diag_uniform
    assert len(diag_uniform["bin_edges"]) == 6
    assert sum(diag_uniform["bin_counts"]) == len(y_true)

    # Adaptive strategy
    diag_adaptive = compute_reliability_diagram_data(y_true, y_pred, n_bins=5, strategy="adaptive")
    assert "ece" in diag_adaptive
    assert diag_adaptive["strategy"] == "adaptive"
    assert sum(diag_adaptive["bin_counts"]) == len(y_true)


def test_evaluate_calibration_breakdown_all_regimes():
    np.random.seed(42)
    classes = ["Pneumonia", "Atelectasis", "Effusion"]
    n_val = 120
    n_test = 200
    n_cal = 60

    # Source validation set
    s_val_logits = np.random.randn(n_val, len(classes))
    s_val_labels = (np.random.rand(n_val, len(classes)) > 0.8).astype(np.float32)

    # Target test set: severely miscalibrated shifted distribution
    t_test_logits = np.random.randn(n_test, len(classes)) * 2.5 + 1.0
    t_test_labels = (np.random.rand(n_test, len(classes)) > 0.85).astype(np.float32)

    # Target cal holdout set (10% calibration)
    t_cal_logits = np.random.randn(n_cal, len(classes)) * 2.5 + 1.0
    t_cal_labels = (np.random.rand(n_cal, len(classes)) > 0.85).astype(np.float32)

    breakdown = evaluate_calibration_breakdown(
        source_val_logits=s_val_logits,
        source_val_labels=s_val_labels,
        target_test_logits=t_test_logits,
        target_test_labels=t_test_labels,
        target_cal_logits=t_cal_logits,
        target_cal_labels=t_cal_labels,
        findings=classes,
        n_bins=10,
        shift_direction="nih_to_chexpert",
    )

    assert breakdown["shift_direction"] == "nih_to_chexpert"
    assert "per_class" in breakdown
    assert "headline_pneumonia" in breakdown
    assert "macro_metrics" in breakdown
    assert "markdown_table" in breakdown

    # Verify per-class contents
    for f in classes:
        cls_data = breakdown["per_class"][f]
        assert "in_domain" in cls_data
        assert "target_uncalibrated" in cls_data
        assert "target_source_scaled" in cls_data
        assert "target_oracle_recalibrated" in cls_data
        assert "ece_shift_delta" in cls_data
        assert "temperature_source" in cls_data
        assert "temperature_oracle" in cls_data
        assert "reliability_diagram" in cls_data
        assert "uncalibrated" in cls_data["reliability_diagram"]
        assert "source_scaled" in cls_data["reliability_diagram"]
        assert "oracle_recalibrated" in cls_data["reliability_diagram"]

    # Verify headline Pneumonia
    pneu = breakdown["headline_pneumonia"]
    assert "ece_shift_delta" in pneu

    # Verify macro metrics
    macro = breakdown["macro_metrics"]
    assert "in_domain_ece" in macro
    assert "target_uncalibrated_ece" in macro
    assert "target_source_scaled_ece" in macro
    assert "target_oracle_recalibrated_ece" in macro
    assert macro["source_mean_temperature"] is not None
    assert macro["oracle_mean_temperature"] is not None

    # Verify markdown table
    assert "### Calibration Degradation & Oracle Recalibration" in breakdown["markdown_table"]
    assert "| **Pneumonia** |" in breakdown["markdown_table"]
    assert "| **Macro Average** |" in breakdown["markdown_table"]


def test_evaluate_calibration_breakdown_without_oracle():
    np.random.seed(42)
    classes = ["Pneumonia", "Effusion"]
    n_val = 50
    n_test = 80

    s_val_logits = np.random.randn(n_val, len(classes))
    s_val_labels = (np.random.rand(n_val, len(classes)) > 0.8).astype(np.float32)

    t_test_logits = np.random.randn(n_test, len(classes))
    t_test_labels = (np.random.rand(n_test, len(classes)) > 0.8).astype(np.float32)

    breakdown = evaluate_calibration_breakdown(
        source_val_logits=s_val_logits,
        source_val_labels=s_val_labels,
        target_test_logits=t_test_logits,
        target_test_labels=t_test_labels,
        target_cal_logits=None,
        target_cal_labels=None,
        findings=classes,
        n_bins=5,
        shift_direction="chexpert_to_nih",
    )

    macro = breakdown["macro_metrics"]
    assert macro["target_oracle_recalibrated_ece"] is None
    assert macro["oracle_mean_temperature"] is None
    assert "—" in breakdown["markdown_table"]

