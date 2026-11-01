"""Tests for calibration calculations (ECE, MCE, Temperature Scaling)."""

import numpy as np
import pytest
from pneumoshift.evaluation.calibration import TemperatureScaling, compute_ece


def test_perfect_calibration_ece():
    # If confidence equals empirical frequency, ECE must be near 0
    # Bin 1: 100 samples with prob 0.2 and 20% positive
    # Bin 2: 100 samples with prob 0.8 and 80% positive
    y_pred = np.array([0.2] * 100 + [0.8] * 100)
    y_true = np.array([1] * 20 + [0] * 80 + [1] * 80 + [0] * 20)

    ece, mce, _ = compute_ece(y_true, y_pred, n_bins=10, strategy="uniform")
    assert ece < 0.05
    assert mce < 0.05


def test_severe_miscalibration_ece():
    # Model is completely overconfident: predicting 0.99 for all samples, but truth is all 0
    y_pred = np.array([0.99] * 100)
    y_true = np.array([0] * 100)

    ece, mce, _ = compute_ece(y_true, y_pred, n_bins=10, strategy="uniform")
    assert ece > 0.95
    assert mce > 0.95


def test_temperature_scaling_optimization():
    logits = np.array([-2.0, -1.0, 1.5, 2.5, -3.0, 3.0])
    labels = np.array([0, 0, 1, 1, 0, 1])

    scaler = TemperatureScaling()
    scaler.fit(logits, labels)

    assert scaler.temperature > 0.0
    cal_probs = scaler.calibrate_probs(logits)
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)

