"""Tests for calibration calculations (ECE, MCE, Temperature Scaling, Platt Scaling)."""

import numpy as np
import pytest
from pneumoshift.evaluation.calibration import PlattScaling, TemperatureScaling, compute_ece


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


def test_platt_scaling_optimization():
    logits = np.array([-2.0, -1.0, 1.5, 2.5, -3.0, 3.0])
    labels = np.array([0, 0, 1, 1, 0, 1])

    scaler = PlattScaling()
    scaler.fit(logits, labels)

    assert scaler.a > 0.0
    cal_probs = scaler.calibrate_probs(logits)
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
    # Check monotonicity preservation
    assert cal_probs[0] < cal_probs[2]
    assert cal_probs[4] < cal_probs[5]


def test_extreme_logits_numerical_stability():
    # Extreme logits (+100.0, -100.0) should not cause overflow, NaN, or Inf
    logits = np.array([-100.0, -50.0, 50.0, 100.0])
    labels = np.array([0, 0, 1, 1])

    t_scaler = TemperatureScaling()
    t_scaler.fit(logits, labels)
    t_probs = t_scaler.calibrate_probs(logits)
    assert not np.any(np.isnan(t_probs))
    assert not np.any(np.isinf(t_probs))

    p_scaler = PlattScaling()
    p_scaler.fit(logits, labels)
    p_probs = p_scaler.calibrate_probs(logits)
    assert not np.any(np.isnan(p_probs))
    assert not np.any(np.isinf(p_probs))


def test_skewed_predictions_adaptive_vs_uniform_ece():
    """Verifies that equal-mass (adaptive) and equal-width (uniform) ECE produce distinct values on skewed prediction distributions."""
    np.random.seed(42)
    n_samples = 10000
    # Skewed prediction distribution typical of low-prevalence chest radiograph findings
    y_pred = np.random.beta(0.5, 10.0, size=n_samples)
    y_true = (np.random.rand(n_samples) < y_pred).astype(float)

    ece_uniform, _, details_uniform = compute_ece(y_true, y_pred, n_bins=15, strategy="uniform")
    ece_adaptive, _, details_adaptive = compute_ece(y_true, y_pred, n_bins=15, strategy="adaptive")

    # Both must be valid non-negative numbers
    assert ece_uniform > 0.0
    assert ece_adaptive > 0.0
    # On skewed data, equal-mass and equal-width partitioning must produce distinct ECE values
    assert abs(ece_uniform - ece_adaptive) > 1e-4
    # Equal-width binning has a heavily skewed distribution (>50% samples in first bin)
    assert details_uniform["bin_counts"][0] > n_samples * 0.5
    # Equal-mass binning distributes samples evenly across bins (~n_samples / n_bins)
    expected_per_bin = n_samples / 15.0
    assert np.all(np.abs(details_adaptive["bin_counts"] - expected_per_bin) < 5)



