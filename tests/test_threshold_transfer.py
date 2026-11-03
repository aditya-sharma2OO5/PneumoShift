"""Tests for decision threshold transfer failure, regret, and continuous score distribution shifts."""

import numpy as np
import pytest

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.thresholds import (
    compute_binary_metrics_at_threshold,
    compute_optimal_thresholds,
    compute_score_distribution_shift,
    evaluate_threshold_transfer,
    evaluate_threshold_transfer_suite,
)


def test_compute_binary_metrics_at_threshold():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([0.9, 0.8, 0.3, 0.1, 0.4, 0.7])

    # At threshold 0.5:
    # y_pred >= 0.5 -> indices 0, 1, 5 (labels: 1, 1, 0) -> 2 TP, 1 FP
    # y_pred < 0.5 -> indices 2, 3, 4 (labels: 0, 0, 1) -> 2 TN, 1 FN
    metrics = compute_binary_metrics_at_threshold(y_true, y_pred, threshold=0.5)

    assert metrics["tp"] == 2
    assert metrics["fp"] == 1
    assert metrics["tn"] == 2
    assert metrics["fn"] == 1
    assert pytest.approx(metrics["sensitivity"], 0.01) == 2 / 3
    assert pytest.approx(metrics["specificity"], 0.01) == 2 / 3
    assert pytest.approx(metrics["precision"], 0.01) == 2 / 3
    assert pytest.approx(metrics["f1_score"], 0.01) == 2 / 3
    assert pytest.approx(metrics["youden_j"], 0.01) == (2 / 3 + 2 / 3 - 1.0)
    assert pytest.approx(metrics["accuracy"], 0.01) == 4 / 6


def test_compute_binary_metrics_empty_and_extremes():
    empty_res = compute_binary_metrics_at_threshold(np.array([]), np.array([]), threshold=0.5)
    assert empty_res["sensitivity"] == 0.0
    assert empty_res["tp"] == 0

    # All predicted negative (threshold > 1.0)
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.2, 0.3, 0.4, 0.1])
    ext_res = compute_binary_metrics_at_threshold(y_true, y_pred, threshold=0.9)
    assert ext_res["tp"] == 0
    assert ext_res["fp"] == 0
    assert ext_res["precision"] == 0.0
    assert ext_res["sensitivity"] == 0.0


def test_compute_optimal_thresholds_criteria():
    np.random.seed(42)
    # Well-separated synthetic scores
    positives = np.random.normal(loc=0.75, scale=0.1, size=100).clip(0.01, 0.99)
    negatives = np.random.normal(loc=0.25, scale=0.1, size=400).clip(0.01, 0.99)

    y_pred = np.concatenate([positives, negatives])
    y_true = np.concatenate([np.ones(100), np.zeros(400)])

    opts = compute_optimal_thresholds(y_true, y_pred, target_sensitivity=0.90)

    assert 0.0 <= opts["youden_threshold"] <= 1.0
    assert 0.0 <= opts["f1_threshold"] <= 1.0
    assert 0.0 <= opts["sens90_threshold"] <= 1.0

    # Sens@90% must guarantee sensitivity >= 0.90
    sens90_metrics = opts["fixed_sensitivity"]["metrics"]
    assert sens90_metrics["sensitivity"] >= 0.89  # Account for discretization

    # Backwards compatibility check
    assert "youden" in opts
    assert "max_f1" in opts
    assert "fixed_sensitivity" in opts


def test_evaluate_threshold_transfer_and_regret():
    np.random.seed(42)
    # Source optimal threshold is ~0.60
    source_thresh = 0.60
    source_metrics = {
        "sensitivity": 0.85,
        "specificity": 0.90,
        "f1_score": 0.80,
        "youden_j": 0.75,
    }

    # In target domain, model is miscalibrated and outputs systematically lower probabilities
    # so source_thresh=0.60 causes sensitivity collapse
    target_pos = np.random.normal(loc=0.45, scale=0.1, size=80).clip(0.01, 0.99)
    target_neg = np.random.normal(loc=0.15, scale=0.08, size=320).clip(0.01, 0.99)

    target_pred = np.concatenate([target_pos, target_neg])
    target_true = np.concatenate([np.ones(80), np.zeros(320)])

    # Target-optimal threshold would be ~0.30
    target_opt_thresh = 0.30

    res = evaluate_threshold_transfer(
        applied_threshold=source_thresh,
        y_true=target_true,
        y_pred=target_pred,
        source_metrics=source_metrics,
        target_optimal_threshold=target_opt_thresh,
    )

    # Sensitivity must drop significantly when transferring source_thresh=0.60 to target
    assert res["delta_sensitivity"] < 0.0  # Sensitivity collapsed
    assert res["relative_sensitivity_drop_percent"] > 0.0
    assert "delta_specificity" in res
    assert "delta_f1" in res
    assert "regret_f1" in res
    assert res["regret_f1"] >= 0.0  # Regret vs target-optimal must be non-negative
    assert "target_optimal_metrics" in res


def test_compute_score_distribution_shift_and_ks_test():
    np.random.seed(42)
    # Source scores: mean 0.50
    source_scores = np.random.beta(a=2, b=2, size=500)
    # Target scores: shifted downwards (mean 0.30)
    target_scores = np.random.beta(a=1.5, b=3.5, size=500)

    source_labels = (source_scores > 0.5).astype(int)
    target_labels = (target_scores > 0.5).astype(int)

    shift = compute_score_distribution_shift(
        source_scores=source_scores,
        target_scores=target_scores,
        source_labels=source_labels,
        target_labels=target_labels,
    )

    assert "source" in shift
    assert "target" in shift
    assert "median_difference" in shift
    assert shift["median_difference"] < 0.0  # Target shifted downwards
    assert "ks_test" in shift
    assert 0.0 <= shift["ks_test"]["statistic"] <= 1.0
    assert 0.0 <= shift["ks_test"]["pvalue"] <= 1.0
    assert shift["ks_test"]["reject_null_005"] is True

    # Stratified analysis
    assert "stratified" in shift
    assert "positives" in shift["stratified"]
    assert "negatives" in shift["stratified"]
    assert "median_difference" in shift["stratified"]["positives"]
    assert "ks_statistic" in shift["stratified"]["positives"]


def test_evaluate_threshold_transfer_suite_full():
    np.random.seed(42)
    classes = ["Pneumonia", "Atelectasis", "Cardiomegaly"]
    n_src = 150
    n_tgt = 250

    # Source validation cohort
    src_labels = (np.random.rand(n_src, len(classes)) > 0.8).astype(np.float32)
    src_probs = np.random.uniform(0.1, 0.9, (n_src, len(classes)))

    # Target test cohort with domain shift
    tgt_labels = (np.random.rand(n_tgt, len(classes)) > 0.85).astype(np.float32)
    tgt_probs = np.random.uniform(0.05, 0.75, (n_tgt, len(classes)))

    suite_results = evaluate_threshold_transfer_suite(
        source_val_labels=src_labels,
        source_val_probs=src_probs,
        target_test_labels=tgt_labels,
        target_test_probs=tgt_probs,
        findings=classes,
        shift_direction="nih_to_chexpert",
    )

    assert suite_results["shift_direction"] == "nih_to_chexpert"
    assert "per_class" in suite_results
    assert "headline_pneumonia" in suite_results
    assert "macro_metrics" in suite_results
    assert "markdown_tables" in suite_results

    # Check per-class contents
    for f in classes:
        cls_data = suite_results["per_class"][f]
        assert "source_optimal_thresholds" in cls_data
        assert "target_optimal_thresholds" in cls_data
        assert "transfers" in cls_data
        assert "youden" in cls_data["transfers"]
        assert "sens90" in cls_data["transfers"]
        assert "max_f1" in cls_data["transfers"]
        assert "score_distribution_shift" in cls_data

    # Check macro drops
    macro = suite_results["macro_metrics"]
    assert "youden" in macro
    assert "sens90" in macro
    assert "mean_delta_sensitivity" in macro["youden"]
    assert "mean_regret_f1" in macro["youden"]

    # Check Markdown tables
    tables = suite_results["markdown_tables"]
    assert "threshold_transfer" in tables
    assert "score_shift" in tables
    assert "Threshold Transfer Failure: Youden's J & Sens@90%" in tables["threshold_transfer"]
    assert "Prediction Score Distribution Shift & KS Tests" in tables["score_shift"]
    assert "**Pneumonia**" in tables["threshold_transfer"]
    assert "**Macro Average**" in tables["threshold_transfer"]

