"""Tests for discrimination metrics against sklearn benchmarks."""

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from pneumoshift.evaluation.metrics import compute_discrimination_metrics


def test_discrimination_metrics_correctness():
    np.random.seed(42)
    y_true = np.array([0, 0, 0, 1, 1, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.2, 0.35, 0.8, 0.65, 0.9, 0.25, 0.7, 0.4, 0.85])

    metrics = compute_discrimination_metrics(y_true, y_pred, fixed_spec_target=0.80, fixed_sens_target=0.80)

    expected_auc = roc_auc_score(y_true, y_pred)
    expected_pr = average_precision_score(y_true, y_pred)

    assert np.isclose(metrics["auroc"], expected_auc, atol=1e-5)
    assert np.isclose(metrics["auprc"], expected_pr, atol=1e-5)
    assert "sens_at_80_spec" in metrics
    assert "spec_at_80_sens" in metrics
    assert 0.0 <= metrics["sens_at_80_spec"] <= 1.0
    assert 0.0 <= metrics["spec_at_80_sens"] <= 1.0

