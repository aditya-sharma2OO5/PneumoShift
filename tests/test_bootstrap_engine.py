"""Unit tests for Phase 6 Clustered Patient Bootstrap Engine."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from pneumoshift.aggregation.bootstrap_engine import (
    PatientClusteredBootstrap,
    compute_cohort_bootstrap_summary,
    save_bootstrap_report,
)
from pneumoshift.aggregation.config_schemas import BootstrapCIResult
from pneumoshift.utils.io import load_json


@pytest.fixture
def clustered_data():
    """Generates synthetic clustered patient prediction data."""
    np.random.seed(42)
    # 20 unique patients, each having 3 images (total 60 samples)
    patient_ids = np.repeat(np.array([f"patient_{i:03d}" for i in range(20)]), 3)
    y_true = np.array([1.0 if int(p.split("_")[1]) % 3 == 0 else 0.0 for p in patient_ids])
    
    # Model A: good predictor
    noise_a = np.random.uniform(-0.1, 0.1, len(y_true))
    y_pred_a = np.clip(y_true * 0.7 + 0.15 + noise_a, 0.01, 0.99)

    # Model B: degraded predictor (more noise, flatter probabilities)
    noise_b = np.random.uniform(-0.35, 0.35, len(y_true))
    y_pred_b = np.clip(y_true * 0.3 + 0.35 + noise_b, 0.01, 0.99)

    return patient_ids, y_true, y_pred_a, y_pred_b


class TestPatientClusteredBootstrap:
    """Verifies patient-clustered bootstrap uncertainty estimation."""

    def test_compute_ci_auroc_bounds(self, clustered_data):
        patient_ids, y_true, y_pred_a, _ = clustered_data
        engine = PatientClusteredBootstrap(n_resamples=100, confidence_level=0.95, seed=42)
        ci_res = engine.compute_ci(patient_ids, y_true, y_pred_a, metric="auroc")

        assert isinstance(ci_res, BootstrapCIResult)
        assert 0.0 <= ci_res.ci_lower <= ci_res.point_estimate <= ci_res.ci_upper <= 1.0
        assert ci_res.standard_error > 0.0
        assert ci_res.confidence_level == 0.95
        assert " [" in ci_res.format_ci(3)

    def test_compute_ci_all_standard_metrics(self, clustered_data):
        patient_ids, y_true, y_pred_a, _ = clustered_data
        engine = PatientClusteredBootstrap(n_resamples=50, confidence_level=0.90, seed=42)

        metrics = ["auprc", "sens_at_90_spec", "spec_at_90_sens", "ece_uniform", "ece_adaptive", "brier_score", "nll", "f1_score"]
        for m in metrics:
            ci = engine.compute_ci(patient_ids, y_true, y_pred_a, metric=m)
            assert ci.ci_lower <= ci.point_estimate <= ci.ci_upper
            assert not np.isnan(ci.point_estimate)

    def test_paired_difference_significance(self, clustered_data):
        patient_ids, y_true, y_pred_a, y_pred_b = clustered_data
        engine = PatientClusteredBootstrap(n_resamples=100, confidence_level=0.95, seed=42)
        diff_res = engine.compute_paired_difference(patient_ids, y_true, y_pred_a, y_pred_b, metric="auroc")

        assert "mean_diff" in diff_res
        assert diff_res["mean_diff"] > 0.0  # Model A is strictly better than Model B
        assert diff_res["ci_lower"] <= diff_res["mean_diff"] <= diff_res["ci_upper"]
        assert 0.0 <= diff_res["p_value"] <= 1.0

    def test_two_sample_difference(self, clustered_data):
        patient_ids, y_true, y_pred_a, y_pred_b = clustered_data
        # Split cohort into two disjoint patient populations
        mask_a = np.array([int(p.split("_")[1]) < 10 for p in patient_ids])
        mask_b = ~mask_a

        engine = PatientClusteredBootstrap(n_resamples=100, confidence_level=0.95, seed=42)
        diff_res = engine.compute_two_sample_difference(
            patient_ids[mask_a], y_true[mask_a], y_pred_a[mask_a],
            patient_ids[mask_b], y_true[mask_b], y_pred_b[mask_b],
            metric="auroc",
        )
        assert "mean_diff" in diff_res
        assert diff_res["ci_lower"] <= diff_res["mean_diff"] <= diff_res["ci_upper"]
        assert 0.0 <= diff_res["p_value"] <= 1.0

    def test_seed_determinism(self, clustered_data):
        patient_ids, y_true, y_pred_a, _ = clustered_data
        eng1 = PatientClusteredBootstrap(n_resamples=50, seed=123)
        eng2 = PatientClusteredBootstrap(n_resamples=50, seed=123)

        ci1 = eng1.compute_ci(patient_ids, y_true, y_pred_a, metric="auroc")
        ci2 = eng2.compute_ci(patient_ids, y_true, y_pred_a, metric="auroc")

        assert ci1.point_estimate == ci2.point_estimate
        assert ci1.ci_lower == ci2.ci_lower
        assert ci1.ci_upper == ci2.ci_upper


class TestCohortBootstrapSummary:
    """Verifies high-level cohort summary generation and serialization."""

    def test_compute_cohort_summary(self, clustered_data):
        patient_ids, y_true, y_pred_a, _ = clustered_data
        summary = compute_cohort_bootstrap_summary(
            patient_ids,
            y_true,
            y_pred_a,
            metrics=["auroc", "ece_uniform"],
            n_resamples=50,
            seed=42,
        )

        assert "auroc" in summary
        assert "ece_uniform" in summary
        assert "formatted" in summary["auroc"]
        assert "point_estimate" in summary["auroc"]

    def test_save_bootstrap_report(self, clustered_data):
        patient_ids, y_true, y_pred_a, _ = clustered_data
        summary = compute_cohort_bootstrap_summary(
            patient_ids,
            y_true,
            y_pred_a,
            metrics=["auroc"],
            n_resamples=20,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "bootstrap_results.json"
            saved_p = save_bootstrap_report(summary, out_file)
            assert saved_p.is_file()
            loaded = load_json(saved_p)
            assert "auroc" in loaded
