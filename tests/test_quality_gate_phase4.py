"""Tests for Phase 4 Quality Gate & Ablation Robustness Verification CLI."""

from pathlib import Path
import pytest

from scripts.verify_phase4_gate import (
    check_architecture_comparison_bounds,
    check_guardrails,
    check_multi_seed_variance_bounds,
    check_phase4_configurations,
    check_uncertainty_ablation_bounds,
    run_phase4_quality_gate,
)


def test_check_phase4_configurations():
    res = check_phase4_configurations()
    assert res["passed"] is True
    assert res["validated_count"] == 7
    assert len(res["failures"]) == 0


def test_check_architecture_comparison_bounds_valid():
    valid_data = {
        "generalization_gap_comparison": {
            "macro_densenet_gap_auroc": 0.12,
            "macro_efficientnet_gap_auroc": 0.05,
            "macro_gap_difference": 0.07,
        },
        "calibration_comparison": {
            "densenet121": {"ece_degradation": 0.05},
            "efficientnet_b0": {"ece_degradation": 0.02},
        },
    }
    res = check_architecture_comparison_bounds(valid_data)
    assert res["passed"] is True


def test_check_architecture_comparison_bounds_missing():
    invalid_data = {
        "generalization_gap_comparison": {},
        "calibration_comparison": {},
    }
    res = check_architecture_comparison_bounds(invalid_data)
    assert res["passed"] is False


def test_check_uncertainty_ablation_bounds_valid():
    valid_data = {
        "discrimination_comparison": {
            "policies_evaluated": ["u_zeros", "u_ones", "u_ignore"]
        },
        "overconfidence_comparison": {
            "macro_metrics": {
                "u_zeros": {"macro_over_prediction_ratio": 1.1},
                "u_ones": {"macro_over_prediction_ratio": 2.8},
                "u_ignore": {"macro_over_prediction_ratio": 1.05},
            }
        },
    }
    res = check_uncertainty_ablation_bounds(valid_data)
    assert res["passed"] is True


def test_check_multi_seed_variance_bounds_valid():
    valid_data = {
        "architecture_multi_seed": {
            "densenet_aggregation": {
                "macro_metrics": {"generalization_gap": {"std": 0.012}}
            },
            "efficientnet_aggregation": {
                "macro_metrics": {"generalization_gap": {"std": 0.008}}
            },
            "macro_gap_significance": {"t_pvalue": 0.015},
        }
    }
    res = check_multi_seed_variance_bounds(valid_data)
    assert res["passed"] is True


def test_check_multi_seed_variance_bounds_high_variance():
    invalid_data = {
        "architecture_multi_seed": {
            "densenet_aggregation": {
                "macro_metrics": {"generalization_gap": {"std": 0.08}}  # > 0.05
            },
            "efficientnet_aggregation": {
                "macro_metrics": {"generalization_gap": {"std": 0.01}}
            },
            "macro_gap_significance": {"t_pvalue": 0.02},
        }
    }
    res = check_multi_seed_variance_bounds(invalid_data)
    assert res["passed"] is False


def test_check_guardrails():
    res = check_guardrails()
    assert res["passed"] is True


def test_run_phase4_quality_gate_smoke(tmp_path: Path):
    results = run_phase4_quality_gate(results_dir=tmp_path, smoke_test=True)
    assert results["overall_passed"] is True
    assert results["checks"]["configurations"]["passed"] is True
    assert results["checks"]["architecture_comparison"]["passed"] is True
    assert results["checks"]["uncertainty_ablation"]["passed"] is True
    assert results["checks"]["multi_seed_replication"]["passed"] is True
    assert results["checks"]["guardrails"]["passed"] is True

