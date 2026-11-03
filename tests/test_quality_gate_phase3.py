"""Tests for Phase 3 Quality Gate & Cross-Domain Robustness Verification."""

from pathlib import Path
import pytest

from scripts.verify_phase3_gate import (
    check_calibration_and_oracle_recovery,
    check_cross_domain_configs,
    check_domain_probe_and_subgroups,
    check_generalization_gap_bounds,
    check_guardrails,
    check_threshold_transfer_collapse,
    run_phase3_quality_gate,
)


def test_check_cross_domain_configs():
    res = check_cross_domain_configs()
    assert res["passed"] is True
    assert "nih_to_chexpert" in res["validated_configs"]
    assert "chexpert_to_nih" in res["validated_configs"]


def test_check_generalization_gap_bounds_valid():
    valid_gap = {
        "headline_pneumonia": {
            "in_domain_auroc": 0.745,
            "cross_domain_auroc": 0.635,
            "delta_auroc": -0.110,
            "delta_auroc_ci_95": [-0.145, -0.075],
        }
    }
    res = check_generalization_gap_bounds(valid_gap, shift_direction="nih_to_chexpert")
    assert res["passed"] is True
    assert res["delta_auroc"] == -0.110


def test_check_generalization_gap_bounds_outside_expectation():
    abnormal_gap = {
        "headline_pneumonia": {
            "in_domain_auroc": 0.745,
            "cross_domain_auroc": 0.300,
            "delta_auroc": -0.445,  # Unrealistic massive collapse
            "delta_auroc_ci_95": [-0.50, -0.39],
        }
    }
    res = check_generalization_gap_bounds(abnormal_gap, shift_direction="nih_to_chexpert")
    assert res["passed"] is False


def test_check_calibration_and_oracle_recovery():
    valid_calib = {
        "macro_metrics": {
            "in_domain_ece": 0.040,
            "target_uncalibrated_ece": 0.120,
            "target_oracle_recalibrated_ece": 0.060,
            "macro_ece_reduction_percent": 50.0,
        }
    }
    res = check_calibration_and_oracle_recovery(valid_calib)
    assert res["passed"] is True
    assert res["macro_ece_reduction_percent"] == 50.0


def test_check_threshold_transfer_collapse():
    valid_thresh = {
        "headline_pneumonia": {
            "transfers": {
                "youden": {"delta_sensitivity": -0.15},
                "sens90": {"delta_sensitivity": -0.12},
            },
            "score_distribution_shift": {
                "ks_test": {"statistic": 0.35, "pvalue": 0.0001}
            },
        }
    }
    res = check_threshold_transfer_collapse(valid_thresh)
    assert res["passed"] is True
    assert res["ks_statistic"] == 0.35


def test_check_domain_probe_and_subgroups():
    valid_domain = {
        "domain_classifier_probe": {"probe_auroc": 0.982},
        "feature_space_divergence": {"mmd": 0.245},
        "subgroup_disparities": {
            "raw_subgroups": {"view": {}, "sex": {}, "age": {}}
        },
    }
    res = check_domain_probe_and_subgroups(valid_domain)
    assert res["passed"] is True
    assert res["probe_auroc"] >= 0.85
    assert res["mmd"] > 0.0


def test_check_guardrails():
    res = check_guardrails()
    assert res["passed"] is True
    assert len(res["violations"]) == 0


def test_run_phase3_quality_gate_smoke_test():
    res = run_phase3_quality_gate(smoke_test=True)
    assert res["overall_passed"] is True
    assert "configs" in res["checks"]
    assert "generalization_gap" in res["checks"]
    assert "calibration_breakdown" in res["checks"]
    assert "threshold_transfer" in res["checks"]
    assert "domain_gap_probing" in res["checks"]
    assert "guardrails" in res["checks"]

