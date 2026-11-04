"""Unit tests for Phase 5 Quality Gate & Explainability Verification CLI."""

from pathlib import Path
import pytest

from scripts.verify_phase5_gate import (
    check_faithfulness_gate,
    check_localization_and_shortcuts_gate,
    check_phase5_configurations,
    check_phase5_guardrails,
    check_sanity_checks_gate,
    run_phase5_quality_gate,
)


def test_check_phase5_configurations():
    res = check_phase5_configurations()
    assert res["passed"] is True
    assert res["validated_count"] == 5
    assert len(res["failures"]) == 0


def test_check_faithfulness_gate_valid():
    valid_data = {
        "in_domain": {
            "cohort_mean_fdr": 1.45,
            "cohort_mean_insertion_auc": 0.42,
            "cohort_mean_deletion_auc": 0.29,
        },
        "comparison": {
            "fdr_drop": 0.35,
        },
    }
    res = check_faithfulness_gate(valid_data)
    assert res["passed"] is True
    assert res["fdr"] == 1.45


def test_check_faithfulness_gate_violation():
    invalid_data = {
        "in_domain": {
            "cohort_mean_fdr": 0.85,
            "cohort_mean_insertion_auc": 0.20,
            "cohort_mean_deletion_auc": 0.35,
        },
        "comparison": {},
    }
    res = check_faithfulness_gate(invalid_data)
    assert res["passed"] is False
    assert len(res["failures"]) > 0


def test_check_sanity_checks_gate_valid():
    valid_data = {
        "modes": {
            "cascading": {
                "passed_adebayo_check": True,
                "drop_percentage": 82.5,
                "mean_spearman": 0.12,
            }
        }
    }
    res = check_sanity_checks_gate(valid_data)
    assert res["passed"] is True
    assert res["drop_percentage"] == 82.5


def test_check_sanity_checks_gate_failure():
    invalid_data = {
        "modes": {
            "cascading": {
                "passed_adebayo_check": False,
                "drop_percentage": 25.0,
                "mean_spearman": 0.75,
            }
        }
    }
    res = check_sanity_checks_gate(invalid_data)
    assert res["passed"] is False


def test_check_localization_and_shortcuts_gate_valid():
    loc_data = {
        "in_domain": {
            "pointing_game_hit_rate": 0.80,
            "mean_energy_in_box": 0.65,
        }
    }
    sc_data = {
        "in_domain_shortcuts": {
            "mean_total_shortcut_energy_ratio": 0.12,
            "mean_central_clinical_energy_ratio": 0.88,
        },
        "comparison": {
            "hypothesis_5e_supported": True,
        },
    }
    res = check_localization_and_shortcuts_gate(loc_data, sc_data)
    assert res["passed"] is True
    assert res["pointing_game_hit_rate"] == 0.80


def test_check_localization_and_shortcuts_gate_missing():
    res = check_localization_and_shortcuts_gate({}, {})
    assert res["passed"] is False


def test_check_phase5_guardrails():
    res = check_phase5_guardrails()
    assert res["passed"] is True
    assert res["guardrails"]["hook_cleanup_isolation"] is True
    assert res["guardrails"]["eval_mode_enforcement"] is True


def test_run_phase5_quality_gate_smoke_test(tmp_path):
    res = run_phase5_quality_gate(results_dir=tmp_path, smoke_test=True)
    assert res["overall_passed"] is True, f"Failed checks: {res['checks']}"
    assert "configurations" in res["checks"]
    assert "faithfulness" in res["checks"]
    assert "sanity_checks" in res["checks"]
    assert "localization_and_shortcuts" in res["checks"]
    assert "guardrails" in res["checks"]
