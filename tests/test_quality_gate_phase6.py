"""Unit tests for Phase 6 Quality Gate & Statistical Meta-Analysis Verification CLI."""

import json
from pathlib import Path
import pytest

from scripts.verify_phase6_gate import (
    EXPECTED_FORMATS,
    EXPECTED_HYPOTHESIS_IDS,
    EXPECTED_TABLE_STEMS,
    check_bootstrap_integrity,
    check_hypothesis_testing_gate,
    check_ingestion_completeness,
    check_master_tables_export,
    check_phase6_configurations,
    check_phase6_guardrails,
    run_phase6_quality_gate,
)


def test_check_phase6_configurations():
    res = check_phase6_configurations()
    assert res["passed"] is True
    assert res["validated_count"] == 3
    assert len(res["failures"]) == 0


def test_check_phase6_configurations_missing(tmp_path):
    res = check_phase6_configurations(configs_dir=tmp_path)
    assert res["passed"] is False
    assert len(res["failures"]) == 3


def test_check_ingestion_completeness_hierarchical():
    valid_manifest = {
        "phase3_cross_domain": {
            "available": True,
            "directions": {
                "nih_to_chexpert": {
                    "densenet121": {"in_domain_auroc": 0.768},
                    "efficientnet_b0": {"in_domain_auroc": 0.774},
                }
            },
        },
        "phase4_ablations": {
            "available": True,
            "uncertainty_ablation": {"u_zeros": {}, "u_ones": {}, "u_ignore": {}},
            "multi_seed": {"seeds": [42, 1337, 2024]},
        },
        "phase5_xai": {
            "available": True,
            "faithfulness": {},
            "localization_and_shortcuts": {},
        },
    }
    res = check_ingestion_completeness(valid_manifest)
    assert res["passed"] is True
    assert res["total_records"] > 0
    assert "phase3" in res["phases_represented"]
    assert "phase4" in res["phases_represented"]
    assert "phase5" in res["phases_represented"]


def test_check_ingestion_completeness_flat():
    valid_manifest = {
        "records": [
            {"phase": "phase3", "model": "densenet121"},
            {"phase": "phase4", "model": "efficientnet_b0"},
            {"phase": "phase5", "model": "densenet121"},
        ]
    }
    res = check_ingestion_completeness(valid_manifest)
    assert res["passed"] is True
    assert res["total_records"] == 3


def test_check_ingestion_completeness_missing_phases():
    invalid_manifest = {
        "records": [
            {"phase": "phase3", "model": "densenet121"},
        ]
    }
    res = check_ingestion_completeness(invalid_manifest)
    assert res["passed"] is False
    assert len(res["failures"]) >= 2  # Missing phase 4 and phase 5


def test_check_ingestion_completeness_empty():
    res = check_ingestion_completeness({})
    assert res["passed"] is False


def test_check_bootstrap_integrity_valid():
    tables_dir = Path("results/tables/master")
    res = check_bootstrap_integrity(tables_dir)
    assert res["passed"] is True
    assert res["ci_verified_count"] > 0
    assert len(res["failures"]) == 0


def test_check_bootstrap_integrity_inverted_bounds(tmp_path):
    table_file = tmp_path / "table1_cross_domain_discrimination.json"
    corrupt_data = [
        {"Cohort": "Test", "AUROC": "0.750 [0.800, 0.700]"}  # lower > upper
    ]
    with open(table_file, "w", encoding="utf-8") as f:
        json.dump(corrupt_data, f)

    res = check_bootstrap_integrity(tmp_path)
    assert res["passed"] is False
    assert any("Inverted CI" in f for f in res["failures"])


def test_check_bootstrap_integrity_point_estimate_outside_bounds(tmp_path):
    table_file = tmp_path / "table1_cross_domain_discrimination.json"
    corrupt_data = [
        {"Cohort": "Test", "AUROC": "0.950 [0.700, 0.800]"}  # est > upper
    ]
    with open(table_file, "w", encoding="utf-8") as f:
        json.dump(corrupt_data, f)

    res = check_bootstrap_integrity(tmp_path)
    assert res["passed"] is False
    assert any("outside CI" in f for f in res["failures"])


def test_check_hypothesis_testing_gate_valid():
    valid_hyp = {
        "results": [
            {"hypothesis_id": "H1", "raw_p_value": 0.0001, "fdr_q_value": 0.00025, "is_significant": True},
            {"hypothesis_id": "H2", "raw_p_value": 0.0002, "fdr_q_value": 0.00033, "is_significant": True},
            {"hypothesis_id": "H3", "raw_p_value": 0.0005, "fdr_q_value": 0.00062, "is_significant": True},
            {"hypothesis_id": "H4", "raw_p_value": 0.0039, "fdr_q_value": 0.00390, "is_significant": True},
            {"hypothesis_id": "H5", "raw_p_value": 0.0001, "fdr_q_value": 0.00025, "is_significant": True},
        ]
    }
    res = check_hypothesis_testing_gate(valid_hyp)
    assert res["passed"] is True
    assert res["significant_count"] == 5
    assert len(res["hypotheses_evaluated"]) == 5


def test_check_hypothesis_testing_gate_missing_hypothesis():
    incomplete_hyp = {
        "results": [
            {"hypothesis_id": "H1", "raw_p_value": 0.001, "fdr_q_value": 0.002, "is_significant": True},
            {"hypothesis_id": "H2", "raw_p_value": 0.002, "fdr_q_value": 0.003, "is_significant": True},
            {"hypothesis_id": "H4", "raw_p_value": 0.004, "fdr_q_value": 0.005, "is_significant": True},
            {"hypothesis_id": "H5", "raw_p_value": 0.005, "fdr_q_value": 0.006, "is_significant": True},
        ]
    }
    res = check_hypothesis_testing_gate(incomplete_hyp)
    assert res["passed"] is False
    assert any("H3 was not evaluated" in f for f in res["failures"])


def test_check_hypothesis_testing_gate_monotonicity_violation():
    monotonicity_broken = {
        "results": [
            {"hypothesis_id": "H1", "raw_p_value": 0.001, "fdr_q_value": 0.050, "is_significant": True},
            {"hypothesis_id": "H2", "raw_p_value": 0.002, "fdr_q_value": 0.010, "is_significant": True},  # smaller q for larger p!
            {"hypothesis_id": "H3", "raw_p_value": 0.003, "fdr_q_value": 0.030, "is_significant": True},
            {"hypothesis_id": "H4", "raw_p_value": 0.004, "fdr_q_value": 0.040, "is_significant": True},
            {"hypothesis_id": "H5", "raw_p_value": 0.005, "fdr_q_value": 0.050, "is_significant": True},
        ]
    }
    res = check_hypothesis_testing_gate(monotonicity_broken)
    assert res["passed"] is False
    assert any("FDR Monotonicity violation" in f for f in res["failures"])


def test_check_master_tables_export_valid():
    tables_dir = Path("results/tables/master")
    res = check_master_tables_export(tables_dir)
    assert res["passed"] is True
    assert res["exported_count"] == 24
    assert res["expected_count"] == 24


def test_check_master_tables_export_empty_dir(tmp_path):
    res = check_master_tables_export(tmp_path)
    assert res["passed"] is False
    assert res["exported_count"] == 0
    assert len(res["failures"]) == 24


def test_check_phase6_guardrails():
    res = check_phase6_guardrails()
    assert res["passed"] is True
    assert res["guardrails"]["patient_clustered_resampling"] is True
    assert res["guardrails"]["latex_booktabs_compliance"] is True


def test_run_phase6_quality_gate_smoke_test(tmp_path):
    res = run_phase6_quality_gate(results_dir=tmp_path, smoke_test=True)
    assert res["overall_passed"] is True, f"Failed checks: {res['checks']}"
    assert "configurations" in res["checks"]
    assert "ingestion_completeness" in res["checks"]
    assert "bootstrap_integrity" in res["checks"]
    assert "hypothesis_testing_fdr" in res["checks"]
    assert "master_tables_exports" in res["checks"]
    assert "guardrails" in res["checks"]
    for check_name, check_data in res["checks"].items():
        assert check_data["passed"] is True, f"Check '{check_name}' failed: {check_data.get('failures')}"

