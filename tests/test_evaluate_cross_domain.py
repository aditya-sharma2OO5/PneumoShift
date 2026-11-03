"""Tests for the unified cross-domain evaluation CLI orchestrator (Task 3.6)."""

import csv
import json
from pathlib import Path
import numpy as np
import pytest

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from scripts.evaluate_cross_domain import (
    build_unified_markdown_report,
    export_csv_tables,
    parse_args,
)


def test_build_unified_markdown_report():
    headline = {
        "finding": "Pneumonia",
        "in_domain_auroc": 0.7450,
        "cross_domain_auroc": 0.6320,
        "delta_auroc": -0.1130,
        "calibration": {
            "in_domain": {"ece_uniform": 0.0450},
            "target_uncalibrated": {"ece_uniform": 0.1280},
            "target_oracle_recalibrated": {"ece_uniform": 0.0620},
            "ece_reduction_percent": 51.56,
        },
    }
    probe = {"probe_auroc": 0.9850, "fold_auroc_std": 0.004}
    calib_m = {"macro_ece_reduction_percent": 48.2}

    report = build_unified_markdown_report(
        shift_direction="nih_to_chexpert",
        headline_pneumonia=headline,
        gen_gap_md="### Generalization Gap Table",
        calib_md="### Calibration Table",
        thresh_md="### Threshold Table",
        score_md="### Score Table",
        feature_gap_md="### Feature Gap Table",
        subgroups_md="### Subgroups Table",
        probe_metrics=probe,
        calib_metrics=calib_m,
    )

    assert "# Phase 3 Cross-Domain Robustness Report: NIH_TO_CHEXPERT" in report
    assert "0.7450" in report
    assert "0.6320" in report
    assert "-0.1130" in report
    assert "0.9850" in report
    assert "+51.6%" in report
    assert "### Generalization Gap Table" in report
    assert "### Calibration Table" in report
    assert "### Threshold Table" in report
    assert "### Score Table" in report
    assert "### Feature Gap Table" in report
    assert "### Subgroups Table" in report
    assert "8. Clinical Risk Implications & Diagnostic Safety Assessment" in report


def test_export_csv_tables(tmp_path: Path):
    classes = ["Pneumonia", "Effusion"]

    gen_gap_report = {
        "per_class": {
            "Pneumonia": {
                "in_domain_auroc": 0.74,
                "cross_domain_auroc": 0.63,
                "delta_auroc": -0.11,
                "delta_auroc_ci_95": [-0.15, -0.07],
                "p_value": 0.0001,
                "p_value_fdr_adjusted": 0.0002,
                "significant_after_fdr": True,
            },
            "Effusion": {
                "in_domain_auroc": 0.85,
                "cross_domain_auroc": 0.78,
                "delta_auroc": -0.07,
                "delta_auroc_ci_95": [-0.10, -0.04],
                "p_value": 0.002,
                "p_value_fdr_adjusted": 0.003,
                "significant_after_fdr": True,
            },
        }
    }

    calib_report = {
        "per_class": {
            "Pneumonia": {
                "in_domain": {"ece_uniform": 0.04},
                "target_uncalibrated": {"ece_uniform": 0.12},
                "target_source_scaled": {"ece_uniform": 0.11},
                "target_oracle_recalibrated": {"ece_uniform": 0.06},
                "ece_reduction_percent": 50.0,
                "temperature_source": 1.15,
                "temperature_oracle": 1.45,
            },
            "Effusion": {
                "in_domain": {"ece_uniform": 0.03},
                "target_uncalibrated": {"ece_uniform": 0.09},
                "target_source_scaled": {"ece_uniform": 0.08},
                "target_oracle_recalibrated": {"ece_uniform": 0.05},
                "ece_reduction_percent": 44.4,
                "temperature_source": 1.05,
                "temperature_oracle": 1.25,
            },
        }
    }

    thresh_report = {
        "per_class": {
            "Pneumonia": {
                "source_optimal_thresholds": {
                    "youden_threshold": 0.45,
                    "sens90_threshold": 0.25,
                    "f1_threshold": 0.40,
                    "youden": {"metrics": {"sensitivity": 0.80, "specificity": 0.82}},
                    "sens90": {"metrics": {"sensitivity": 0.90, "specificity": 0.65}},
                    "max_f1": {"metrics": {"sensitivity": 0.78, "specificity": 0.85}},
                },
                "transfers": {
                    "youden": {
                        "sensitivity": 0.65,
                        "delta_sensitivity": -0.15,
                        "specificity": 0.80,
                        "delta_specificity": -0.02,
                        "f1_score": 0.45,
                        "target_optimal_f1": 0.52,
                        "regret_f1": 0.07,
                    },
                    "sens90": {
                        "sensitivity": 0.78,
                        "delta_sensitivity": -0.12,
                        "specificity": 0.62,
                        "delta_specificity": -0.03,
                        "f1_score": 0.42,
                        "target_optimal_f1": 0.50,
                        "regret_f1": 0.08,
                    },
                    "max_f1": {
                        "sensitivity": 0.62,
                        "delta_sensitivity": -0.16,
                        "specificity": 0.82,
                        "delta_specificity": -0.03,
                        "f1_score": 0.46,
                        "target_optimal_f1": 0.52,
                        "regret_f1": 0.06,
                    },
                },
                "score_distribution_shift": {
                    "source": {"median": 0.25, "q25": 0.10, "q75": 0.45},
                    "target": {"median": 0.15, "q25": 0.05, "q75": 0.30},
                    "median_difference": -0.10,
                    "ks_test": {"statistic": 0.35, "pvalue": 0.0001},
                },
            },
            "Effusion": {
                "source_optimal_thresholds": {
                    "youden_threshold": 0.50,
                    "sens90_threshold": 0.30,
                    "f1_threshold": 0.48,
                    "youden": {"metrics": {"sensitivity": 0.85, "specificity": 0.88}},
                    "sens90": {"metrics": {"sensitivity": 0.90, "specificity": 0.75}},
                    "max_f1": {"metrics": {"sensitivity": 0.84, "specificity": 0.89}},
                },
                "transfers": {
                    "youden": {
                        "sensitivity": 0.75,
                        "delta_sensitivity": -0.10,
                        "specificity": 0.85,
                        "delta_specificity": -0.03,
                        "f1_score": 0.60,
                        "target_optimal_f1": 0.66,
                        "regret_f1": 0.06,
                    },
                    "sens90": {
                        "sensitivity": 0.82,
                        "delta_sensitivity": -0.08,
                        "specificity": 0.72,
                        "delta_specificity": -0.03,
                        "f1_score": 0.58,
                        "target_optimal_f1": 0.65,
                        "regret_f1": 0.07,
                    },
                    "max_f1": {
                        "sensitivity": 0.74,
                        "delta_sensitivity": -0.10,
                        "specificity": 0.86,
                        "delta_specificity": -0.03,
                        "f1_score": 0.61,
                        "target_optimal_f1": 0.66,
                        "regret_f1": 0.05,
                    },
                },
                "score_distribution_shift": {
                    "source": {"median": 0.30, "q25": 0.15, "q75": 0.55},
                    "target": {"median": 0.22, "q25": 0.10, "q75": 0.42},
                    "median_difference": -0.08,
                    "ks_test": {"statistic": 0.28, "pvalue": 0.001},
                },
            },
        }
    }

    domain_report = {
        "subgroup_disparities": {
            "per_class_disparities": {
                "Pneumonia": {
                    "view": {"PA_auroc": 0.70, "AP_auroc": 0.60, "delta_view_pa_minus_ap": 0.10},
                    "sex": {"M_auroc": 0.64, "F_auroc": 0.62, "delta_sex_m_minus_f": 0.02},
                    "age": {"<40_auroc": 0.66, "40-60_auroc": 0.63, ">60_auroc": 0.61, "max_age_disparity": 0.05},
                },
                "Effusion": {
                    "view": {"PA_auroc": 0.82, "AP_auroc": 0.75, "delta_view_pa_minus_ap": 0.07},
                    "sex": {"M_auroc": 0.79, "F_auroc": 0.77, "delta_sex_m_minus_f": 0.02},
                    "age": {"<40_auroc": 0.80, "40-60_auroc": 0.78, ">60_auroc": 0.76, "max_age_disparity": 0.04},
                },
            }
        }
    }

    export_csv_tables(
        out_dir=tmp_path,
        findings=classes,
        gen_gap_report=gen_gap_report,
        calib_report=calib_report,
        thresh_report=thresh_report,
        domain_report=domain_report,
    )

    # Verify all 5 CSV files exist and are populated
    for csv_name in [
        "generalization_gap.csv",
        "calibration_breakdown.csv",
        "threshold_transfer.csv",
        "score_distribution_shift.csv",
        "subgroup_disparities.csv",
    ]:
        fpath = tmp_path / csv_name
        assert fpath.is_file(), f"{csv_name} was not created"
        with open(fpath, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) >= 3  # Header + 2 classes
            assert reader[0][0] == "finding"
            assert reader[1][0] == "Pneumonia"

