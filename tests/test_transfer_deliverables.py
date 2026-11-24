"""Unit tests for camera-ready transfer deliverables (scripts/generate_transfer_figures.py & scripts/export_paper_tables.py)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from pneumoshift.analysis.provenance import ARCHITECTURES, SEEDS
from scripts.export_paper_tables import export_all_paper_tables
from scripts.generate_transfer_figures import (
    generate_transfer_figures,
    plot_fig1_transfer_overview,
    plot_fig2_roc_pr_curves,
    plot_fig3_calibration_reliability,
    plot_fig4_threshold_transfer_discordance,
    plot_fig5_primary_endpoints_and_policies,
    plot_fig6_subgroups_and_outlier_sensitivity,
)


def _stat_block(est: float, half_width: float = 0.03, p_value: float = 0.002) -> Dict[str, Any]:
    return {
        "estimate": float(est),
        "ci_lower": float(est - half_width),
        "ci_upper": float(est + half_width),
        "p_value": float(p_value),
    }


def _make_mock_results() -> Dict[str, Any]:
    """Constructs a complete schema-accurate mock transfer results dictionary."""
    thresholds_per_seed: Dict[str, Dict[str, float]] = {}
    source_scalers: Dict[str, Dict[str, float]] = {}
    per_seed_endpoints: List[Dict[str, Any]] = []
    secondary_calibration: List[Dict[str, Any]] = []
    recalibration_target_10pct_platt: List[Dict[str, Any]] = []
    label_policies: List[Dict[str, Any]] = []
    exploratory_subgroups: List[Dict[str, Any]] = []
    delong_sensitivity: List[Dict[str, Any]] = []

    regimes = (
        "nih_test_uncalibrated",
        "target_uncalibrated",
        "target_source_temp",
        "target_source_platt",
    )
    policies = (
        ("u_zeros", 1000, 25, 0.025),
        ("u_ones", 1000, 108, 0.108),
        ("u_excluded", 920, 25, 0.0272),
        ("explicitly_labeled_only", 40, 25, 0.625),
    )
    subgroups = (
        ("projection", "AP", 800, 18),
        ("projection", "PA", 200, 7),
        ("sex", "Female", 450, 11),
        ("sex", "Male", 550, 14),
        ("age_band", "<40", 150, 3),
        ("age_band", "40-64", 400, 9),
        ("age_band", ">=65", 450, 13),
    )

    for a_idx, arch in enumerate(ARCHITECTURES):
        for s_idx, seed in enumerate(SEEDS):
            mkey = f"{arch}_{seed}"
            thresholds_per_seed[mkey] = {
                "tau_sens95": 0.055 + 0.005 * s_idx,
                "tau_sens90": 0.095 + 0.005 * s_idx,
                "tau_youden": 0.420 + 0.010 * s_idx,
            }
            source_scalers[mkey] = {
                "temp_T": 1.25 + 0.05 * s_idx,
                "platt_a": 0.65 + 0.03 * s_idx,
                "platt_b": -3.10 - 0.05 * s_idx,
            }
            per_seed_endpoints.append({
                "architecture": arch,
                "seed": seed,
                "delta_auroc": 0.08 - 0.01 * a_idx + 0.003 * s_idx,
                "delta_slope": 0.17 - 0.14 * a_idx + 0.01 * s_idx,
                "delta_sens_tau90": -0.02 - 0.005 * a_idx,
            })
            for r_idx, reg in enumerate(regimes):
                secondary_calibration.append({
                    "architecture": arch,
                    "seed": seed,
                    "regime": reg,
                    "auroc": 0.68 - 0.05 * (r_idx > 0) + 0.002 * s_idx,
                    "auprc": 0.045 - 0.008 * (r_idx > 0) + 0.001 * s_idx,
                    "ece_uniform": 0.29 - 0.07 * r_idx + 0.005 * s_idx,
                    "ece_adaptive": 0.29 - 0.07 * r_idx + 0.004 * s_idx,
                    "brier": 0.15 - 0.03 * r_idx,
                    "bss": -5.5 + 1.5 * r_idx,
                    "nll": 0.46 - 0.09 * r_idx,
                    "slope": 0.56 - 0.08 * (r_idx == 1) + 0.02 * s_idx,
                    "intercept": -3.2 + 0.4 * r_idx,
                    "prevalence": 0.024,
                    "mean_pred": 0.31 - 0.07 * r_idx,
                })
            recalibration_target_10pct_platt.append({
                "architecture": arch,
                "seed": seed,
                "auprc_nih_test": 0.045,
                "auprc_target": 0.038,
                "source_temp_T": 1.25,
                "source_platt_a": 0.65,
                "source_platt_b": -3.10,
                "n_splits": 20,
                "seeds": list(range(100, 120)),
                "platt_a_mean": 0.52,
                "platt_a_sd": 0.03,
                "platt_b_mean": -3.65,
                "platt_b_sd": 0.08,
                "ece_uniform_mean": 0.0012 + 0.0001 * s_idx,
                "ece_uniform_sd": 0.0001,
                "ece_adaptive_mean": 0.0024 + 0.0001 * s_idx,
                "ece_adaptive_sd": 0.0002,
                "brier_mean": 0.0238,
                "brier_sd": 0.0001,
                "bss_mean": 0.004,
                "bss_sd": 0.002,
                "nll_mean": 0.1135,
                "nll_sd": 0.0008,
                "slope_mean": 1.02 + 0.01 * s_idx,
                "slope_sd": 0.03,
                "intercept_mean": 0.08,
                "intercept_sd": 0.09,
            })
            for pk, n_img, n_pos, prev in policies:
                label_policies.append({
                    "policy": pk,
                    "architecture": arch,
                    "seed": seed,
                    "n_images": n_img,
                    "n_pos_images": n_pos,
                    "image_prevalence": prev,
                    "n_patients": int(n_img * 0.4),
                    "n_pos_patients": max(1, int(n_pos * 0.8)),
                    "patient_prevalence": prev,
                    "auroc": 0.59 + (0.08 if pk == "explicitly_labeled_only" else 0.01 * a_idx) + 0.003 * s_idx,
                    "auprc": 0.036 if pk != "explicitly_labeled_only" else 0.81 + 0.005 * s_idx,
                    "sens_tau_sens90": 0.96 + 0.002 * s_idx,
                    "spec_tau_sens90": 0.07 + 0.005 * s_idx,
                    "ppv_tau_sens90": 0.025,
                    "npv_tau_sens90": 0.988,
                    "sens_tau_youden": 0.72,
                    "spec_tau_youden": 0.46,
                    "ece_uniform": 0.30 + 0.005 * s_idx,
                    "ece_adaptive": 0.30,
                    "brier": 0.15,
                    "bss": -5.1,
                    "nll": 0.47,
                    "slope": 0.43,
                    "intercept": -3.3,
                    "prevalence": prev,
                    "mean_pred": 0.32,
                })
            for ax_n, lvl, n_img, n_pos in subgroups:
                exploratory_subgroups.append({
                    "status": "EXPLORATORY",
                    "subgroup_axis": ax_n,
                    "subgroup_level": lvl,
                    "architecture": arch,
                    "seed": seed,
                    "n_images": n_img,
                    "n_pos_images": n_pos,
                    "image_prevalence": n_pos / n_img,
                    "n_patients": int(n_img * 0.4),
                    "n_pos_patients": max(1, n_pos),
                    "patient_prevalence": n_pos / max(1, int(n_img * 0.4)),
                    "auroc": 0.60 + 0.003 * s_idx - (0.01 if lvl == ">=65" else 0.0),
                    "auprc": 0.038 + 0.001 * s_idx,
                    "sens_tau_sens90": 0.96 + 0.002 * s_idx,
                    "spec_tau_sens90": 0.15 - (0.11 if lvl == ">=65" else 0.03) + 0.004 * s_idx,
                    "ece_uniform": 0.28 + 0.005 * s_idx,
                    "ece_adaptive": 0.28,
                    "brier": 0.14,
                    "bss": -5.0,
                    "nll": 0.45,
                    "slope": 0.45 + 0.01 * s_idx,
                    "intercept": -3.3,
                    "prevalence": n_pos / n_img,
                    "mean_pred": 0.31,
                })
            delong_sensitivity.append({
                "architecture": arch,
                "seed": seed,
                "auroc_a": 0.68,
                "auroc_b": 0.60,
                "delta_auroc": 0.08,
                "se": 0.01,
                "z_stat": 7.5,
                "p_value": 5e-7,
                "ci_lower": 0.06,
                "ci_upper": 0.10,
                "n_pos_a": 50,
                "n_neg_a": 450,
                "n_pos_b": 50,
                "n_neg_b": 950,
                "caveat": "synthetic unit test fixture",
            })

    metric_defaults = {
        "auroc": (0.680, 0.598, 0.082),
        "slope": (0.560, 0.420, 0.140),
        "intercept": (-3.39, -3.36, -0.03),
        "sens_tau_sens90": (0.942, 0.968, -0.026),
        "spec_tau_sens90": (0.198, 0.064, 0.134),
        "ppv_tau_sens90": (0.0253, 0.0252, 0.0001),
        "npv_tau_sens90": (0.994, 0.987, 0.007),
        "sens_tau_youden": (0.76, 0.72, 0.04),
        "spec_tau_youden": (0.52, 0.45, 0.07),
        "ppv_tau_youden": (0.034, 0.031, 0.003),
        "npv_tau_youden": (0.990, 0.985, 0.005),
        "sens_tau_sens95": (0.980, 0.990, -0.010),
        "spec_tau_sens95": (0.095, 0.024, 0.071),
        "ppv_tau_sens95": (0.024, 0.025, -0.001),
        "npv_tau_sens95": (0.995, 0.990, 0.005),
    }

    bootstrap_by_architecture: Dict[str, Any] = {}
    for arch in ARCHITECTURES:
        coh_a = {k: _stat_block(v[0]) for k, v in metric_defaults.items()}
        coh_b = {k: _stat_block(v[1]) for k, v in metric_defaults.items()}
        cont = {k: _stat_block(v[2], half_width=0.02, p_value=0.001 if k != "slope" or arch == "densenet121" else 0.68) for k, v in metric_defaults.items()}
        per_seed = {}
        for seed in SEEDS:
            per_seed[str(seed)] = {"cohort_a": coh_a, "cohort_b": coh_b, "contrast": cont}
        bootstrap_by_architecture[arch] = {
            "seeds": list(SEEDS),
            "n_resamples": 2000,
            "bootstrap_seed": 42,
            "per_seed": per_seed,
            "joint": {"cohort_a": coh_a, "cohort_b": coh_b, "contrast": cont},
        }

    primary_entries = [
        ("densenet121", "Endpoint_1_Delta_AUROC", "Delta AUROC (NIH test - CheXpert)", 0.675, 0.591, 0.084, 0.0010, 0.0015, True),
        ("densenet121", "Endpoint_2_Delta_Slope", "Delta Calibration Slope", 0.581, 0.406, 0.175, 0.0010, 0.0015, True),
        ("densenet121", "Endpoint_3_Delta_Sens_tau90", "Delta Sensitivity @ tau_sens90", 0.946, 0.966, -0.020, 0.0200, 0.0240, True),
        ("efficientnet_b0", "Endpoint_1_Delta_AUROC", "Delta AUROC (NIH test - CheXpert)", 0.683, 0.608, 0.075, 0.0010, 0.0015, True),
        ("efficientnet_b0", "Endpoint_2_Delta_Slope", "Delta Calibration Slope", 0.519, 0.502, 0.017, 0.6857, 0.6857, False),
        ("efficientnet_b0", "Endpoint_3_Delta_Sens_tau90", "Delta Sensitivity @ tau_sens90", 0.938, 0.970, -0.032, 0.0010, 0.0015, True),
    ]
    primary_list = []
    for arch, eid, elbl, m_a, m_b, delta, p_v, q_v, sig in primary_entries:
        primary_list.append({
            "endpoint_id": eid,
            "endpoint_label": elbl,
            "architecture": arch,
            "metric_key": eid,
            "nih_test_mean": m_a,
            "nih_test_ci_lower": m_a - 0.022,
            "nih_test_ci_upper": m_a + 0.022,
            "nih_test_between_seed_sd": 0.005,
            "target_mean": m_b,
            "target_ci_lower": m_b - 0.008,
            "target_ci_upper": m_b + 0.008,
            "target_between_seed_sd": 0.015,
            "delta_mean": delta,
            "delta_ci_lower": delta - 0.025,
            "delta_ci_upper": delta + 0.025,
            "delta_between_seed_sd": 0.012,
            "p_value_two_sided": p_v,
            "bh_q_value": q_v,
            "bh_significant_0_05": sig,
        })

    return {
        "provenance": {
            "mode": "PRODUCTION",
            "dry_run": False,
            "note": "Mock production payload for deliverable tests",
            "run_id": "mock_real_v1",
            "target_name": "chexpert",
            "analysis_plan_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "reference_bundle_sha256": "c" * 64,
            "git_commit": "d79bc756ba9a60d628674f2af1a683bbc25e2e6e",
            "gpu_name": "Tesla T4",
            "torch_version": "2.11.0+cu130",
            "analysis_script_path": "scripts/analyze_transfer.py",
            "analysis_script_sha256": "d" * 64,
            "analysis_module_sha256s": {},
            "bootstrap_iterations": 2000,
            "bootstrap_seed": 42,
            "target_10pct_platt_splits": 20,
            "calibration_bins": 15,
            "exclusions": [],
        },
        "cohort_summary": {
            "source_val": {"n_images": 500, "n_pos_images": 15, "image_prevalence": 0.03, "n_patients": 200, "n_pos_patients": 10, "patient_prevalence": 0.05},
            "source_test": {"n_images": 500, "n_pos_images": 15, "image_prevalence": 0.03, "n_patients": 200, "n_pos_patients": 10, "patient_prevalence": 0.05},
            "target_primary_u_zeros": {"n_images": 1000, "n_pos_images": 25, "image_prevalence": 0.025, "n_patients": 400, "n_pos_patients": 18, "patient_prevalence": 0.045},
        },
        "thresholds_per_seed": thresholds_per_seed,
        "source_scalers": source_scalers,
        "primary_endpoints_6test_bh_family": primary_list,
        "bootstrap_by_architecture": bootstrap_by_architecture,
        "secondary_calibration": secondary_calibration,
        "recalibration_target_10pct_platt": recalibration_target_10pct_platt,
        "label_policies": label_policies,
        "exploratory_subgroups": exploratory_subgroups,
        "delong_sensitivity": delong_sensitivity,
        "outlier_sensitivity_pre_e_001": [
            {
                "sensitivity_id": "pre_e_001_all",
                "model": "efficientnet_b0_2024",
                "cohort": "chexpert_all_rows",
                "n_images": 1000,
                "n_pos_images": 25,
                "auroc": 0.5981,
                "auprc": 0.0347,
                "uncal_slope": 0.4946,
                "uncal_intercept": -3.3,
                "uncal_nll": 0.5645,
                "uncal_brier": 0.16,
                "uncal_ece_uniform": 0.3852,
                "uncal_ece_adaptive": 0.3850,
                "source_temp_nll": 0.44,
                "source_temp_ece_uniform": 0.25,
                "source_platt_nll": 0.1146,
                "source_platt_ece_uniform": 0.008,
                "eff_b0_joint_delta_auroc": 0.074,
                "eff_b0_joint_delta_slope": 0.017,
            },
            {
                "sensitivity_id": "pre_e_001_excl",
                "model": "efficientnet_b0_2024",
                "cohort": "chexpert_excl_corrupted",
                "n_images": 998,
                "n_pos_images": 24,
                "auroc": 0.5983,
                "auprc": 0.0347,
                "uncal_slope": 0.5254,
                "uncal_intercept": -3.3,
                "uncal_nll": 0.5642,
                "uncal_brier": 0.16,
                "uncal_ece_uniform": 0.3852,
                "uncal_ece_adaptive": 0.3850,
                "source_temp_nll": 0.44,
                "source_temp_ece_uniform": 0.25,
                "source_platt_nll": 0.1143,
                "source_platt_ece_uniform": 0.008,
                "eff_b0_joint_delta_auroc": 0.074,
                "eff_b0_joint_delta_slope": 0.015,
            },
        ],
    }


def _make_mock_bundle(n_nih: int = 500, n_chx: int = 800) -> Dict[str, Any]:
    """Constructs synthetic prediction arrays covering both cohorts and all 6 checkpoints."""
    rng = np.random.default_rng(7)
    y_nih = (rng.random(n_nih) < 0.06).astype(int)
    raw_tgt = np.where(rng.random(n_chx) < 0.06, 1.0, 0.0)
    pids_chx = np.array([f"chx_pat_{i // 2}" for i in range(n_chx)], dtype=object)

    models: Dict[str, Dict[str, np.ndarray]] = {}
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            r = np.random.default_rng(seed + (1 if arch == "densenet121" else 2))
            z_nih = r.normal(-0.8, 1.1, size=n_nih) + 0.9 * y_nih
            z_tgt = r.normal(-0.6, 1.3, size=n_chx) + 0.6 * (raw_tgt == 1.0)
            models[f"{arch}_{seed}"] = {"nih_test_logits": z_nih, "target_logits": z_tgt}

    return {
        "source_test": {"y_true": y_nih},
        "target": {"raw_pneumonia": raw_tgt, "patient_ids": pids_chx},
        "models": models,
    }


def test_export_all_paper_tables_and_latex_escaping(tmp_path: Path) -> None:
    """Verifies all 6 paper tables export cleanly in 4 formats with proper LaTeX escaping."""
    results = _make_mock_results()
    exported = export_all_paper_tables(results, tmp_path)
    assert set(exported.keys()) == {f"table{i}" for i in range(1, 7)}
    for tid, pmap in exported.items():
        assert set(pmap.keys()) == {"csv", "json", "md", "tex"}
        for fmt, fpath in pmap.items():
            assert fpath.is_file(), f"Missing {tid}.{fmt}"
            assert fpath.stat().st_size > 100

    t1_tex = exported["table1"]["tex"].read_text(encoding="utf-8")
    t2_tex = exported["table2"]["tex"].read_text(encoding="utf-8")
    t5_tex = exported["table5"]["tex"].read_text(encoding="utf-8")
    assert r"\&" in t1_tex and r"\&" in t2_tex
    assert "q$<$0.05" in t1_tex
    assert r"$\pm$" in t2_tex
    assert r"$<$40" in t5_tex and r"$\ge$65" in t5_tex
    assert r"\toprule" in t1_tex and r"\bottomrule" in t1_tex


def test_generate_transfer_figures_synthetic_bundle(tmp_path: Path) -> None:
    """Verifies all 6 transfer figure builders export non-empty PDF, SVG, and PNG files."""
    results = _make_mock_results()
    bundle = _make_mock_bundle()
    out_a = tmp_path / "figs_a"
    out_b = tmp_path / "figs_b"
    saved = generate_transfer_figures(bundle, results, out_dirs=[out_a, out_b], formats=("pdf", "svg", "png"))

    expected_stems = {
        "fig1_study_overview",
        "fig2_roc_pr_curves",
        "fig3_calibration_reliability",
        "fig4_threshold_transfer_discordance",
        "fig5_primary_endpoints_and_policies",
        "fig6_subgroups_and_outlier_sensitivity",
    }
    assert set(saved.keys()) == expected_stems
    for out_dir in (out_a, out_b):
        for stem in expected_stems:
            for ext in ("pdf", "svg", "png"):
                fp = out_dir / f"{stem}.{ext}"
                assert fp.is_file(), f"Missing exported figure {fp}"
                assert fp.stat().st_size > 1000


def test_individual_figure_builders_axes_counts() -> None:
    """Verifies each individual figure builder constructs valid Matplotlib figures with expected axes."""
    results = _make_mock_results()
    bundle = _make_mock_bundle()
    builders = [
        (lambda: plot_fig1_transfer_overview(results), 1),
        (lambda: plot_fig2_roc_pr_curves(bundle, results), 2),
        (lambda: plot_fig3_calibration_reliability(bundle, results), 5),
        (lambda: plot_fig4_threshold_transfer_discordance(bundle, results), 2),
        (lambda: plot_fig5_primary_endpoints_and_policies(results), 2),
        (lambda: plot_fig6_subgroups_and_outlier_sensitivity(results), 2),
    ]
    for fn, min_axes in builders:
        fig = fn()
        try:
            assert len(fig.axes) >= min_axes
        finally:
            plt.close(fig)


@pytest.mark.skipif(
    not (PROJECT_ROOT / "outputs" / "transfer_oldckpt_real_v1" / "paper_tables").is_dir()
    or not (PROJECT_ROOT / "results" / "figures" / "fig2_roc_pr_curves.png").is_file(),
    reason="Production oldckpt_real_v1 figure/table deliverables not present locally.",
)
def test_production_deliverables_inventory() -> None:
    """Confirms all 18 production figure files and 24 camera-ready paper table files exist cleanly."""
    fig_dir = PROJECT_ROOT / "results" / "figures"
    stems = (
        "fig1_study_overview",
        "fig2_roc_pr_curves",
        "fig3_calibration_reliability",
        "fig4_threshold_transfer_discordance",
        "fig5_primary_endpoints_and_policies",
        "fig6_subgroups_and_outlier_sensitivity",
    )
    for stem in stems:
        for ext in ("pdf", "svg", "png"):
            p = fig_dir / f"{stem}.{ext}"
            assert p.is_file(), f"Missing production figure: {p}"
            assert p.stat().st_size > 5000

    tbl_dir = PROJECT_ROOT / "outputs" / "transfer_oldckpt_real_v1" / "paper_tables"
    files = list(tbl_dir.glob("*"))
    assert len(files) == 24
    # Confirm quarantined synthetic tokens remain blocked against production manuscript
    manuscript_text = (PROJECT_ROOT / "docs" / "MANUSCRIPT.md").read_text(encoding="utf-8")
    assert "[PENDING" not in manuscript_text
    assert "Table 6:" in manuscript_text
    assert "Figure 6" in manuscript_text
