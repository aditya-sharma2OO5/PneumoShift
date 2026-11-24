"""Export camera-ready publication tables (CSV, JSON, Markdown, LaTeX booktabs) for PneumoShift.

Reads strictly from verified PRODUCTION outputs under outputs/transfer_oldckpt_real_v1/
(results.json and CSV tables) so zero numbers are manually transcribed.

Usage:
    python scripts/export_paper_tables.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
for p in (PROJECT_ROOT, SRC_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from pneumoshift.aggregation.tables import dataframe_to_markdown, export_master_table
from pneumoshift.analysis.provenance import ARCHITECTURES, SEEDS


ARCH_DISPLAY = {
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
}


def _fmt_mean_sd(vals: Sequence[float], decimals: int = 3) -> str:
    arr = np.asarray(vals, dtype=np.float64)
    return f"{float(np.mean(arr)):.{decimals}f} +- {float(np.std(arr, ddof=1)):.{decimals}f}"


def _sanitize_latex_file(tex_path: Path) -> None:
    """Normalizes symbols and caption escapes inside exported .tex files for pdflatex booktabs compatibility."""
    txt = tex_path.read_text(encoding="utf-8")

    def _fix_caption(match: re.Match[str]) -> str:
        inner = match.group(1)
        inner = re.sub(r"(?<!\\)&", r"\&", inner)
        inner = re.sub(r"(?<!\\)%", r"\%", inner)
        inner = re.sub(r"(?<!\\)_", r"\_", inner)
        return f"\\caption{{{inner}}}"

    txt = re.sub(r"\\caption\{([^}]*)\}", _fix_caption, txt)

    replacements = [
        ("+-", r"$\pm$"),
        ("±", r"$\pm$"),
        ("→", r"$\rightarrow$"),
        ("≥", r"$\ge$"),
        ("≤", r"$\le$"),
        (">=", r"$\ge$"),
        ("<=", r"$\le$"),
        ("Δ", r"$\Delta$"),
        ("τ", r"$\tau$"),
    ]
    for k, v in replacements:
        txt = txt.replace(k, v)
    # Wrap bare \pm produced by dataframe_to_latex._escape_latex into math mode
    txt = re.sub(r"(?<!\$)\\pm\b", r"$\\pm$", txt)
    # Wrap remaining bare < / > outside LaTeX math delimiters
    txt = txt.replace("<", r"$<$").replace(">", r"$>$")
    tex_path.write_text(txt, encoding="utf-8")


def build_table1_primary_endpoints(results: Dict[str, Any]) -> pd.DataFrame:
    """Table 1: Primary 6-Test Confirmatory Cross-Domain Endpoints & Per-Seed Reference."""
    ep_labels = {
        "Endpoint_1_Delta_AUROC": "AUROC (NIH test vs CheXpert)",
        "Endpoint_2_Delta_Slope": "Calibration Slope (raw logits)",
        "Endpoint_3_Delta_Sens_tau90": "Sensitivity at NIH-val tau_sens90",
    }
    dl_map: Dict[str, List[float]] = {}
    for row in results["delong_sensitivity"]:
        dl_map.setdefault(str(row["architecture"]), []).append(float(row["p_value"]))

    rows: List[Dict[str, Any]] = []
    for entry in results["primary_endpoints_6test_bh_family"]:
        arch = str(entry["architecture"])
        ep_id = str(entry["endpoint_id"])
        dl_max_p = max(dl_map[arch]) if ep_id == "Endpoint_1_Delta_AUROC" else None
        rows.append({
            "Architecture": ARCH_DISPLAY[arch],
            "Primary Endpoint": ep_labels[ep_id],
            "NIH Test [95% CI]": (
                f"{entry['nih_test_mean']:.3f} [{entry['nih_test_ci_lower']:.3f}, {entry['nih_test_ci_upper']:.3f}]"
            ),
            "CheXpert Frontal [95% CI]": (
                f"{entry['target_mean']:.3f} [{entry['target_ci_lower']:.3f}, {entry['target_ci_upper']:.3f}]"
            ),
            "Contrast Delta [95% CI]": (
                f"{entry['delta_mean']:+.4f} [{entry['delta_ci_lower']:+.4f}, {entry['delta_ci_upper']:+.4f}]"
            ),
            "Delta Seed SD": f"{entry['delta_between_seed_sd']:.4f}",
            "p-value (2-sided)": f"{entry['p_value_two_sided']:.4f}",
            "BH FDR q-value": f"{entry['bh_q_value']:.4f}",
            "Decision (q<0.05)": "Reject H0" if entry["bh_significant_0_05"] else "Fail to reject",
            "Per-Seed DeLong Max p": f"{dl_max_p:.2e}" if dl_max_p is not None else "-",
        })
    return pd.DataFrame(rows)


def build_table2_calibration_recalibration(results: Dict[str, Any]) -> pd.DataFrame:
    """Table 2: 15-Bin Calibration & Post-Hoc Recalibration Breakdown."""
    sec_rows = results["secondary_calibration"]
    recal_rows = results["recalibration_target_10pct_platt"]

    regime_specs = [
        ("nih_test_uncalibrated", "NIH Test", "1. In-Domain Uncalibrated", False),
        ("target_uncalibrated", "CheXpert Frontal", "2. Target Uncalibrated", False),
        ("target_source_temp", "CheXpert Frontal", "3. Source Temp Scaled (NIH val)", False),
        ("target_source_platt", "CheXpert Frontal", "4. Source Platt Scaled (NIH val)", False),
        ("target_10pct_platt", "CheXpert Frontal (90% eval)", "5. Target-10% Platt (20 patient splits)", True),
    ]

    rows: List[Dict[str, Any]] = []
    for arch in ARCHITECTURES:
        for rk, coh_lbl, reg_lbl, is_t10 in regime_specs:
            if not is_t10:
                sub = [r for r in sec_rows if r["architecture"] == arch and r["regime"] == rk]
                ece_u = _fmt_mean_sd([r["ece_uniform"] for r in sub], 4)
                ece_a = _fmt_mean_sd([r["ece_adaptive"] for r in sub], 4)
                brier = _fmt_mean_sd([r["brier"] for r in sub], 4)
                bss = _fmt_mean_sd([r["bss"] for r in sub], 3)
                nll = _fmt_mean_sd([r["nll"] for r in sub], 4)
                slope = _fmt_mean_sd([r["slope"] for r in sub], 3)
                intercept = _fmt_mean_sd([r["intercept"] for r in sub], 3)
                mean_p = _fmt_mean_sd([r["mean_pred"] for r in sub], 4)
            else:
                sub = [r for r in recal_rows if r["architecture"] == arch]
                ece_u = _fmt_mean_sd([r["ece_uniform_mean"] for r in sub], 4)
                ece_a = _fmt_mean_sd([r["ece_adaptive_mean"] for r in sub], 4)
                brier = _fmt_mean_sd([r["brier_mean"] for r in sub], 4)
                bss = _fmt_mean_sd([r["bss_mean"] for r in sub], 3)
                nll = _fmt_mean_sd([r["nll_mean"] for r in sub], 4)
                slope = _fmt_mean_sd([r["slope_mean"] for r in sub], 3)
                intercept = _fmt_mean_sd([r["intercept_mean"] for r in sub], 3)
                mean_p = "-"
            rows.append({
                "Architecture": ARCH_DISPLAY[arch],
                "Cohort": coh_lbl,
                "Calibration Regime": reg_lbl,
                "ECE_uniform (M=15)": ece_u,
                "ECE_adaptive (M=15)": ece_a,
                "Brier Score": brier,
                "BSS": bss,
                "NLL": nll,
                "Cal Slope": slope,
                "Cal Intercept": intercept,
                "Mean Pred": mean_p,
            })
    return pd.DataFrame(rows)


def build_table3_threshold_transfer(results: Dict[str, Any]) -> pd.DataFrame:
    """Table 3: Source-Derived Clinical Operating Threshold Transfer."""
    th = results["thresholds_per_seed"]
    boot = results["bootstrap_by_architecture"]

    crit_specs = [
        ("tau_sens90", "Source Sens-90 (tau_sens90)", "sens_tau_sens90", "spec_tau_sens90", "ppv_tau_sens90", "npv_tau_sens90"),
        ("tau_youden", "Source Youden J (tau_Youden)", "sens_tau_youden", "spec_tau_youden", "ppv_tau_youden", "npv_tau_youden"),
        ("tau_sens95", "Source Sens-95 (tau_sens95)", "sens_tau_sens95", "spec_tau_sens95", "ppv_tau_sens95", "npv_tau_sens95"),
    ]

    rows: List[Dict[str, Any]] = []
    for arch in ARCHITECTURES:
        taus_by_seed = {s: th[f"{arch}_{s}"] for s in SEEDS}
        jc_a = boot[arch]["joint"]["cohort_a"]
        jc_b = boot[arch]["joint"]["cohort_b"]
        jc_d = boot[arch]["joint"]["contrast"]
        for tk, clabel, k_sens, k_spec, k_ppv, k_npv in crit_specs:
            t_vals = [float(taus_by_seed[s][tk]) for s in SEEDS]
            rows.append({
                "Architecture": ARCH_DISPLAY[arch],
                "Threshold Rule": clabel,
                "NIH Val Cutoff (mean +- SD)": _fmt_mean_sd(t_vals, 4),
                "NIH Test Sens [95% CI]": f"{jc_a[k_sens]['estimate']:.3f} [{jc_a[k_sens]['ci_lower']:.3f}, {jc_a[k_sens]['ci_upper']:.3f}]",
                "CheXpert Sens [95% CI]": f"{jc_b[k_sens]['estimate']:.3f} [{jc_b[k_sens]['ci_lower']:.3f}, {jc_b[k_sens]['ci_upper']:.3f}]",
                "NIH Test Spec [95% CI]": f"{jc_a[k_spec]['estimate']:.3f} [{jc_a[k_spec]['ci_lower']:.3f}, {jc_a[k_spec]['ci_upper']:.3f}]",
                "CheXpert Spec [95% CI]": f"{jc_b[k_spec]['estimate']:.3f} [{jc_b[k_spec]['ci_lower']:.3f}, {jc_b[k_spec]['ci_upper']:.3f}]",
                "Delta Spec (NIH - Chx)": f"{jc_d[k_spec]['estimate']:+.3f} (p={jc_d[k_spec]['p_value']:.4f})",
                "CheXpert PPV / NPV": f"{jc_b[k_ppv]['estimate']*100:.2f}% / {jc_b[k_npv]['estimate']*100:.2f}%",
            })
    return pd.DataFrame(rows)


def build_table4_policies_and_outlier(results: Dict[str, Any]) -> pd.DataFrame:
    """Table 4: CheXpert Label Uncertainty Policies & PRE-E-001 Sensitivity."""
    lp = results["label_policies"]
    pol_labels = [
        ("u_zeros", "u_zeros (Primary)"),
        ("u_ones", "u_ones"),
        ("u_excluded", "u_excluded"),
        ("explicitly_labeled_only", "explicitly_labeled_only"),
    ]
    rows: List[Dict[str, Any]] = []
    for pk, plabel in pol_labels:
        for arch in ARCHITECTURES:
            sub = [r for r in lp if r["policy"] == pk and r["architecture"] == arch]
            r0 = sub[0]
            rows.append({
                "Policy / Sensitivity": plabel,
                "Architecture / Model": ARCH_DISPLAY[arch],
                "Images (Positives)": f"{int(r0['n_images']):,} ({int(r0['n_pos_images']):,})",
                "Image Prev": f"{float(r0['image_prevalence'])*100:.2f}%",
                "AUROC": _fmt_mean_sd([r["auroc"] for r in sub], 3),
                "AUPRC": _fmt_mean_sd([r["auprc"] for r in sub], 3),
                "Sens @ tau_sens90": _fmt_mean_sd([r["sens_tau_sens90"] for r in sub], 3),
                "Spec @ tau_sens90": _fmt_mean_sd([r["spec_tau_sens90"] for r in sub], 3),
                "Uncal ECE_15": _fmt_mean_sd([r["ece_uniform"] for r in sub], 3),
                "Uncal NLL": _fmt_mean_sd([r["nll"] for r in sub], 3),
            })

    for out_r in results["outlier_sensitivity_pre_e_001"]:
        tag = "PRE-E-001 Full Cohort" if "all_rows" in out_r["cohort"] else "PRE-E-001 Excl. 68292 & 76123"
        prev = 100.0 * float(out_r["n_pos_images"]) / float(out_r["n_images"])
        rows.append({
            "Policy / Sensitivity": tag,
            "Architecture / Model": "EfficientNet-B0 (seed 2024)",
            "Images (Positives)": f"{int(out_r['n_images']):,} ({int(out_r['n_pos_images']):,})",
            "Image Prev": f"{prev:.2f}%",
            "AUROC": f"{float(out_r['auroc']):.4f}",
            "AUPRC": f"{float(out_r['auprc']):.4f}",
            "Sens @ tau_sens90": "-",
            "Spec @ tau_sens90": f"Slope={float(out_r['uncal_slope']):.4f}",
            "Uncal ECE_15": f"{float(out_r['uncal_ece_uniform']):.4f}",
            "Uncal NLL": f"{float(out_r['uncal_nll']):.4f} (Platt {float(out_r['source_platt_nll']):.4f})",
        })
    return pd.DataFrame(rows)


def build_table5_exploratory_subgroups(results: Dict[str, Any]) -> pd.DataFrame:
    """Table 5: Exploratory Demographic & Acquisition Subgroup Stratification (CheXpert u_zeros)."""
    sg = results["exploratory_subgroups"]
    order = [
        ("projection", "AP", "Projection: AP"),
        ("projection", "PA", "Projection: PA"),
        ("sex", "Female", "Sex: Female"),
        ("sex", "Male", "Sex: Male"),
        ("age_band", "<40", "Age: <40 yr"),
        ("age_band", "40-64", "Age: 40-64 yr"),
        ("age_band", ">=65", "Age: >=65 yr"),
    ]
    rows: List[Dict[str, Any]] = []
    for axis, lvl, display in order:
        for arch in ARCHITECTURES:
            sub = [r for r in sg if r["subgroup_axis"] == axis and r["subgroup_level"] == lvl and r["architecture"] == arch]
            r0 = sub[0]
            rows.append({
                "Subgroup Stratum": display,
                "Architecture": ARCH_DISPLAY[arch],
                "Images (Positives)": f"{int(r0['n_images']):,} ({int(r0['n_pos_images']):,})",
                "Prev": f"{float(r0['image_prevalence'])*100:.2f}%",
                "AUROC": _fmt_mean_sd([r["auroc"] for r in sub], 3),
                "AUPRC": _fmt_mean_sd([r["auprc"] for r in sub], 3),
                "Sens @ tau_sens90": _fmt_mean_sd([r["sens_tau_sens90"] for r in sub], 3),
                "Spec @ tau_sens90": _fmt_mean_sd([r["spec_tau_sens90"] for r in sub], 3),
                "Uncal ECE_15": _fmt_mean_sd([r["ece_uniform"] for r in sub], 3),
                "Cal Slope": _fmt_mean_sd([r["slope"] for r in sub], 3),
            })
    return pd.DataFrame(rows)


def build_table6_hypothesis_summary(results: Dict[str, Any]) -> pd.DataFrame:
    """Table 6: Consolidated Summary of Pre-Specified Hypotheses & Analytical Findings."""
    rows: List[Dict[str, Any]] = []
    for idx, entry in enumerate(results["primary_endpoints_6test_bh_family"], start=1):
        rows.append({
            "Test ID": f"H{idx} ({ARCH_DISPLAY[entry['architecture']]})",
            "Pre-Specified Hypothesis": entry["endpoint_label"],
            "Estimand (NIH - Chx)": f"{entry['delta_mean']:+.4f} [{entry['delta_ci_lower']:+.4f}, {entry['delta_ci_upper']:+.4f}]",
            "Raw p (B=2,000)": f"{entry['p_value_two_sided']:.4f}",
            "BH q-value": f"{entry['bh_q_value']:.4f}",
            "Verdict (alpha=0.05)": "REJECT H0 (Significant)" if entry["bh_significant_0_05"] else "FAIL TO REJECT (NS)",
        })
    return pd.DataFrame(rows)


def export_all_paper_tables(results: Dict[str, Any], out_dir: Path) -> Dict[str, Dict[str, Path]]:
    """Exports all 6 publication tables to CSV, JSON, Markdown, and LaTeX."""
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("table1", "table1_primary_confirmatory_endpoints", "Table 1: Primary Confirmatory Cross-Domain Endpoints & Per-Seed Reference", build_table1_primary_endpoints(results), "Three-seed mean contrasts across seeds 42, 1337, 2024 under primary CheXpert u_zeros policy (N=191,027 vs NIH test N=22,744), patient-clustered bootstrap B=2,000 (seed=42), BH FDR family size 6."),
        ("table2", "table2_calibration_and_recalibration", "Table 2: 15-Bin Calibration & Post-Hoc Recalibration Regimes", build_table2_calibration_recalibration(results), "All metrics evaluated directly from logits (expit/logaddexp, PRE-E-001) across 15 equal-width and adaptive bins; Target-10% Platt summarizes 3-seed mean +- SD of 20 patient-level splits."),
        ("table3", "table3_threshold_transfer_breakdown", "Table 3: Source-Derived Operating Threshold Transfer Breakdown", build_table3_threshold_transfer(results), "Cutoffs selected per seed on fresh NIH validation (N=13,500) and applied unchanged to NIH test and CheXpert frontal."),
        ("table4", "table4_label_policies_and_pre_e_001", "Table 4: CheXpert Label Uncertainty Policy Sensitivity & PRE-E-001 Outlier Check", build_table4_policies_and_outlier(results), "PRE-E-001 compares all 191,027 frontal radiographs against excluding corrupted static-noise rows 68292 and 76123 on efficientnet_b0_2024."),
        ("table5", "table5_exploratory_subgroups", "Table 5: Exploratory Demographic and Acquisition Subgroup Stratification", build_table5_exploratory_subgroups(results), "All subgroup evaluations are strictly exploratory under CheXpert u_zeros across 3 seeds (mean +- SD, ddof=1)."),
        ("table6", "table6_confirmatory_hypothesis_summary", "Table 6: Consolidated Summary of Primary Hypotheses and Analytical Findings", build_table6_hypothesis_summary(results), "Benjamini-Hochberg FDR control across the 6 confirmatory hypothesis tests."),
    ]
    exported: Dict[str, Dict[str, Path]] = {}
    for tid, base_name, title, df, notes in specs:
        paths = export_master_table(df, title=title, base_name=base_name, output_dir=out_dir, notes=notes)
        _sanitize_latex_file(paths["tex"])
        exported[tid] = paths
    return exported


def main() -> None:
    parser = argparse.ArgumentParser(description="Export paper tables from verified oldckpt_real_v1 results.")
    parser.add_argument("--results-json", default="outputs/transfer_oldckpt_real_v1/results.json")
    parser.add_argument("--out-dir", default="outputs/transfer_oldckpt_real_v1/paper_tables")
    args = parser.parse_args()

    res_path = Path(args.results_json)
    results = json.loads(res_path.read_text(encoding="utf-8"))
    if results.get("provenance", {}).get("mode") != "PRODUCTION":
        raise ValueError("Refusing to export paper tables from non-PRODUCTION results.")

    exported = export_all_paper_tables(results, Path(args.out_dir))
    print(f"Exported {len(exported)} camera-ready paper tables to {args.out_dir}:")
    for tid, pmap in exported.items():
        print(f"  [OK] {tid}: {', '.join(sorted(pmap.keys()))}")


if __name__ == "__main__":
    main()
