"""Cross-dataset architecture robustness comparison engine (DenseNet121 vs EfficientNet-B0).

Compares generalization gaps (in-domain vs cross-domain AUROC and AUPRC), probability
calibration degradation (ECE, MCE, Brier score), and penultimate feature space domain
separability (linear domain probe AUROC and multi-kernel RBF MMD).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.calibration import compute_calibration_metrics
from pneumoshift.evaluation.cross_domain import _extract_metric_value
from pneumoshift.evaluation.domain_gap import compute_feature_mmd, train_domain_classifier_probe
from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.architecture_comparison")


def compare_architecture_generalization_gaps(
    densenet_in_domain: Dict[str, Any],
    densenet_cross_domain: Dict[str, Any],
    efficientnet_in_domain: Dict[str, Any],
    efficientnet_cross_domain: Dict[str, Any],
    shift_direction: str = "nih_to_chexpert",
    findings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compares discrimination generalization gaps between DenseNet121 and EfficientNet-B0.

    Generalization Gap:
        Δ_arch = In_Domain_AUROC - Cross_Domain_AUROC
        A smaller positive gap Δ indicates greater transfer robustness.
        Gap Difference = Δ_DenseNet - Δ_EfficientNet:
            > 0: EfficientNet-B0 exhibits a narrower generalization gap (more robust).
            < 0: DenseNet121 exhibits a narrower generalization gap.

    Args:
        densenet_in_domain: DenseNet121 in-domain evaluation metrics.
        densenet_cross_domain: DenseNet121 cross-domain evaluation metrics.
        efficientnet_in_domain: EfficientNet-B0 in-domain evaluation metrics.
        efficientnet_cross_domain: EfficientNet-B0 cross-domain evaluation metrics.
        shift_direction: Direction of shift evaluated (e.g. 'nih_to_chexpert').
        findings: Target pathology classes (default: SHARED_CLASSES).

    Returns:
        Structured dict with per-class and macro gap comparisons.
    """
    classes = findings or SHARED_CLASSES
    comparison: Dict[str, Dict[str, Any]] = {}

    densenet_auroc_gaps = []
    efficientnet_auroc_gaps = []

    for f in classes:
        dn_in = _extract_metric_value(densenet_in_domain, f, "auroc")
        dn_cross = _extract_metric_value(densenet_cross_domain, f, "auroc")
        eff_in = _extract_metric_value(efficientnet_in_domain, f, "auroc")
        eff_cross = _extract_metric_value(efficientnet_cross_domain, f, "auroc")

        dn_gap = (dn_in - dn_cross) if (dn_in is not None and dn_cross is not None) else None
        eff_gap = (eff_in - eff_cross) if (eff_in is not None and eff_cross is not None) else None

        if dn_gap is not None:
            densenet_auroc_gaps.append(dn_gap)
        if eff_gap is not None:
            efficientnet_auroc_gaps.append(eff_gap)

        gap_diff = (dn_gap - eff_gap) if (dn_gap is not None and eff_gap is not None) else None

        if gap_diff is not None:
            if abs(gap_diff) < 0.005:
                winner = "Comparable"
            elif gap_diff > 0:
                winner = "EfficientNet-B0"
            else:
                winner = "DenseNet121"
        else:
            winner = "N/A"

        comparison[f] = {
            "densenet_in_domain_auroc": dn_in,
            "densenet_cross_domain_auroc": dn_cross,
            "densenet_gap_auroc": dn_gap,
            "efficientnet_in_domain_auroc": eff_in,
            "efficientnet_cross_domain_auroc": eff_cross,
            "efficientnet_gap_auroc": eff_gap,
            "gap_difference": gap_diff,
            "more_robust_architecture": winner,
        }

    macro_dn_gap = float(np.mean(densenet_auroc_gaps)) if densenet_auroc_gaps else None
    macro_eff_gap = float(np.mean(efficientnet_auroc_gaps)) if efficientnet_auroc_gaps else None
    macro_diff = (
        (macro_dn_gap - macro_eff_gap)
        if (macro_dn_gap is not None and macro_eff_gap is not None)
        else None
    )

    if macro_diff is not None:
        if abs(macro_diff) < 0.005:
            overall_winner = "Comparable"
        elif macro_diff > 0:
            overall_winner = "EfficientNet-B0"
        else:
            overall_winner = "DenseNet121"
    else:
        overall_winner = "N/A"

    return {
        "shift_direction": shift_direction,
        "per_class": comparison,
        "macro_densenet_gap_auroc": macro_dn_gap,
        "macro_efficientnet_gap_auroc": macro_eff_gap,
        "macro_gap_difference": macro_diff,
        "overall_robust_architecture": overall_winner,
    }


def compare_architecture_calibration(
    densenet_in_domain: Dict[str, Any],
    densenet_cross_domain: Dict[str, Any],
    efficientnet_in_domain: Dict[str, Any],
    efficientnet_cross_domain: Dict[str, Any],
    headline_class: str = HEADLINE_CLASS,
) -> Dict[str, Any]:
    """Compares probability calibration and shift degradation between DenseNet121 and EfficientNet-B0.

    Quantifies Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
    and Brier score pre-shift (in-domain) and post-shift (cross-domain).

    Calibration Degradation:
        Δ_ECE = Cross_Domain_ECE - In_Domain_ECE
        A smaller positive Δ_ECE indicates better calibration retention.
    """
    def _get_cal(m: Dict[str, Any]) -> Dict[str, Optional[float]]:
        # If nested under calibration or headline
        if "calibration" in m:
            cal = m["calibration"]
            return {
                "ece": cal.get("ece_uniform", cal.get("ece")),
                "mce": cal.get("mce"),
                "brier": cal.get("brier_score", cal.get("brier")),
            }
        return {
            "ece": m.get("ece_uniform", m.get("val_ece", m.get("ece"))),
            "mce": m.get("mce", m.get("val_mce")),
            "brier": m.get("brier_score", m.get("brier")),
        }

    dn_in = _get_cal(densenet_in_domain)
    dn_cross = _get_cal(densenet_cross_domain)
    eff_in = _get_cal(efficientnet_in_domain)
    eff_cross = _get_cal(efficientnet_cross_domain)

    dn_ece_deg = (
        (dn_cross["ece"] - dn_in["ece"])
        if (dn_cross["ece"] is not None and dn_in["ece"] is not None)
        else None
    )
    eff_ece_deg = (
        (eff_cross["ece"] - eff_in["ece"])
        if (eff_cross["ece"] is not None and eff_in["ece"] is not None)
        else None
    )

    return {
        "headline_class": headline_class,
        "densenet121": {
            "in_domain_ece": dn_in["ece"],
            "cross_domain_ece": dn_cross["ece"],
            "ece_degradation": dn_ece_deg,
            "in_domain_mce": dn_in["mce"],
            "cross_domain_mce": dn_cross["mce"],
            "in_domain_brier": dn_in["brier"],
            "cross_domain_brier": dn_cross["brier"],
        },
        "efficientnet_b0": {
            "in_domain_ece": eff_in["ece"],
            "cross_domain_ece": eff_cross["ece"],
            "ece_degradation": eff_ece_deg,
            "in_domain_mce": eff_in["mce"],
            "cross_domain_mce": eff_cross["mce"],
            "in_domain_brier": eff_in["brier"],
            "cross_domain_brier": eff_cross["brier"],
        },
    }


def compare_architecture_domain_divergence(
    densenet_source_features: np.ndarray,
    densenet_target_features: np.ndarray,
    efficientnet_source_features: np.ndarray,
    efficientnet_target_features: np.ndarray,
    seed: int = 42,
) -> Dict[str, Any]:
    """Compares penultimate representation separability via linear domain probing and MMD.

    Evaluates whether DenseNet121 (D=1024) or EfficientNet-B0 (D=1280) features
    exhibit greater domain separability. A lower linear domain probe AUROC
    and lower MMD signify a more hospital-invariant feature representation.
    """
    logger.info("Probing DenseNet121 penultimate representation (D=%d)...", densenet_source_features.shape[1])
    dn_probe = train_domain_classifier_probe(
        source_features=densenet_source_features,
        target_features=densenet_target_features,
        seed=seed,
    )
    dn_mmd = compute_feature_mmd(densenet_source_features, densenet_target_features)

    logger.info("Probing EfficientNet-B0 penultimate representation (D=%d)...", efficientnet_source_features.shape[1])
    eff_probe = train_domain_classifier_probe(
        source_features=efficientnet_source_features,
        target_features=efficientnet_target_features,
        seed=seed,
    )
    eff_mmd = compute_feature_mmd(efficientnet_source_features, efficientnet_target_features)

    dn_auroc = dn_probe.get("probe_auroc", dn_probe.get("domain_probe_auroc", 0.5))
    eff_auroc = eff_probe.get("probe_auroc", eff_probe.get("domain_probe_auroc", 0.5))

    return {
        "densenet121": {
            "feature_dim": densenet_source_features.shape[1],
            "domain_probe_auroc": dn_auroc,
            "domain_probe_std": dn_probe.get("fold_auroc_std", 0.0),
            "mmd_distance": dn_mmd,
        },
        "efficientnet_b0": {
            "feature_dim": efficientnet_source_features.shape[1],
            "domain_probe_auroc": eff_auroc,
            "domain_probe_std": eff_probe.get("fold_auroc_std", 0.0),
            "mmd_distance": eff_mmd,
        },
        "probe_auroc_difference": (dn_auroc - eff_auroc),
        "mmd_difference": (dn_mmd - eff_mmd),
        "less_separable_feature_space": (
            "EfficientNet-B0" if eff_auroc < dn_auroc else "DenseNet121"
        ),
    }


def generate_architecture_comparison_tables(
    gap_comparison: Dict[str, Any],
    cal_comparison: Optional[Dict[str, Any]] = None,
    divergence_comparison: Optional[Dict[str, Any]] = None,
    shift_direction: str = "nih_to_chexpert",
) -> str:
    """Generates publication Markdown tables comparing DenseNet121 vs EfficientNet-B0."""
    lines = [
        f"## Model Architecture Inductive Bias Robustness ({shift_direction.upper()})",
        "",
        "### 1. Generalization Gap Comparison (DenseNet121 vs. EfficientNet-B0)",
        "",
        "| Finding | DenseNet In | DenseNet Cross | Δ DenseNet | EfficientNet In | EfficientNet Cross | Δ EfficientNet | Gap Difference (Δ_DN - Δ_EFF) | Narrower Gap |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    per_class = gap_comparison.get("per_class", {})
    for f, metrics in per_class.items():
        dn_in = f"{metrics['densenet_in_domain_auroc']:.4f}" if metrics['densenet_in_domain_auroc'] is not None else "—"
        dn_cross = f"{metrics['densenet_cross_domain_auroc']:.4f}" if metrics['densenet_cross_domain_auroc'] is not None else "—"
        dn_gap = f"{metrics['densenet_gap_auroc']:+.4f}" if metrics['densenet_gap_auroc'] is not None else "—"

        eff_in = f"{metrics['efficientnet_in_domain_auroc']:.4f}" if metrics['efficientnet_in_domain_auroc'] is not None else "—"
        eff_cross = f"{metrics['efficientnet_cross_domain_auroc']:.4f}" if metrics['efficientnet_cross_domain_auroc'] is not None else "—"
        eff_gap = f"{metrics['efficientnet_gap_auroc']:+.4f}" if metrics['efficientnet_gap_auroc'] is not None else "—"

        gap_diff = f"{metrics['gap_difference']:+.4f}" if metrics['gap_difference'] is not None else "—"
        winner = metrics.get("more_robust_architecture", "—")

        bold_f = f"**{f}**" if f == HEADLINE_CLASS else f
        lines.append(
            f"| {bold_f} | {dn_in} | {dn_cross} | {dn_gap} | {eff_in} | {eff_cross} | {eff_gap} | {gap_diff} | {winner} |"
        )

    # Macro average row
    m_dn_gap = gap_comparison.get("macro_densenet_gap_auroc")
    m_eff_gap = gap_comparison.get("macro_efficientnet_gap_auroc")
    m_diff = gap_comparison.get("macro_gap_difference")
    m_win = gap_comparison.get("overall_robust_architecture", "—")

    str_dn_gap = f"{m_dn_gap:+.4f}" if m_dn_gap is not None else "—"
    str_eff_gap = f"{m_eff_gap:+.4f}" if m_eff_gap is not None else "—"
    str_m_diff = f"{m_diff:+.4f}" if m_diff is not None else "—"

    lines.append(
        f"| **Macro Average** | — | — | **{str_dn_gap}** | — | — | **{str_eff_gap}** | **{str_m_diff}** | **{m_win}** |"
    )
    lines.append("")

    if cal_comparison:
        lines.extend([
            "### 2. Probability Calibration Shift (Pre- vs. Post-Shift)",
            "",
            "| Architecture | In-Domain ECE | Cross-Domain ECE | ECE Degradation (Δ ECE) | In-Domain Brier | Cross-Domain Brier |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        for arch_key, arch_name in [("densenet121", "DenseNet121"), ("efficientnet_b0", "EfficientNet-B0")]:
            cal = cal_comparison.get(arch_key, {})
            ece_in = f"{cal.get('in_domain_ece', 0.0):.4f}" if cal.get('in_domain_ece') is not None else "—"
            ece_cross = f"{cal.get('cross_domain_ece', 0.0):.4f}" if cal.get('cross_domain_ece') is not None else "—"
            ece_deg = f"{cal.get('ece_degradation', 0.0):+.4f}" if cal.get('ece_degradation') is not None else "—"
            brier_in = f"{cal.get('in_domain_brier', 0.0):.4f}" if cal.get('in_domain_brier') is not None else "—"
            brier_cross = f"{cal.get('cross_domain_brier', 0.0):.4f}" if cal.get('cross_domain_brier') is not None else "—"
            lines.append(f"| {arch_name} | {ece_in} | {ece_cross} | {ece_deg} | {brier_in} | {brier_cross} |")
        lines.append("")

    if divergence_comparison:
        lines.extend([
            "### 3. Penultimate Representation Space Domain Probing",
            "",
            "| Architecture | Feature Dim (D) | Domain Probe AUROC (5-Fold CV) | Multi-Kernel RBF MMD | Less Domain-Separable |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for arch_key, arch_name in [("densenet121", "DenseNet121"), ("efficientnet_b0", "EfficientNet-B0")]:
            div = divergence_comparison.get(arch_key, {})
            f_dim = div.get("feature_dim", "—")
            p_auc = div.get("domain_probe_auroc")
            p_std = div.get("domain_probe_std", 0.0)
            str_p = f"{p_auc:.4f} ± {p_std:.4f}" if p_auc is not None else "—"
            mmd_val = f"{div.get('mmd_distance', 0.0):.4f}" if div.get('mmd_distance') is not None else "—"
            less_sep = "✓" if divergence_comparison.get("less_separable_feature_space") == arch_name else " "
            lines.append(f"| {arch_name} | {f_dim} | {str_p} | {mmd_val} | {less_sep} |")
        lines.append("")

    return "\n".join(lines)


def run_architecture_comparison_suite(
    densenet_in_domain: Dict[str, Any],
    densenet_cross_domain: Dict[str, Any],
    efficientnet_in_domain: Dict[str, Any],
    efficientnet_cross_domain: Dict[str, Any],
    shift_direction: str = "nih_to_chexpert",
    findings: Optional[List[str]] = None,
    densenet_source_features: Optional[np.ndarray] = None,
    densenet_target_features: Optional[np.ndarray] = None,
    efficientnet_source_features: Optional[np.ndarray] = None,
    efficientnet_target_features: Optional[np.ndarray] = None,
    output_dir: Optional[str | Path] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Orchestrates comprehensive architecture comparison between DenseNet121 and EfficientNet-B0.

    Computes:
        1. Generalization Gap comparison across all classes.
        2. Calibration degradation comparison.
        3. Feature space divergence and domain probing (if features provided).
        4. Publication Markdown table report.

    Args:
        densenet_in_domain: DenseNet121 in-domain metrics.
        densenet_cross_domain: DenseNet121 cross-domain metrics.
        efficientnet_in_domain: EfficientNet-B0 in-domain metrics.
        efficientnet_cross_domain: EfficientNet-B0 cross-domain metrics.
        shift_direction: 'nih_to_chexpert' or 'chexpert_to_nih'.
        findings: Target classes to compare.
        densenet_source_features: Optional penultimate embeddings from DenseNet on source data.
        densenet_target_features: Optional penultimate embeddings from DenseNet on target data.
        efficientnet_source_features: Optional penultimate embeddings from EfficientNet on source data.
        efficientnet_target_features: Optional penultimate embeddings from EfficientNet on target data.
        output_dir: Destination path for saving results.
        seed: Random seed.

    Returns:
        Structured dictionary of the comparison report.
    """
    logger.info("Executing architecture comparison suite for %s...", shift_direction)

    gap_report = compare_architecture_generalization_gaps(
        densenet_in_domain=densenet_in_domain,
        densenet_cross_domain=densenet_cross_domain,
        efficientnet_in_domain=efficientnet_in_domain,
        efficientnet_cross_domain=efficientnet_cross_domain,
        shift_direction=shift_direction,
        findings=findings,
    )

    cal_report = compare_architecture_calibration(
        densenet_in_domain=densenet_in_domain,
        densenet_cross_domain=densenet_cross_domain,
        efficientnet_in_domain=efficientnet_in_domain,
        efficientnet_cross_domain=efficientnet_cross_domain,
    )

    divergence_report = None
    if (
        densenet_source_features is not None
        and densenet_target_features is not None
        and efficientnet_source_features is not None
        and efficientnet_target_features is not None
    ):
        divergence_report = compare_architecture_domain_divergence(
            densenet_source_features=densenet_source_features,
            densenet_target_features=densenet_target_features,
            efficientnet_source_features=efficientnet_source_features,
            efficientnet_target_features=efficientnet_target_features,
            seed=seed,
        )

    table_md = generate_architecture_comparison_tables(
        gap_comparison=gap_report,
        cal_comparison=cal_report,
        divergence_comparison=divergence_report,
        shift_direction=shift_direction,
    )

    results = {
        "shift_direction": shift_direction,
        "generalization_gap_comparison": gap_report,
        "calibration_comparison": cal_report,
        "divergence_comparison": divergence_report,
        "table_markdown": table_md,
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_json(results, out_p / f"architecture_comparison_{shift_direction}.json")
        with open(out_p / f"architecture_comparison_{shift_direction}.md", "w", encoding="utf-8") as f:
            f.write(table_md)
        logger.info("Saved architecture comparison results to %s", out_p)

    return results

