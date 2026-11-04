"""Shortcut feature detection, peripheral energy quantification, and cross-domain escalation analysis."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats

from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.xai.shortcut_analysis")


def compute_shortcut_energy_ratio(
    saliency_map: np.ndarray,
    border_fraction: float = 0.05,
    corner_fraction: float = 0.08,
) -> Dict[str, float]:
    """Calculates attribution energy localized in peripheral image borders and corners.

    High attribution in image margins frequently indicates spurious shortcut learning
    (e.g., radiopaque text markers, scanner collimation edges, or external lead wires).

    Args:
        saliency_map: 2D heatmap in [0, 1] of shape (H, W).
        border_fraction: Margin fraction for outer perimeter strip (default: 0.05).
        corner_fraction: Margin fraction for corner squares (default: 0.08).

    Returns:
        Dict containing border_energy_ratio, corner_energy_ratio, lateral_marker_energy_ratio,
        total_shortcut_energy_ratio, and central_clinical_energy_ratio.
    """
    h, w = saliency_map.shape
    total_energy = float(np.sum(saliency_map))
    if total_energy < 1e-8:
        return {
            "border_energy_ratio": 0.0,
            "corner_energy_ratio": 0.0,
            "lateral_marker_energy_ratio": 0.0,
            "total_shortcut_energy_ratio": 0.0,
            "central_clinical_energy_ratio": 1.0,
        }

    # 1. Border Mask (perimeter band)
    border_mask = np.zeros((h, w), dtype=bool)
    b_h = max(1, int(round(h * border_fraction)))
    b_w = max(1, int(round(w * border_fraction)))
    border_mask[:b_h, :] = True
    border_mask[-b_h:, :] = True
    border_mask[:, :b_w] = True
    border_mask[:, -b_w:] = True

    # 2. Corner Mask (quadrant corners)
    corner_mask = np.zeros((h, w), dtype=bool)
    c_h = max(1, int(round(h * corner_fraction)))
    c_w = max(1, int(round(w * corner_fraction)))
    corner_mask[:c_h, :c_w] = True
    corner_mask[:c_h, -c_w:] = True
    corner_mask[-c_h:, :c_w] = True
    corner_mask[-c_h:, -c_w:] = True

    # 3. Lateral Marker Mask (top-left & top-right margin where "L" / "R" lead stamps sit)
    lateral_marker_mask = np.zeros((h, w), dtype=bool)
    lm_h = max(1, int(round(h * 0.15)))
    lm_w = max(1, int(round(w * 0.15)))
    lateral_marker_mask[:lm_h, :lm_w] = True
    lateral_marker_mask[:lm_h, -lm_w:] = True

    # Union of all shortcut areas
    total_shortcut_mask = border_mask | corner_mask | lateral_marker_mask

    border_energy = float(np.sum(saliency_map[border_mask]))
    corner_energy = float(np.sum(saliency_map[corner_mask]))
    lateral_energy = float(np.sum(saliency_map[lateral_marker_mask]))
    total_shortcut_energy = float(np.sum(saliency_map[total_shortcut_mask]))

    border_ratio = float(border_energy / total_energy)
    corner_ratio = float(corner_energy / total_energy)
    lateral_ratio = float(lateral_energy / total_energy)
    total_shortcut_ratio = float(total_shortcut_energy / total_energy)
    central_clinical_ratio = float(max(0.0, 1.0 - total_shortcut_ratio))

    return {
        "border_energy_ratio": border_ratio,
        "corner_energy_ratio": corner_ratio,
        "lateral_marker_energy_ratio": lateral_ratio,
        "total_shortcut_energy_ratio": total_shortcut_ratio,
        "central_clinical_energy_ratio": central_clinical_ratio,
    }


def evaluate_cohort_shortcut_energy(
    saliency_maps: List[np.ndarray],
    border_fraction: float = 0.05,
    corner_fraction: float = 0.08,
) -> Dict[str, Any]:
    """Evaluates shortcut energy distribution across a cohort of radiographs.

    Args:
        saliency_maps: List of 2D attribution heatmaps.
        border_fraction: Fraction for outer perimeter strip.
        corner_fraction: Fraction for corner squares.

    Returns:
        Dict containing cohort means and standard deviations of shortcut metrics,
        along with per-sample ratios.
    """
    if not saliency_maps:
        raise ValueError("saliency_maps cannot be empty")

    border_ratios: List[float] = []
    corner_ratios: List[float] = []
    lateral_ratios: List[float] = []
    total_shortcut_ratios: List[float] = []
    central_ratios: List[float] = []

    for cam in saliency_maps:
        res = compute_shortcut_energy_ratio(cam, border_fraction, corner_fraction)
        border_ratios.append(res["border_energy_ratio"])
        corner_ratios.append(res["corner_energy_ratio"])
        lateral_ratios.append(res["lateral_marker_energy_ratio"])
        total_shortcut_ratios.append(res["total_shortcut_energy_ratio"])
        central_ratios.append(res["central_clinical_energy_ratio"])

    return {
        "num_samples": len(saliency_maps),
        "mean_border_energy_ratio": float(np.mean(border_ratios)),
        "std_border_energy_ratio": float(np.std(border_ratios)),
        "mean_corner_energy_ratio": float(np.mean(corner_ratios)),
        "std_corner_energy_ratio": float(np.std(corner_ratios)),
        "mean_lateral_marker_energy_ratio": float(np.mean(lateral_ratios)),
        "std_lateral_marker_energy_ratio": float(np.std(lateral_ratios)),
        "mean_total_shortcut_energy_ratio": float(np.mean(total_shortcut_ratios)),
        "std_total_shortcut_energy_ratio": float(np.std(total_shortcut_ratios)),
        "mean_central_clinical_energy_ratio": float(np.mean(central_ratios)),
        "std_central_clinical_energy_ratio": float(np.std(central_ratios)),
        "total_shortcut_ratios": total_shortcut_ratios,
        "border_ratios": border_ratios,
        "corner_ratios": corner_ratios,
    }


def compare_shortcut_escalation(
    in_domain_shortcut_ratios: List[float],
    transferred_shortcut_ratios: List[float],
    in_domain_in_lung_ratios: Optional[List[float]] = None,
    transferred_in_lung_ratios: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Tests Hypothesis 5E: Shortcut Energy Escalation & Anatomical Plausibility Breakdown.

    Hypothesis 5E states: Cross-domain distribution shift induces a statistically significant
    increase in peripheral shortcut energy (image borders and corner text markers) and a
    corresponding reduction in anatomical in-lung attribution ratio, exposing hospital-specific
    acquisition confounding.

    Args:
        in_domain_shortcut_ratios: List of total shortcut energy ratios for in-domain model.
        transferred_shortcut_ratios: List of total shortcut energy ratios for transferred model.
        in_domain_in_lung_ratios: Optional list of in-lung attribution ratios for in-domain model.
        transferred_in_lung_ratios: Optional list of in-lung attribution ratios for transferred model.

    Returns:
        Dict with statistical comparisons (Mann-Whitney U test p-values, absolute/relative
        escalation, and Hypothesis 5E verdict).
    """
    if not in_domain_shortcut_ratios or not transferred_shortcut_ratios:
        raise ValueError("Shortcut ratio lists cannot be empty")

    mean_in = float(np.mean(in_domain_shortcut_ratios))
    mean_tr = float(np.mean(transferred_shortcut_ratios))
    std_in = float(np.std(in_domain_shortcut_ratios))
    std_tr = float(np.std(transferred_shortcut_ratios))

    abs_diff = float(mean_tr - mean_in)
    rel_diff = float((abs_diff / max(mean_in, 1e-6)) * 100.0)

    # Mann-Whitney U test on shortcut ratios (alternative='greater' tests if transferred > in-domain)
    if in_domain_shortcut_ratios != transferred_shortcut_ratios:
        try:
            stat_u, p_val_shortcut = scipy.stats.mannwhitneyu(
                transferred_shortcut_ratios,
                in_domain_shortcut_ratios,
                alternative="greater",
            )
            stat_u = float(stat_u)
            p_val_shortcut = float(p_val_shortcut)
        except Exception:
            stat_u, p_val_shortcut = 0.0, 1.0
    else:
        stat_u, p_val_shortcut = 0.0, 1.0

    significant_shortcut_escalation = bool(p_val_shortcut < 0.05 and abs_diff > 0)

    # In-lung plausibility comparison if provided
    in_lung_stats: Dict[str, Any] = {}
    significant_in_lung_drop = False
    if in_domain_in_lung_ratios and transferred_in_lung_ratios:
        mean_lung_in = float(np.mean(in_domain_in_lung_ratios))
        mean_lung_tr = float(np.mean(transferred_in_lung_ratios))
        lung_abs_drop = float(mean_lung_in - mean_lung_tr)
        if in_domain_in_lung_ratios != transferred_in_lung_ratios:
            try:
                _, p_val_lung = scipy.stats.mannwhitneyu(
                    in_domain_in_lung_ratios,
                    transferred_in_lung_ratios,
                    alternative="greater",
                )
                p_val_lung = float(p_val_lung)
            except Exception:
                p_val_lung = 1.0
        else:
            p_val_lung = 1.0

        significant_in_lung_drop = bool(p_val_lung < 0.05 and lung_abs_drop > 0)
        in_lung_stats = {
            "mean_in_lung_in_domain": mean_lung_in,
            "mean_in_lung_transferred": mean_lung_tr,
            "absolute_in_lung_drop": lung_abs_drop,
            "p_value_in_lung_drop": p_val_lung,
            "significant_in_lung_drop": significant_in_lung_drop,
        }

    hypothesis_5e_supported = significant_shortcut_escalation or significant_in_lung_drop

    if hypothesis_5e_supported:
        verdict = (
            f"SUPPORTED: Out-of-domain transfer induces significant shortcut energy escalation "
            f"({mean_in:.1%} -> {mean_tr:.1%}, +{rel_diff:.1f}%, p={p_val_shortcut:.4f}), "
            f"indicating post-shift reliance on peripheral scanner artifacts and radiopaque markers."
        )
    else:
        verdict = (
            f"NOT SIGNIFICANT: Shortcut energy shift ({mean_in:.1%} -> {mean_tr:.1%}, "
            f"p={p_val_shortcut:.4f}) does not meet the p < 0.05 significance threshold."
        )

    return {
        "mean_shortcut_in_domain": mean_in,
        "std_shortcut_in_domain": std_in,
        "mean_shortcut_transferred": mean_tr,
        "std_shortcut_transferred": std_tr,
        "absolute_escalation": abs_diff,
        "relative_escalation_percent": rel_diff,
        "mann_whitney_u_statistic": stat_u,
        "p_value_shortcut_escalation": p_val_shortcut,
        "significant_shortcut_escalation": significant_shortcut_escalation,
        "in_lung_plausibility_stats": in_lung_stats,
        "hypothesis_5e_supported": hypothesis_5e_supported,
        "verdict": verdict,
    }


def format_localization_and_shortcuts_table(
    localization_results: Dict[str, Any],
    shortcut_results: Dict[str, Any],
    comparison_results: Optional[Dict[str, Any]] = None,
) -> str:
    """Formats clinical localization and shortcut analysis results into publication-ready Markdown.

    Args:
        localization_results: Output from evaluate_cohort_localization.
        shortcut_results: Output from evaluate_cohort_shortcut_energy.
        comparison_results: Optional output from compare_shortcut_escalation or compare_cross_domain_localization.

    Returns:
        Formatted GitHub Flavored Markdown string.
    """
    lines: List[str] = [
        "# Clinical Localization Benchmarking & Shortcut Energy Quantification",
        "",
        "> **Hypotheses 5D & 5E Audit**: Evaluates whether model visual explanations align with",
        "> radiologist-annotated pneumonia bounding boxes and anatomical lung parenchyma, or whether",
        "> cross-dataset transfer provokes explainability collapse and reliance on peripheral shortcut features.",
        "",
        "## 1. Ground-Truth Bounding Box Localization (Pointing Game & IoU)",
        "",
        f"- **Cohort Radiographs Evaluated**: {localization_results.get('num_samples', 'N/A')}",
        f"- **Pointing Game Hit Rate**: **{localization_results.get('pointing_game_hit_rate', 0.0):.1%}** ({localization_results.get('pointing_game_hits', 0)} / {localization_results.get('num_samples', 0)} hits)",
        f"- **Attribution Energy in Bounding Box**: **{localization_results.get('mean_energy_in_box', 0.0):.4f}** ± {localization_results.get('std_energy_in_box', 0.0):.4f}",
        f"- **Overall Mean Binarized IoU**: **{localization_results.get('overall_mean_iou', 0.0):.4f}** ± {localization_results.get('std_mean_iou', 0.0):.4f}",
        "",
        "| Binarization Threshold ($\\tau$) | Mean IoU Across Cohort |",
        "|:---|:---:|",
    ]

    iou_by_thresh = localization_results.get("mean_iou_by_threshold", {})
    for tau_str, val in iou_by_thresh.items():
        lines.append(f"| `{tau_str}` | {val:.4f} |")
    lines.append("")

    # Section 2: Shortcut Energy Quantification
    lines.extend([
        "## 2. Anatomical Plausibility & Peripheral Shortcut Energy Breakdown",
        "",
        "| Metric Compartment | Energy Ratio (Mean ± Std) | Description |",
        "|:---|:---:|:---|",
        f"| **Central Clinical Energy Ratio** | {shortcut_results.get('mean_central_clinical_energy_ratio', 0.0):.4f} ± {shortcut_results.get('std_central_clinical_energy_ratio', 0.0):.4f} | Attribution situated in central diagnostic thoracic field |",
        f"| **Total Peripheral Shortcut Energy** | {shortcut_results.get('mean_total_shortcut_energy_ratio', 0.0):.4f} ± {shortcut_results.get('std_total_shortcut_energy_ratio', 0.0):.4f} | Sum of border, corner, and lateral marker energy |",
        f"| **Border Margin Energy (5% Margin)** | {shortcut_results.get('mean_border_energy_ratio', 0.0):.4f} ± {shortcut_results.get('std_border_energy_ratio', 0.0):.4f} | Peripheral perimeter strips (collimation / edges) |",
        f"| **Corner Energy (8% Margin)** | {shortcut_results.get('mean_corner_energy_ratio', 0.0):.4f} ± {shortcut_results.get('std_corner_energy_ratio', 0.0):.4f} | Four image corner squares |",
        f"| **Lateral Lead Marker Zones** | {shortcut_results.get('mean_lateral_marker_energy_ratio', 0.0):.4f} ± {shortcut_results.get('std_lateral_marker_energy_ratio', 0.0):.4f} | Top lateral corners hosting radiopaque L/R stamps |",
        "",
    ])

    # Section 3: Comparative Cross-Domain Transfer Analysis
    if comparison_results:
        lines.extend([
            "## 3. Cross-Domain Transfer Shortcut Escalation Audit",
            "",
            f"**Hypothesis 5E Verdict**: {comparison_results.get('verdict', 'N/A')}",
            "",
            "| Comparison Dimension | In-Domain Baseline | Cross-Domain Transferred | Shift ($\\Delta$) | Mann-Whitney $p$-value |",
            "|:---|:---:|:---:|:---:|:---:|",
            f"| Total Shortcut Energy | {comparison_results.get('mean_shortcut_in_domain', 0.0):.1%} | "
            f"{comparison_results.get('mean_shortcut_transferred', 0.0):.1%} | "
            f"+{comparison_results.get('relative_escalation_percent', 0.0):.1f}% | "
            f"{comparison_results.get('p_value_shortcut_escalation', 1.0):.4e} |",
            "",
        ])

    return "\n".join(lines)


def save_localization_and_shortcuts_report(
    report_data: Dict[str, Any],
    output_path: Union[str, Path] = "results/tables/xai/localization_and_shortcuts.json",
) -> Tuple[Path, Path]:
    """Exports localization and shortcut analysis results to JSON and academic Markdown.

    Args:
        report_data: Consolidated dictionary with 'localization', 'shortcuts', and optional 'comparison'.
        output_path: Target path for the JSON export.

    Returns:
        Tuple of (json_path, markdown_path).
    """
    json_path = Path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(report_data, json_path)

    md_path = json_path.with_suffix(".md")
    loc_res = report_data.get("localization", {})
    sc_res = report_data.get("shortcuts", {})
    comp_res = report_data.get("comparison", None)

    md_content = format_localization_and_shortcuts_table(loc_res, sc_res, comp_res)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved localization & shortcuts JSON to %s and Markdown to %s", json_path, md_path)
    return json_path, md_path
