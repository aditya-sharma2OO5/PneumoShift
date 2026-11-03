"""Cross-dataset CheXpert uncertainty label policy ablation engine (U-Zeros vs U-Ones vs U-Ignore).

Evaluates models trained under differing CheXpert uncertainty label policies deployed
cross-domain onto NIH ChestX-ray14. Quantifies:
  1. In-domain vs cross-domain discrimination and generalization gaps (Irvin et al., 2019 Table 2).
  2. Probability overconfidence and positive prediction bias (mean prediction vs prevalence, ECE).
  3. Threshold transfer stability and clinical specificity collapse.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.calibration import compute_calibration_metrics
from pneumoshift.evaluation.cross_domain import _extract_metric_value
from pneumoshift.evaluation.thresholds import (
    compute_binary_metrics_at_threshold,
    compute_optimal_thresholds,
)
from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.uncertainty_ablation")

SUPPORTED_POLICIES = ["u_zeros", "u_ones", "u_ignore"]


def compare_uncertainty_policy_discrimination(
    policy_results: Dict[str, Dict[str, Any]],
    findings: Optional[List[str]] = None,
    shift_direction: str = "chexpert_to_nih",
) -> Dict[str, Any]:
    """Compares in-domain and cross-domain AUROC and generalization gaps across uncertainty policies.

    Args:
        policy_results: Dictionary keyed by policy name ('u_zeros', 'u_ones', 'u_ignore'),
            where each value is a dict containing 'in_domain' and 'cross_domain' metric dicts.
        findings: Target pathology classes (default: SHARED_CLASSES).
        shift_direction: Direction of shift evaluated.

    Returns:
        Structured dictionary containing per-class and macro metrics across policies.
    """
    classes = findings or SHARED_CLASSES
    policies = [p for p in SUPPORTED_POLICIES if p in policy_results]

    per_class: Dict[str, Dict[str, Any]] = {}
    macro_summary: Dict[str, Dict[str, Any]] = {p: {"in_auroc": [], "cross_auroc": [], "gap": []} for p in policies}

    for f in classes:
        per_class[f] = {}
        for p in policies:
            p_data = policy_results[p]
            in_metrics = p_data.get("in_domain", {})
            cross_metrics = p_data.get("cross_domain", {})

            in_auc = _extract_metric_value(in_metrics, f, "auroc")
            cross_auc = _extract_metric_value(cross_metrics, f, "auroc")
            gap = (in_auc - cross_auc) if (in_auc is not None and cross_auc is not None) else None

            per_class[f][p] = {
                "in_domain_auroc": in_auc,
                "cross_domain_auroc": cross_auc,
                "gap_auroc": gap,
            }

            if in_auc is not None:
                macro_summary[p]["in_auroc"].append(in_auc)
            if cross_auc is not None:
                macro_summary[p]["cross_auroc"].append(cross_auc)
            if gap is not None:
                macro_summary[p]["gap"].append(gap)

    macro_results: Dict[str, Any] = {}
    best_policy_for_gap = None
    min_macro_gap = float("inf")

    for p in policies:
        m_in = float(np.mean(macro_summary[p]["in_auroc"])) if macro_summary[p]["in_auroc"] else None
        m_cross = float(np.mean(macro_summary[p]["cross_auroc"])) if macro_summary[p]["cross_auroc"] else None
        m_gap = float(np.mean(macro_summary[p]["gap"])) if macro_summary[p]["gap"] else None

        macro_results[p] = {
            "macro_in_domain_auroc": m_in,
            "macro_cross_domain_auroc": m_cross,
            "macro_generalization_gap": m_gap,
        }

        if m_gap is not None and m_gap < min_macro_gap:
            min_macro_gap = m_gap
            best_policy_for_gap = p

    return {
        "shift_direction": shift_direction,
        "policies_evaluated": policies,
        "per_class": per_class,
        "macro_metrics": macro_results,
        "narrowest_gap_policy": best_policy_for_gap,
    }


def quantify_probability_overconfidence(
    y_true: np.ndarray,
    policy_predictions: Dict[str, np.ndarray],
    findings: Optional[List[str]] = None,
    n_bins: int = 15,
) -> Dict[str, Any]:
    """Quantifies probability overconfidence and positive prediction bias under domain shift.

    For each policy and pathology:
        - Mean predicted probability: p̄_c = (1/N) ∑ p_{i, c}
        - Observed target prevalence: prev_c = (1/N) ∑ y_{i, c}
        - Over-prediction ratio (OPR): p̄_c / max(prev_c, 1e-5)
        - Absolute overconfidence bias: p̄_c - prev_c
        - ECE and Brier score

    Models trained under U-Ones are hypothesized to show severe overconfidence (OPR >> 1.0)
    when transferred to NIH ChestX-ray14.

    Args:
        y_true: Target ground truth array (N, C) with binary labels {0.0, 1.0}.
        policy_predictions: Dict mapping policy name to prediction array (N, C).
        findings: Target pathology classes (default: SHARED_CLASSES).
        n_bins: Number of bins for ECE calculation.

    Returns:
        Structured dict with per-class and macro overconfidence metrics across policies.
    """
    classes = findings or SHARED_CLASSES
    policies = [p for p in SUPPORTED_POLICIES if p in policy_predictions]

    yt_arr = np.array(y_true, dtype=float)
    per_class: Dict[str, Dict[str, Any]] = {}
    macro_oprs: Dict[str, List[float]] = {p: [] for p in policies}
    macro_eces: Dict[str, List[float]] = {p: [] for p in policies}
    macro_briers: Dict[str, List[float]] = {p: [] for p in policies}

    for idx, f in enumerate(classes):
        col_true = yt_arr[:, idx]
        valid = (col_true == 0.0) | (col_true == 1.0)
        yt_v = col_true[valid]
        prevalence = float(np.mean(yt_v)) if len(yt_v) > 0 else 0.0

        per_class[f] = {"observed_prevalence": prevalence, "policies": {}}

        for p in policies:
            yp_arr = np.array(policy_predictions[p], dtype=float)
            col_pred = yp_arr[valid, idx]

            cal_metrics = compute_calibration_metrics(yt_v, col_pred, n_bins=n_bins)
            mean_pred = float(np.mean(col_pred)) if len(col_pred) > 0 else 0.0
            opr = float(mean_pred / max(prevalence, 1e-5))
            bias = float(mean_pred - prevalence)

            ece_val = cal_metrics.get("ece_uniform", 0.0)
            brier_val = cal_metrics.get("brier_score", 0.0)

            per_class[f]["policies"][p] = {
                "mean_prediction": mean_pred,
                "over_prediction_ratio": opr,
                "absolute_bias": bias,
                "ece_uniform": ece_val,
                "brier_score": brier_val,
            }

            macro_oprs[p].append(opr)
            macro_eces[p].append(ece_val)
            macro_briers[p].append(brier_val)

    macro_summary: Dict[str, Any] = {}
    for p in policies:
        macro_summary[p] = {
            "macro_over_prediction_ratio": float(np.mean(macro_oprs[p])) if macro_oprs[p] else None,
            "macro_ece_uniform": float(np.mean(macro_eces[p])) if macro_eces[p] else None,
            "macro_brier_score": float(np.mean(macro_briers[p])) if macro_briers[p] else None,
        }

    return {
        "per_class": per_class,
        "macro_metrics": macro_summary,
    }


def quantify_threshold_transfer_collapse(
    source_val_true: np.ndarray,
    source_val_preds_by_policy: Dict[str, np.ndarray],
    target_true: np.ndarray,
    target_preds_by_policy: Dict[str, np.ndarray],
    findings: Optional[List[str]] = None,
    threshold_criterion: str = "youden",
) -> Dict[str, Any]:
    """Quantifies threshold drift and specificity collapse when transferring decision cutoffs.

    For each policy and finding:
        1. Select source-optimal threshold τ_source on source validation set (e.g. Youden's J).
        2. Select target-optimal threshold τ_target on target data.
        3. Evaluate τ_source transferred onto target data -> Sens_trans, Spec_trans, PPR_trans.
        4. Specificity Collapse:
           Collapse_Spec = Spec_target-opt - Spec_trans
           Under U-Ones, prediction distributions shift positive, inducing severe false positives
           (high positive prediction rate, low specificity).

    Args:
        source_val_true: Source validation ground truth labels (N_src, C).
        source_val_preds_by_policy: Dict of source validation predictions per policy.
        target_true: Target test ground truth labels (N_tgt, C).
        target_preds_by_policy: Dict of target test predictions per policy.
        findings: Target pathology classes (default: SHARED_CLASSES).
        threshold_criterion: 'youden', 'max_f1', or 'sens90'.

    Returns:
        Structured dict with per-class and macro threshold transfer and specificity collapse metrics.
    """
    classes = findings or SHARED_CLASSES
    policies = [p for p in SUPPORTED_POLICIES if p in source_val_preds_by_policy and p in target_preds_by_policy]

    src_true = np.array(source_val_true, dtype=float)
    tgt_true = np.array(target_true, dtype=float)

    per_class: Dict[str, Dict[str, Any]] = {}
    macro_collapse: Dict[str, List[float]] = {p: [] for p in policies}

    for idx, f in enumerate(classes):
        per_class[f] = {}
        for p in policies:
            s_yt = src_true[:, idx]
            s_yp = np.array(source_val_preds_by_policy[p], dtype=float)[:, idx]

            t_yt = tgt_true[:, idx]
            t_yp = np.array(target_preds_by_policy[p], dtype=float)[:, idx]

            # 1. Source optimal threshold
            s_opt = compute_optimal_thresholds(s_yt, s_yp)
            if threshold_criterion == "youden":
                tau_source = s_opt["youden_threshold"]
            elif threshold_criterion in ("f1", "max_f1"):
                tau_source = s_opt["f1_threshold"]
            else:
                tau_source = s_opt["sens90_threshold"]

            # 2. Target optimal threshold
            t_opt = compute_optimal_thresholds(t_yt, t_yp)
            if threshold_criterion == "youden":
                tau_target = t_opt["youden_threshold"]
                target_opt_spec = t_opt["youden"]["metrics"]["specificity"]
                target_opt_sens = t_opt["youden"]["metrics"]["sensitivity"]
            elif threshold_criterion in ("f1", "max_f1"):
                tau_target = t_opt["f1_threshold"]
                target_opt_spec = t_opt["max_f1"]["metrics"]["specificity"]
                target_opt_sens = t_opt["max_f1"]["metrics"]["sensitivity"]
            else:
                tau_target = t_opt["sens90_threshold"]
                target_opt_spec = t_opt["fixed_sensitivity"]["metrics"]["specificity"]
                target_opt_sens = t_opt["fixed_sensitivity"]["metrics"]["sensitivity"]

            # 3. Transferred performance
            trans_metrics = compute_binary_metrics_at_threshold(t_yt, t_yp, threshold=tau_source)
            trans_spec = trans_metrics["specificity"]
            trans_sens = trans_metrics["sensitivity"]
            trans_ppr = trans_metrics["predicted_positive_rate"]

            spec_collapse = float(target_opt_spec - trans_spec)
            thresh_drift = float(abs(tau_source - tau_target))

            per_class[f][p] = {
                "source_threshold": float(tau_source),
                "target_optimal_threshold": float(tau_target),
                "threshold_drift": thresh_drift,
                "transferred_sensitivity": trans_sens,
                "transferred_specificity": trans_spec,
                "transferred_ppr": trans_ppr,
                "target_optimal_specificity": target_opt_spec,
                "target_optimal_sensitivity": target_opt_sens,
                "specificity_collapse": spec_collapse,
            }

            macro_collapse[p].append(spec_collapse)

    macro_summary: Dict[str, Any] = {}
    for p in policies:
        macro_summary[p] = {
            "macro_specificity_collapse": float(np.mean(macro_collapse[p])) if macro_collapse[p] else None
        }

    return {
        "criterion": threshold_criterion,
        "per_class": per_class,
        "macro_metrics": macro_summary,
    }


def generate_uncertainty_ablation_tables(
    discrimination_report: Dict[str, Any],
    overconfidence_report: Optional[Dict[str, Any]] = None,
    threshold_collapse_report: Optional[Dict[str, Any]] = None,
    shift_direction: str = "chexpert_to_nih",
) -> str:
    """Formats publication Markdown tables for uncertainty policy ablation."""
    policies = discrimination_report.get("policies_evaluated", ["u_zeros", "u_ones", "u_ignore"])
    lines = [
        f"## CheXpert Uncertainty Policy Robustness Ablation ({shift_direction.upper()})",
        "",
        "### 1. Cross-Dataset Discrimination & Generalization Gap by Policy (Irvin et al., 2019 Table 2)",
        "",
        "| Finding | " + " | ".join([f"{p.upper()} In | {p.upper()} Cross | Δ {p.upper()}" for p in policies]) + " | Narrowest Gap |",
        "| :--- | " + " | ".join([":---: | :---: | :---:" for _ in policies]) + " | :---: |",
    ]

    per_class_disc = discrimination_report.get("per_class", {})
    for f, p_dict in per_class_disc.items():
        row_cols = []
        best_gap = float("inf")
        best_p = "—"

        for p in policies:
            data = p_dict.get(p, {})
            in_auc = f"{data['in_domain_auroc']:.4f}" if data.get('in_domain_auroc') is not None else "—"
            cross_auc = f"{data['cross_domain_auroc']:.4f}" if data.get('cross_domain_auroc') is not None else "—"
            gap_val = data.get("gap_auroc")
            str_gap = f"{gap_val:+.4f}" if gap_val is not None else "—"

            if gap_val is not None and gap_val < best_gap:
                best_gap = gap_val
                best_p = p.upper()

            row_cols.extend([in_auc, cross_auc, str_gap])

        bold_f = f"**{f}**" if f == HEADLINE_CLASS else f
        lines.append(f"| {bold_f} | " + " | ".join(row_cols) + f" | **{best_p}** |")

    # Macro average row
    macro_disc = discrimination_report.get("macro_metrics", {})
    macro_cols = []
    for p in policies:
        m_in = f"{macro_disc[p]['macro_in_domain_auroc']:.4f}" if macro_disc.get(p, {}).get("macro_in_domain_auroc") is not None else "—"
        m_cross = f"{macro_disc[p]['macro_cross_domain_auroc']:.4f}" if macro_disc.get(p, {}).get("macro_cross_domain_auroc") is not None else "—"
        m_gap = f"{macro_disc[p]['macro_generalization_gap']:+.4f}" if macro_disc.get(p, {}).get("macro_generalization_gap") is not None else "—"
        macro_cols.extend([f"**{m_in}**", f"**{m_cross}**", f"**{m_gap}**"])

    best_macro_p = discrimination_report.get("narrowest_gap_policy", "—")
    str_best_p = best_macro_p.upper() if isinstance(best_macro_p, str) else "—"
    lines.append(f"| **Macro Average** | " + " | ".join(macro_cols) + f" | **{str_best_p}** |")
    lines.append("")

    if overconfidence_report:
        lines.extend([
            "### 2. Probability Overconfidence & Positive Prediction Bias (Prevalence vs. p̄)",
            "",
            "| Finding | Prev | " + " | ".join([f"{p.upper()} p̄ | {p.upper()} OPR | {p.upper()} ECE" for p in policies]) + " |",
            "| :--- | :---: | " + " | ".join([":---: | :---: | :---:" for _ in policies]) + " |",
        ])
        per_class_over = overconfidence_report.get("per_class", {})
        for f, data in per_class_over.items():
            prev_str = f"{data['observed_prevalence']:.4f}"
            row_cols = [prev_str]
            for p in policies:
                p_metrics = data.get("policies", {}).get(p, {})
                mean_p = f"{p_metrics.get('mean_prediction', 0.0):.4f}"
                opr = f"{p_metrics.get('over_prediction_ratio', 0.0):.2f}x"
                ece = f"{p_metrics.get('ece_uniform', 0.0):.4f}"
                row_cols.extend([mean_p, opr, ece])

            bold_f = f"**{f}**" if f == HEADLINE_CLASS else f
            lines.append(f"| {bold_f} | " + " | ".join(row_cols) + " |")

        # Macro row for overconfidence
        m_over = overconfidence_report.get("macro_metrics", {})
        macro_row = ["—"]
        for p in policies:
            m_p = m_over.get(p, {})
            m_opr = f"**{m_p.get('macro_over_prediction_ratio', 0.0):.2f}x**"
            m_ece = f"**{m_p.get('macro_ece_uniform', 0.0):.4f}**"
            macro_row.extend(["—", m_opr, m_ece])
        lines.append(f"| **Macro Average** | " + " | ".join(macro_row) + " |")
        lines.append("")

    if threshold_collapse_report:
        lines.extend([
            "### 3. Threshold Transfer Stability & Specificity Collapse",
            "",
            "| Finding | " + " | ".join([f"{p.upper()} τ | {p.upper()} Sens | {p.upper()} Spec | {p.upper()} Collapse (Δ Spec)" for p in policies]) + " |",
            "| :--- | " + " | ".join([":---: | :---: | :---: | :---:" for _ in policies]) + " |",
        ])
        per_class_coll = threshold_collapse_report.get("per_class", {})
        for f, p_dict in per_class_coll.items():
            row_cols = []
            for p in policies:
                m = p_dict.get(p, {})
                tau = f"{m.get('source_threshold', 0.0):.4f}"
                sens = f"{m.get('transferred_sensitivity', 0.0):.4f}"
                spec = f"{m.get('transferred_specificity', 0.0):.4f}"
                coll = f"{m.get('specificity_collapse', 0.0):+.4f}"
                row_cols.extend([tau, sens, spec, coll])

            bold_f = f"**{f}**" if f == HEADLINE_CLASS else f
            lines.append(f"| {bold_f} | " + " | ".join(row_cols) + " |")

        m_coll = threshold_collapse_report.get("macro_metrics", {})
        macro_row = []
        for p in policies:
            mc = m_coll.get(p, {}).get("macro_specificity_collapse", 0.0)
            macro_row.extend(["—", "—", "—", f"**{mc:+.4f}**"])
        lines.append(f"| **Macro Average** | " + " | ".join(macro_row) + " |")
        lines.append("")

    return "\n".join(lines)


def run_uncertainty_policy_ablation_suite(
    policy_eval_results: Dict[str, Dict[str, Any]],
    target_true: Optional[np.ndarray] = None,
    target_predictions_by_policy: Optional[Dict[str, np.ndarray]] = None,
    source_val_true: Optional[np.ndarray] = None,
    source_val_predictions_by_policy: Optional[Dict[str, np.ndarray]] = None,
    findings: Optional[List[str]] = None,
    threshold_criterion: str = "youden",
    shift_direction: str = "chexpert_to_nih",
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Orchestrates comprehensive uncertainty policy ablation suite.

    Computes:
        1. Discrimination and generalization gap comparisons across policies.
        2. Probability overconfidence and positive prediction bias.
        3. Threshold transfer stability and clinical specificity collapse.
        4. Publication Markdown tables.

    Args:
        policy_eval_results: Dict of evaluation results per policy (e.g. {'u_zeros': {'in_domain': ..., 'cross_domain': ...}}).
        target_true: Optional target test ground truth matrix (N, C).
        target_predictions_by_policy: Optional dict of target prediction arrays per policy.
        source_val_true: Optional source validation ground truth matrix (N_src, C).
        source_val_predictions_by_policy: Optional dict of source validation predictions per policy.
        findings: Target pathology classes.
        threshold_criterion: 'youden', 'max_f1', or 'sens90'.
        shift_direction: Direction of shift.
        output_dir: Destination path for saving results.

    Returns:
        Structured dictionary containing all ablation reports and markdown text.
    """
    logger.info("Executing uncertainty policy ablation suite for %s...", shift_direction)

    disc_report = compare_uncertainty_policy_discrimination(
        policy_results=policy_eval_results,
        findings=findings,
        shift_direction=shift_direction,
    )

    overconfidence_report = None
    if target_true is not None and target_predictions_by_policy is not None:
        overconfidence_report = quantify_probability_overconfidence(
            y_true=target_true,
            policy_predictions=target_predictions_by_policy,
            findings=findings,
        )

    threshold_collapse_report = None
    if (
        source_val_true is not None
        and source_val_predictions_by_policy is not None
        and target_true is not None
        and target_predictions_by_policy is not None
    ):
        threshold_collapse_report = quantify_threshold_transfer_collapse(
            source_val_true=source_val_true,
            source_val_preds_by_policy=source_val_predictions_by_policy,
            target_true=target_true,
            target_preds_by_policy=target_predictions_by_policy,
            findings=findings,
            threshold_criterion=threshold_criterion,
        )

    table_md = generate_uncertainty_ablation_tables(
        discrimination_report=disc_report,
        overconfidence_report=overconfidence_report,
        threshold_collapse_report=threshold_collapse_report,
        shift_direction=shift_direction,
    )

    results = {
        "shift_direction": shift_direction,
        "discrimination_comparison": disc_report,
        "overconfidence_comparison": overconfidence_report,
        "threshold_collapse_comparison": threshold_collapse_report,
        "table_markdown": table_md,
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_json(results, out_p / f"uncertainty_ablation_{shift_direction}.json")
        with open(out_p / f"uncertainty_ablation_{shift_direction}.md", "w", encoding="utf-8") as f:
            f.write(table_md)
        logger.info("Saved uncertainty policy ablation results to %s", out_p)

    return results

