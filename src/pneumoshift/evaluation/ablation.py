"""Single-task vs multi-task ablation comparison engine for pneumonia representation learning."""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pneumoshift.evaluation.bootstrap import paired_bootstrap_auroc_diff
from pneumoshift.evaluation.domain_gap import compute_feature_mmd
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.ablation")


def _extract_pneumonia_metric(metrics: Dict[str, Any], key_suffix: str) -> Optional[float]:
    """Helper to extract a metric value for pneumonia across varying naming schemes."""
    candidates = [
        f"val_pneumonia_{key_suffix}",
        f"test_pneumonia_{key_suffix}",
        f"pneumonia_{key_suffix}",
        f"{key_suffix}_pneumonia",
    ]
    for c in candidates:
        if c in metrics and metrics[c] is not None:
            try:
                val = float(metrics[c])
                if not np.isnan(val):
                    return val
            except (ValueError, TypeError):
                pass

    # Check nested per_class dict
    if "per_class" in metrics and "Pneumonia" in metrics["per_class"]:
        pneu_dict = metrics["per_class"]["Pneumonia"]
        if key_suffix in pneu_dict:
            try:
                val = float(pneu_dict[key_suffix])
                if not np.isnan(val):
                    return val
            except (ValueError, TypeError):
                pass

    return None


def compare_single_vs_multitask(
    multitask_results: Dict[str, Any],
    singletask_results: Dict[str, Any],
    cohort_name: str = "NIH ChestX-ray14",
    patient_ids: Optional[np.ndarray] = None,
    y_true: Optional[np.ndarray] = None,
    y_pred_multi: Optional[np.ndarray] = None,
    y_pred_single: Optional[np.ndarray] = None,
    feats_multi: Optional[np.ndarray] = None,
    feats_single: Optional[np.ndarray] = None,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Quantifies discrimination difference and feature representation shift between

    multi-label learning and dedicated single-task pneumonia models.

    Args:
        multitask_results: Evaluation results dictionary from multi-task model.
        singletask_results: Evaluation results dictionary from single-task ablation model.
        cohort_name: Name of evaluated dataset cohort.
        patient_ids: Optional array of patient identifiers for paired bootstrapping.
        y_true: Optional ground truth binary vector.
        y_pred_multi: Optional paired predictions from multi-task model.
        y_pred_single: Optional paired predictions from single-task model.
        feats_multi: Optional penultimate features from multi-task model (N, D).
        feats_single: Optional penultimate features from single-task model (N, D).
        n_resamples: Number of bootstrap iterations for paired significance test.
        seed: Random seed.

    Returns:
        Structured dictionary containing metric deltas, paired statistical tests,
        feature representation metrics, and a formatted Markdown summary table.
    """
    multi_auroc = _extract_pneumonia_metric(multitask_results, "auroc")
    single_auroc = _extract_pneumonia_metric(singletask_results, "auroc")

    multi_auprc = _extract_pneumonia_metric(multitask_results, "auprc")
    single_auprc = _extract_pneumonia_metric(singletask_results, "auprc")

    multi_ece = _extract_pneumonia_metric(multitask_results, "ece")
    single_ece = _extract_pneumonia_metric(singletask_results, "ece")

    delta_auroc = (multi_auroc - single_auroc) if (multi_auroc and single_auroc) else None
    delta_auprc = (multi_auprc - single_auprc) if (multi_auprc and single_auprc) else None
    delta_ece = (multi_ece - single_ece) if (multi_ece and single_ece) else None

    # Paired bootstrap hypothesis test: H0: AUROC_multi == AUROC_single
    significance_test: Optional[Dict[str, Any]] = None
    if (
        patient_ids is not None
        and y_true is not None
        and y_pred_multi is not None
        and y_pred_single is not None
    ):
        try:
            significance_test = paired_bootstrap_auroc_diff(
                patient_ids=patient_ids,
                y_true=y_true,
                y_pred_a=y_pred_multi,
                y_pred_b=y_pred_single,
                n_resamples=n_resamples,
                seed=seed,
            )
        except Exception as exc:
            logger.warning("Paired bootstrap failed: %s", exc)

    # Feature representation comparison (MMD and variance)
    feature_analysis: Optional[Dict[str, Any]] = None
    if feats_multi is not None and feats_single is not None:
        try:
            mmd_val = compute_feature_mmd(feats_multi, feats_single)
            norm_multi = float(np.mean(np.linalg.norm(feats_multi, axis=1)))
            norm_single = float(np.mean(np.linalg.norm(feats_single, axis=1)))
            feature_analysis = {
                "mmd_distance": mmd_val,
                "mean_norm_multitask": norm_multi,
                "mean_norm_singletask": norm_single,
            }
        except Exception as exc:
            logger.warning("Feature representation analysis failed: %s", exc)

    # Markdown comparison table
    rows = [
        f"### Single-Task vs Multi-Task Pneumonia Ablation ({cohort_name})",
        "",
        "| Metric | Multi-Task Model | Single-Task Ablation | Delta (Multi - Single) | Interpretation |",
        "| :--- | :---: | :---: | :---: | :--- |",
    ]

    def _fmt(val: Optional[float]) -> str:
        return f"{val:.4f}" if val is not None else "—"

    # AUROC row
    auroc_interp = "Multi-label regularizes" if (delta_auroc and delta_auroc > 0) else "Single-task superior / equivalent"
    rows.append(f"| **Pneumonia AUROC** | {_fmt(multi_auroc)} | {_fmt(single_auroc)} | {_fmt(delta_auroc)} | {auroc_interp} |")

    # AUPRC row
    auprc_interp = "Higher precision under multi-label" if (delta_auprc and delta_auprc > 0) else "Single-task comparable"
    rows.append(f"| **Pneumonia AUPRC** | {_fmt(multi_auprc)} | {_fmt(single_auprc)} | {_fmt(delta_auprc)} | {auprc_interp} |")

    # ECE row
    ece_interp = "Better calibrated" if (delta_ece and delta_ece < 0) else "Higher calibration error"
    rows.append(f"| **ECE (15 bins)** | {_fmt(multi_ece)} | {_fmt(single_ece)} | {_fmt(delta_ece)} | {ece_interp} |")

    if significance_test:
        rows.append("")
        rows.append(f"**Paired Bootstrap Significance**: $p = {significance_test['p_value']:.4f}$, "
                    f"95% CI: [{significance_test['ci_lower']:+.4f}, {significance_test['ci_upper']:+.4f}], "
                    f"Significant: **{significance_test['significant']}**")

    if feature_analysis:
        rows.append("")
        rows.append(f"**Feature Space Divergence**: MMD = `{feature_analysis['mmd_distance']:.4f}`")

    table_markdown = "\n".join(rows)

    return {
        "cohort_name": cohort_name,
        "multi_auroc": multi_auroc,
        "single_auroc": single_auroc,
        "delta_auroc": delta_auroc,
        "multi_auprc": multi_auprc,
        "single_auprc": single_auprc,
        "delta_auprc": delta_auprc,
        "multi_ece": multi_ece,
        "single_ece": single_ece,
        "delta_ece": delta_ece,
        "significance_test": significance_test,
        "feature_analysis": feature_analysis,
        "table_markdown": table_markdown,
    }

