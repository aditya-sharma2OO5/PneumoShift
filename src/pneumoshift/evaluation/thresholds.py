"""Decision threshold selection, transferability drop analysis, and score distribution shifts.

Evaluates decision threshold collapse and continuous prediction score divergence
when deploying models across domain boundaries (e.g., NIH ChestX-ray14 <-> Stanford CheXpert),
mirroring the methodology of Pan et al. (2019).
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import scipy.stats
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.thresholds")


def compute_binary_metrics_at_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculates comprehensive binary classification metrics at a fixed decision threshold.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].
        threshold: Decision cutoff threshold in [0, 1].

    Returns:
        Dict containing sensitivity, specificity, precision, NPV, F1 score,
        Youden's J, accuracy, balanced accuracy, and confusion counts.
    """
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    yp = y_pred[valid]

    n_total = len(yt)
    if n_total == 0:
        return {
            "threshold": float(threshold),
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "npv": 0.0,
            "f1_score": 0.0,
            "youden_j": 0.0,
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "positive_prevalence": 0.0,
            "predicted_positive_rate": 0.0,
        }

    bin_pred = (yp >= threshold).astype(int)
    tp = int(np.sum((bin_pred == 1) & (yt == 1)))
    fp = int(np.sum((bin_pred == 1) & (yt == 0)))
    tn = int(np.sum((bin_pred == 0) & (yt == 0)))
    fn = int(np.sum((bin_pred == 0) & (yt == 1)))

    sens = float(tp / max(tp + fn, 1))
    spec = float(tn / max(tn + fp, 1))
    prec = float(tp / max(tp + fp, 1)) if (tp + fp) > 0 else 0.0
    npv = float(tn / max(tn + fn, 1)) if (tn + fn) > 0 else 0.0
    f1 = float(2.0 * prec * sens / max(prec + sens, 1e-8)) if (prec + sens) > 0 else 0.0
    youden = float(sens + spec - 1.0)
    acc = float((tp + tn) / n_total)
    bacc = float((sens + spec) / 2.0)

    return {
        "threshold": float(threshold),
        "sensitivity": sens,
        "specificity": spec,
        "precision": prec,
        "npv": npv,
        "f1_score": f1,
        "youden_j": youden,
        "accuracy": acc,
        "balanced_accuracy": bacc,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "positive_prevalence": float(np.mean(yt)),
        "predicted_positive_rate": float(np.mean(bin_pred)),
    }


def compute_optimal_thresholds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_sensitivity: float = 0.90,
) -> Dict[str, Any]:
    """Computes source-optimal classification thresholds based on distinct clinical criteria.

    Three standard clinical regimes:
      1. Youden's J index: argmax (TPR - FPR) = argmax (Sensitivity + Specificity - 1)
      2. Max F1 score: argmax (2 * Precision * Recall / (Precision + Recall))
      3. Fixed Sensitivity (e.g. 90%): Highest threshold guaranteeing Sensitivity >= target_sensitivity

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].
        target_sensitivity: Desired minimum sensitivity (default: 0.90).

    Returns:
        Structured dict with optimal thresholds and performance metrics for each criterion.
    """
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    yp = y_pred[valid]

    n_samples = len(yt)
    n_pos = int(np.sum(yt == 1))

    if n_samples == 0 or n_pos == 0 or n_pos == n_samples:
        default_metrics = compute_binary_metrics_at_threshold(yt, yp, 0.5)
        return {
            "youden_threshold": 0.5,
            "f1_threshold": 0.5,
            "sens90_threshold": 0.5,
            "youden": {"threshold": 0.5, "metrics": default_metrics},
            "max_f1": {"threshold": 0.5, "metrics": default_metrics},
            "fixed_sensitivity": {
                "threshold": 0.5,
                "target_sensitivity": target_sensitivity,
                "metrics": default_metrics,
            },
        }

    # 1. Youden's J threshold from ROC curve
    fpr, tpr, roc_thresh = roc_curve(yt, yp)
    # Clip thresholds to [0.0, 1.0] (roc_curve sets first threshold to max + 1)
    roc_thresh_clipped = np.clip(roc_thresh, 0.0, 1.0)
    youden_indices = tpr - fpr
    youden_idx = int(np.argmax(youden_indices))
    youden_threshold = float(roc_thresh_clipped[youden_idx])
    youden_metrics = compute_binary_metrics_at_threshold(yt, yp, youden_threshold)

    # 2. Maximum F1 threshold from Precision-Recall curve
    precisions, recalls, pr_thresh = precision_recall_curve(yt, yp)
    pr_thresh_clipped = np.clip(pr_thresh, 0.0, 1.0)
    denom = precisions + recalls
    f1_scores = np.where(denom > 0, 2.0 * (precisions * recalls) / np.maximum(denom, 1e-8), 0.0)

    if len(pr_thresh_clipped) > 0:
        best_f1_idx = int(np.argmax(f1_scores[:-1]))
        f1_threshold = float(pr_thresh_clipped[best_f1_idx])
    else:
        f1_threshold = 0.5
    f1_metrics = compute_binary_metrics_at_threshold(yt, yp, f1_threshold)

    # 3. Fixed Sensitivity (>= target_sensitivity) threshold
    eligible = np.where(tpr >= target_sensitivity)[0]
    if len(eligible) > 0:
        # Highest threshold that still meets the sensitivity requirement
        sens_threshold = float(np.max(roc_thresh_clipped[eligible]))
    else:
        # If no point reaches target_sensitivity, pick the one with highest TPR
        sens_threshold = float(roc_thresh_clipped[np.argmax(tpr)])
    sens_metrics = compute_binary_metrics_at_threshold(yt, yp, sens_threshold)

    return {
        "youden_threshold": youden_threshold,
        "f1_threshold": f1_threshold,
        "sens90_threshold": sens_threshold,
        "youden": {
            "threshold": youden_threshold,
            "metrics": youden_metrics,
        },
        "max_f1": {
            "threshold": f1_threshold,
            "metrics": f1_metrics,
        },
        "fixed_sensitivity": {
            "threshold": sens_threshold,
            "target_sensitivity": target_sensitivity,
            "metrics": sens_metrics,
        },
    }


def evaluate_threshold_transfer(
    applied_threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    source_metrics: Optional[Dict[str, float]] = None,
    target_optimal_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluates cross-dataset threshold transferability and performance degradation.

    Quantifies performance drops (ΔSensitivity, ΔSpecificity, ΔF1) and decision regret
    when applying a threshold optimized on the source domain directly onto the target domain.

    Args:
        applied_threshold: Threshold derived from source domain.
        y_true: Binary ground truth labels in target domain.
        y_pred: Predicted probabilities in target domain.
        source_metrics: Optional baseline metrics achieved in source domain at this threshold.
        target_optimal_threshold: Optional optimal threshold tuned directly on target domain.

    Returns:
        Structured evaluation dict with transferred metrics, performance drops, and regrets.
    """
    target_metrics = compute_binary_metrics_at_threshold(y_true, y_pred, applied_threshold)

    results: Dict[str, Any] = {
        "applied_threshold": float(applied_threshold),
        "sensitivity": target_metrics["sensitivity"],
        "specificity": target_metrics["specificity"],
        "f1_score": target_metrics["f1_score"],
        "precision": target_metrics["precision"],
        "youden_j": target_metrics["youden_j"],
        "accuracy": target_metrics["accuracy"],
        "balanced_accuracy": target_metrics["balanced_accuracy"],
        "target_metrics": target_metrics,
    }

    # Compare against source domain baseline performance
    if source_metrics is not None:
        src_sens = source_metrics.get("sensitivity", 0.0)
        src_spec = source_metrics.get("specificity", 0.0)
        src_f1 = source_metrics.get("f1_score", 0.0)
        src_youden = source_metrics.get("youden_j", 0.0)

        delta_sens = float(target_metrics["sensitivity"] - src_sens)
        delta_spec = float(target_metrics["specificity"] - src_spec)
        delta_f1 = float(target_metrics["f1_score"] - src_f1)
        delta_youden = float(target_metrics["youden_j"] - src_youden)

        rel_sens_drop = float((abs(delta_sens) / max(src_sens, 1e-4)) * 100.0) if delta_sens < 0 else 0.0
        rel_f1_drop = float((abs(delta_f1) / max(src_f1, 1e-4)) * 100.0) if delta_f1 < 0 else 0.0

        results["delta_sensitivity"] = delta_sens
        results["delta_specificity"] = delta_spec
        results["delta_f1"] = delta_f1
        results["delta_youden"] = delta_youden
        results["relative_sensitivity_drop_percent"] = rel_sens_drop
        results["relative_f1_drop_percent"] = rel_f1_drop

    # Compare against target-optimal re-tuning (upper bound / regret)
    if target_optimal_threshold is not None:
        opt_metrics = compute_binary_metrics_at_threshold(y_true, y_pred, target_optimal_threshold)
        opt_f1 = opt_metrics["f1_score"]
        f1_drop = float(opt_f1 - target_metrics["f1_score"])
        youden_drop = float(opt_metrics["youden_j"] - target_metrics["youden_j"])

        results["target_optimal_threshold"] = float(target_optimal_threshold)
        results["target_optimal_f1"] = opt_f1
        results["target_optimal_youden"] = opt_metrics["youden_j"]
        results["target_optimal_metrics"] = opt_metrics
        results["f1_absolute_drop"] = f1_drop
        results["f1_relative_drop_percent"] = float((f1_drop / max(opt_f1, 1e-4)) * 100.0)
        results["regret_f1"] = max(f1_drop, 0.0)
        results["regret_youden"] = max(youden_drop, 0.0)

    return results


def compute_score_distribution_shift(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
    source_labels: Optional[np.ndarray] = None,
    target_labels: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Quantifies shift in continuous prediction score distributions (median, IQR, and KS-test).

    Mirrors Pan et al. (2019) Table 4 and performs two-sample Kolmogorov-Smirnov tests
    to formally test divergence between source and target prediction score distributions.

    Args:
        source_scores: 1D array of prediction probabilities on source validation cohort.
        target_scores: 1D array of prediction probabilities on target test cohort.
        source_labels: Optional ground truth labels for stratified score analysis.
        target_labels: Optional ground truth labels for stratified score analysis.

    Returns:
        Structured dict with medians, IQRs, mean/std, median shifts, and KS-test statistics.
    """
    s_clean = np.asarray(source_scores, dtype=np.float64).flatten()
    t_clean = np.asarray(target_scores, dtype=np.float64).flatten()

    def get_summary_stats(arr: np.ndarray) -> Dict[str, Any]:
        if len(arr) == 0:
            return {"median": 0.0, "iqr": (0.0, 0.0), "mean": 0.0, "std": 0.0, "q25": 0.0, "q75": 0.0, "count": 0}
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        return {
            "median": float(np.median(arr)),
            "iqr": (q25, q75),
            "q25": q25,
            "q75": q75,
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "count": len(arr),
        }

    s_stats = get_summary_stats(s_clean)
    t_stats = get_summary_stats(t_clean)

    # Two-sample Kolmogorov-Smirnov test for distribution divergence
    if len(s_clean) > 0 and len(t_clean) > 0:
        ks_res = scipy.stats.ks_2samp(s_clean, t_clean)
        ks_stat = float(ks_res.statistic)
        ks_pval = float(ks_res.pvalue)
    else:
        ks_stat = 0.0
        ks_pval = 1.0

    shift_results: Dict[str, Any] = {
        "source": s_stats,
        "target": t_stats,
        "median_difference": float(t_stats["median"] - s_stats["median"]),
        "mean_difference": float(t_stats["mean"] - s_stats["mean"]),
        "ks_test": {
            "statistic": ks_stat,
            "pvalue": ks_pval,
            "reject_null_005": bool(ks_pval < 0.05),
        },
    }

    # Stratified analysis by positive and negative ground truth (if labels provided)
    if source_labels is not None and target_labels is not None:
        s_lbl = np.asarray(source_labels).flatten()
        t_lbl = np.asarray(target_labels).flatten()

        s_pos = s_clean[s_lbl == 1]
        t_pos = t_clean[t_lbl == 1]
        s_neg = s_clean[s_lbl == 0]
        t_neg = t_clean[t_lbl == 0]

        pos_s_stats = get_summary_stats(s_pos)
        pos_t_stats = get_summary_stats(t_pos)
        neg_s_stats = get_summary_stats(s_neg)
        neg_t_stats = get_summary_stats(t_neg)

        ks_pos_stat, ks_pos_pval = (
            (float(scipy.stats.ks_2samp(s_pos, t_pos).statistic), float(scipy.stats.ks_2samp(s_pos, t_pos).pvalue))
            if len(s_pos) > 0 and len(t_pos) > 0
            else (0.0, 1.0)
        )
        ks_neg_stat, ks_neg_pval = (
            (float(scipy.stats.ks_2samp(s_neg, t_neg).statistic), float(scipy.stats.ks_2samp(s_neg, t_neg).pvalue))
            if len(s_neg) > 0 and len(t_neg) > 0
            else (0.0, 1.0)
        )

        shift_results["stratified"] = {
            "positives": {
                "source": pos_s_stats,
                "target": pos_t_stats,
                "median_difference": float(pos_t_stats["median"] - pos_s_stats["median"]),
                "ks_statistic": ks_pos_stat,
                "ks_pvalue": ks_pos_pval,
            },
            "negatives": {
                "source": neg_s_stats,
                "target": neg_t_stats,
                "median_difference": float(neg_t_stats["median"] - neg_s_stats["median"]),
                "ks_statistic": ks_neg_stat,
                "ks_pvalue": ks_neg_pval,
            },
        }

    return shift_results


def evaluate_threshold_transfer_suite(
    source_val_labels: np.ndarray,
    source_val_probs: np.ndarray,
    target_test_labels: np.ndarray,
    target_test_probs: np.ndarray,
    findings: Optional[List[str]] = None,
    shift_direction: str = "nih_to_chexpert",
) -> Dict[str, Any]:
    """Orchestrates comprehensive cross-dataset threshold transfer and score shift analysis.

    Evaluates:
      1. Source-optimal thresholds under Youden's J, Max F1, and Fixed 90% Sensitivity.
      2. Threshold transfer drops (ΔSens, ΔSpec, ΔF1, regret vs target-optimal).
      3. Continuous score distribution shifts (median, IQR, KS-test, stratified by label).
      4. Structured Markdown comparison tables for literature reporting (Pan et al. 2019 Table 4).

    Args:
        source_val_labels: Binary labels on source validation cohort (N_src, C).
        source_val_probs: Model probabilities on source validation cohort (N_src, C).
        target_test_labels: Binary labels on target test cohort (N_tgt, C).
        target_test_probs: Model probabilities on target test cohort (N_tgt, C).
        findings: List of pathology finding names (default: SHARED_CLASSES).
        shift_direction: Identifier string (e.g. 'nih_to_chexpert' or 'chexpert_to_nih').

    Returns:
        Structured dict with per-class and macro results, headline Pneumonia, and Markdown tables.
    """
    classes = findings or SHARED_CLASSES
    num_classes = len(classes)

    per_class_results: Dict[str, Any] = {}

    thresh_table_rows = [
        f"### Threshold Transfer Failure: Youden's J & Sens@90% ({shift_direction.upper()})",
        "",
        "| Finding | Criterion | Source Thresh | Source Sens | Target Sens | Δ Sens | Source Spec | Target Spec | Δ Spec | Transferred F1 | Target-Opt F1 | Regret F1 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    score_table_rows = [
        f"### Prediction Score Distribution Shift & KS Tests ({shift_direction.upper()})",
        "",
        "| Finding | Source Median [IQR] | Target Median [IQR] | Δ Median | Positives Δ Median | Negatives Δ Median | KS Stat | p-value |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    macro_drops = {
        "youden": {"delta_sens": [], "delta_spec": [], "delta_f1": [], "regret_f1": []},
        "sens90": {"delta_sens": [], "delta_spec": [], "delta_f1": [], "regret_f1": []},
        "max_f1": {"delta_sens": [], "delta_spec": [], "delta_f1": [], "regret_f1": []},
    }

    for idx, f in enumerate(classes):
        s_yt = source_val_labels[:, idx] if source_val_labels.ndim > 1 else source_val_labels
        s_yp = source_val_probs[:, idx] if source_val_probs.ndim > 1 else source_val_probs

        t_yt = target_test_labels[:, idx] if target_test_labels.ndim > 1 else target_test_labels
        t_yp = target_test_probs[:, idx] if target_test_probs.ndim > 1 else target_test_probs

        # 1. Source-optimal thresholds
        src_opts = compute_optimal_thresholds(s_yt, s_yp, target_sensitivity=0.90)

        # 2. Target-optimal thresholds (for regret calculation)
        tgt_opts = compute_optimal_thresholds(t_yt, t_yp, target_sensitivity=0.90)

        # 3. Evaluate transfers for all 3 criteria
        transfer_youden = evaluate_threshold_transfer(
            applied_threshold=src_opts["youden_threshold"],
            y_true=t_yt,
            y_pred=t_yp,
            source_metrics=src_opts["youden"]["metrics"],
            target_optimal_threshold=tgt_opts["youden_threshold"],
        )

        transfer_sens90 = evaluate_threshold_transfer(
            applied_threshold=src_opts["sens90_threshold"],
            y_true=t_yt,
            y_pred=t_yp,
            source_metrics=src_opts["fixed_sensitivity"]["metrics"],
            target_optimal_threshold=tgt_opts["sens90_threshold"],
        )

        transfer_max_f1 = evaluate_threshold_transfer(
            applied_threshold=src_opts["f1_threshold"],
            y_true=t_yt,
            y_pred=t_yp,
            source_metrics=src_opts["max_f1"]["metrics"],
            target_optimal_threshold=tgt_opts["f1_threshold"],
        )

        # Track macro drops
        for crit_key, transfer_res in [
            ("youden", transfer_youden),
            ("sens90", transfer_sens90),
            ("max_f1", transfer_max_f1),
        ]:
            macro_drops[crit_key]["delta_sens"].append(transfer_res.get("delta_sensitivity", 0.0))
            macro_drops[crit_key]["delta_spec"].append(transfer_res.get("delta_specificity", 0.0))
            macro_drops[crit_key]["delta_f1"].append(transfer_res.get("delta_f1", 0.0))
            macro_drops[crit_key]["regret_f1"].append(transfer_res.get("regret_f1", 0.0))

        # 4. Continuous score distribution shift & KS test
        score_shift = compute_score_distribution_shift(
            source_scores=s_yp,
            target_scores=t_yp,
            source_labels=s_yt,
            target_labels=t_yt,
        )

        per_class_results[f] = {
            "source_optimal_thresholds": src_opts,
            "target_optimal_thresholds": tgt_opts,
            "transfers": {
                "youden": transfer_youden,
                "sens90": transfer_sens90,
                "max_f1": transfer_max_f1,
            },
            "score_distribution_shift": score_shift,
        }

        # Build table rows (Youden and Sens90)
        s_y_m = src_opts["youden"]["metrics"]
        s_s90_m = src_opts["fixed_sensitivity"]["metrics"]

        thresh_table_rows.append(
            f"| **{f}** | Youden J | {src_opts['youden_threshold']:.3f} | "
            f"{s_y_m['sensitivity']:.3f} | {transfer_youden['sensitivity']:.3f} | "
            f"{transfer_youden.get('delta_sensitivity', 0.0):+.3f} | "
            f"{s_y_m['specificity']:.3f} | {transfer_youden['specificity']:.3f} | "
            f"{transfer_youden.get('delta_specificity', 0.0):+.3f} | "
            f"{transfer_youden['f1_score']:.3f} | {transfer_youden.get('target_optimal_f1', 0.0):.3f} | "
            f"{transfer_youden.get('regret_f1', 0.0):.3f} |"
        )
        thresh_table_rows.append(
            f"| | Sens@90% | {src_opts['sens90_threshold']:.3f} | "
            f"{s_s90_m['sensitivity']:.3f} | {transfer_sens90['sensitivity']:.3f} | "
            f"{transfer_sens90.get('delta_sensitivity', 0.0):+.3f} | "
            f"{s_s90_m['specificity']:.3f} | {transfer_sens90['specificity']:.3f} | "
            f"{transfer_sens90.get('delta_specificity', 0.0):+.3f} | "
            f"{transfer_sens90['f1_score']:.3f} | {transfer_sens90.get('target_optimal_f1', 0.0):.3f} | "
            f"{transfer_sens90.get('regret_f1', 0.0):.3f} |"
        )

        # Build score shift table row
        s_med = score_shift["source"]["median"]
        s_q25, s_q75 = score_shift["source"]["iqr"]
        t_med = score_shift["target"]["median"]
        t_q25, t_q75 = score_shift["target"]["iqr"]
        d_med = score_shift["median_difference"]
        pos_d = score_shift.get("stratified", {}).get("positives", {}).get("median_difference", 0.0)
        neg_d = score_shift.get("stratified", {}).get("negatives", {}).get("median_difference", 0.0)
        ks_stat = score_shift["ks_test"]["statistic"]
        ks_pval = score_shift["ks_test"]["pvalue"]
        p_str = f"{ks_pval:.2e}" if ks_pval < 0.001 else f"{ks_pval:.4f}"

        score_table_rows.append(
            f"| **{f}** | {s_med:.3f} [{s_q25:.3f}, {s_q75:.3f}] | "
            f"{t_med:.3f} [{t_q25:.3f}, {t_q75:.3f}] | {d_med:+.3f} | "
            f"{pos_d:+.3f} | {neg_d:+.3f} | {ks_stat:.3f} | {p_str} |"
        )

    # Compute macro summary
    macro_summary = {}
    for crit in ["youden", "sens90", "max_f1"]:
        macro_summary[crit] = {
            "mean_delta_sensitivity": float(np.mean(macro_drops[crit]["delta_sens"])),
            "mean_delta_specificity": float(np.mean(macro_drops[crit]["delta_spec"])),
            "mean_delta_f1": float(np.mean(macro_drops[crit]["delta_f1"])),
            "mean_regret_f1": float(np.mean(macro_drops[crit]["regret_f1"])),
        }

    thresh_table_rows.append(
        f"| **Macro Average** | Youden J | — | — | — | "
        f"{macro_summary['youden']['mean_delta_sensitivity']:+.3f} | — | — | "
        f"{macro_summary['youden']['mean_delta_specificity']:+.3f} | — | — | "
        f"{macro_summary['youden']['mean_regret_f1']:.3f} |"
    )
    thresh_table_rows.append(
        f"| **Macro Average** | Sens@90% | — | — | — | "
        f"{macro_summary['sens90']['mean_delta_sensitivity']:+.3f} | — | — | "
        f"{macro_summary['sens90']['mean_delta_specificity']:+.3f} | — | — | "
        f"{macro_summary['sens90']['mean_regret_f1']:.3f} |"
    )

    return {
        "shift_direction": shift_direction,
        "per_class": per_class_results,
        "headline_pneumonia": per_class_results.get(HEADLINE_CLASS, {}),
        "macro_metrics": macro_summary,
        "markdown_tables": {
            "threshold_transfer": "\n".join(thresh_table_rows),
            "score_shift": "\n".join(score_table_rows),
        },
    }
