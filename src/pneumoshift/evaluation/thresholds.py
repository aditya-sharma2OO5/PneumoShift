"""Decision threshold selection, transferability drop analysis, and score distribution shifts."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve, roc_curve


def compute_optimal_thresholds(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes source-optimal classification thresholds based on distinct clinical criteria.

    - Youden's J index: argmax (TPR - FPR) = argmax (Sensitivity + Specificity - 1)
    - Max F1 score: argmax (2 * P * R / (P + R))
    - Fixed 90% Sensitivity: lowest threshold achieving >= 90% TPR
    """
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    yp = y_pred[valid]

    fpr, tpr, roc_thresh = roc_curve(yt, yp)
    youden_idx = np.argmax(tpr - fpr)
    youden_threshold = float(roc_thresh[youden_idx])

    # F1 score threshold
    precisions, recalls, pr_thresh = precision_recall_curve(yt, yp)
    f1_scores = 2 * (precisions * recalls) / np.maximum(precisions + recalls, 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1]) if len(pr_thresh) > 0 else 0
    f1_threshold = float(pr_thresh[best_f1_idx]) if len(pr_thresh) > 0 else 0.5

    # 90% Sensitivity threshold
    eligible_sens = np.where(tpr >= 0.90)[0]
    sens90_threshold = float(np.max(roc_thresh[eligible_sens])) if len(eligible_sens) > 0 else 0.5

    return {
        "youden_threshold": youden_threshold,
        "f1_threshold": f1_threshold,
        "sens90_threshold": sens90_threshold,
    }


def evaluate_threshold_transfer(
    applied_threshold: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_optimal_threshold: Optional[float] = None,
) -> Dict[str, float]:
    """Evaluates cross-dataset threshold transferability and performance degradation.

    Mirrors the evaluation methodology in Pan et al. (2019).
    """
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    yp = y_pred[valid]

    bin_pred = (yp >= applied_threshold).astype(int)
    tp = np.sum((bin_pred == 1) & (yt == 1))
    fp = np.sum((bin_pred == 1) & (yt == 0))
    tn = np.sum((bin_pred == 0) & (yt == 0))
    fn = np.sum((bin_pred == 0) & (yt == 1))

    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    f1 = float(f1_score(yt, bin_pred, zero_division=0))

    results = {
        "applied_threshold": applied_threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1_score": f1,
    }

    if target_optimal_threshold is not None:
        target_pred = (yp >= target_optimal_threshold).astype(int)
        opt_f1 = float(f1_score(yt, target_pred, zero_division=0))
        f1_drop = opt_f1 - f1
        results["target_optimal_threshold"] = target_optimal_threshold
        results["target_optimal_f1"] = opt_f1
        results["f1_absolute_drop"] = f1_drop
        results["f1_relative_drop_percent"] = float((f1_drop / max(opt_f1, 1e-4)) * 100.0)

    return results


def compute_score_distribution_shift(
    source_scores: np.ndarray,
    target_scores: np.ndarray,
) -> Dict[str, Any]:
    """Quantifies shift in continuous prediction score distributions (median and IQR).

    Mirrors Pan et al. (2019) Table 4.
    """
    s_med = float(np.median(source_scores))
    s_q25 = float(np.percentile(source_scores, 25))
    s_q75 = float(np.percentile(source_scores, 75))

    t_med = float(np.median(target_scores))
    t_q25 = float(np.percentile(target_scores, 25))
    t_q75 = float(np.percentile(target_scores, 75))

    return {
        "source": {"median": s_med, "iqr": (s_q25, s_q75)},
        "target": {"median": t_med, "iqr": (t_q25, t_q75)},
        "median_difference": float(t_med - s_med),
    }

