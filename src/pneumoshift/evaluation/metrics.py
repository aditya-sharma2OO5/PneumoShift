"""Discrimination performance metrics: AUROC, AUPRC, fixed-operating points, and gaps."""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


def compute_discrimination_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fixed_spec_target: float = 0.90,
    fixed_sens_target: float = 0.90,
) -> Dict[str, float]:
    """Computes comprehensive discrimination metrics for binary classification.

    Args:
        y_true: Ground truth binary labels of shape (N,).
        y_pred: Predicted probabilities of shape (N,).
        fixed_spec_target: Target specificity to compute operating sensitivity.
        fixed_sens_target: Target sensitivity to compute operating specificity.

    Returns:
        Dict containing auroc, auprc, sens_at_spec, spec_at_sens, prevalence.
    """
    valid_mask = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid_mask]
    yp = y_pred[valid_mask]

    n_pos = np.sum(yt == 1.0)
    n_neg = np.sum(yt == 0.0)

    if n_pos == 0 or n_neg == 0:
        return {
            "auroc": float("nan"),
            "auprc": float("nan"),
            f"sens_at_{int(fixed_spec_target*100)}_spec": float("nan"),
            f"spec_at_{int(fixed_sens_target*100)}_sens": float("nan"),
            "prevalence": float(n_pos / len(yt)) if len(yt) > 0 else 0.0,
        }

    auroc = float(roc_auc_score(yt, yp))
    auprc = float(average_precision_score(yt, yp))

    fpr, tpr, roc_thresholds = roc_curve(yt, yp)
    specificity = 1.0 - fpr

    # Sensitivity at >= target specificity
    spec_eligible = np.where(specificity >= fixed_spec_target)[0]
    sens_at_spec = float(np.max(tpr[spec_eligible])) if len(spec_eligible) > 0 else 0.0

    # Specificity at >= target sensitivity
    sens_eligible = np.where(tpr >= fixed_sens_target)[0]
    spec_at_sens = float(np.max(specificity[sens_eligible])) if len(sens_eligible) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        f"sens_at_{int(fixed_spec_target*100)}_spec": sens_at_spec,
        f"spec_at_{int(fixed_sens_target*100)}_sens": spec_at_sens,
        "prevalence": float(n_pos / len(yt)),
    }


def compute_generalization_gap(
    in_domain: Dict[str, float],
    cross_domain: Dict[str, float],
) -> Dict[str, float]:
    """Computes Generalization Gap = in_domain - cross_domain for all shared metrics."""
    gap = {}
    for k, v in in_domain.items():
        if k in cross_domain and not np.isnan(v) and not np.isnan(cross_domain[k]):
            gap[f"gap_{k}"] = float(v - cross_domain[k])
    return gap

