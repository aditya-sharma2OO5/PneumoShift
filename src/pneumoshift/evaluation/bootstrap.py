"""Patient-level cluster bootstrapping, paired difference testing, and FDR correction."""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def patient_bootstrap_ci(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Computes patient-level clustered non-parametric bootstrap confidence intervals.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    unique_patients = np.unique(patient_ids)
    point_estimate = float(metric_fn(y_true, y_pred))

    patient_to_indices: Dict[Any, np.ndarray] = {}
    for p in unique_patients:
        patient_to_indices[p] = np.where(patient_ids == p)[0]

    rng = np.random.RandomState(seed)
    bootstrap_estimates: List[float] = []

    for _ in range(n_resamples):
        sampled_patients = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        sample_indices = np.concatenate([patient_to_indices[p] for p in sampled_patients])

        sample_y_true = y_true[sample_indices]
        sample_y_pred = y_pred[sample_indices]

        if len(np.unique(sample_y_true)) < 2:
            continue

        try:
            score = float(metric_fn(sample_y_true, sample_y_pred))
            if not np.isnan(score):
                bootstrap_estimates.append(score)
        except Exception:
            continue

    if len(bootstrap_estimates) < 10:
        return point_estimate, point_estimate, point_estimate

    alpha = 1.0 - confidence_level
    ci_lower = float(np.percentile(bootstrap_estimates, 100 * (alpha / 2.0)))
    ci_upper = float(np.percentile(bootstrap_estimates, 100 * (1.0 - alpha / 2.0)))

    return point_estimate, ci_lower, ci_upper


def paired_bootstrap_auroc_diff(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, float]:
    """Calculates paired bootstrap differences in AUROC between two models on identical cases."""
    unique_patients = np.unique(patient_ids)
    orig_diff = float(roc_auc_score(y_true, y_pred_a) - roc_auc_score(y_true, y_pred_b))

    patient_to_indices = {p: np.where(patient_ids == p)[0] for p in unique_patients}
    rng = np.random.RandomState(seed)
    diffs = []

    for _ in range(n_resamples):
        sampled_p = rng.choice(unique_patients, size=len(unique_patients), replace=True)
        idx = np.concatenate([patient_to_indices[p] for p in sampled_p])
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        auc_a = roc_auc_score(yt, y_pred_a[idx])
        auc_b = roc_auc_score(yt, y_pred_b[idx])
        diffs.append(auc_a - auc_b)

    ci_lower = float(np.percentile(diffs, 2.5)) if diffs else orig_diff
    ci_upper = float(np.percentile(diffs, 97.5)) if diffs else orig_diff

    # Two-sided empirical p-value for hypothesis diff == 0
    if diffs:
        p_val = float(2.0 * min(np.mean(np.array(diffs) <= 0), np.mean(np.array(diffs) >= 0)))
        p_val = min(p_val, 1.0)
    else:
        p_val = 1.0

    return {
        "mean_diff": orig_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val,
        "significant": bool(ci_lower > 0 or ci_upper < 0),
    }


def benjamini_hochberg_fdr(p_values: List[float], q: float = 0.05) -> Tuple[List[float], List[bool]]:
    """Applies Benjamini-Hochberg procedure to control False Discovery Rate."""
    p_vals = np.array(p_values)
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]

    adjusted = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        adj = sorted_p[i] * n / (i + 1)
        cum_min = min(cum_min, adj)
        adjusted[i] = cum_min

    adjusted_p = np.zeros(n)
    adjusted_p[sorted_idx] = np.clip(adjusted, 0.0, 1.0)
    significant = adjusted_p <= q

    return adjusted_p.tolist(), significant.tolist()

