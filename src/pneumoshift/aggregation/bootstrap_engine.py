"""Patient-clustered non-parametric bootstrap uncertainty engine for Phase 6.

Provides high-performance, vectorized patient-level clustered bootstrapping,
paired and independent two-sample difference testing, and standard error estimation
across discrimination, calibration, operating-point, and explainability metrics.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score, roc_curve

from pneumoshift.aggregation.config_schemas import BootstrapCIResult
from pneumoshift.evaluation.calibration import compute_ece
from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.aggregation.bootstrap_engine")


def _metric_sens_at_90_spec(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes sensitivity at fixed >= 90% specificity."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    spec = 1.0 - fpr
    eligible = np.where(spec >= 0.90)[0]
    return float(np.max(tpr[eligible])) if len(eligible) > 0 else 0.0


def _metric_spec_at_90_sens(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes specificity at fixed >= 90% sensitivity."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    spec = 1.0 - fpr
    eligible = np.where(tpr >= 0.90)[0]
    return float(np.max(spec[eligible])) if len(eligible) > 0 else 0.0


def _metric_ece_uniform(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes 15-bin uniform Expected Calibration Error."""
    ece, _, _ = compute_ece(y_true, y_pred, n_bins=15, strategy="uniform")
    return float(ece)


def _metric_ece_adaptive(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes 15-bin adaptive Expected Calibration Error."""
    ece, _, _ = compute_ece(y_true, y_pred, n_bins=15, strategy="adaptive")
    return float(ece)


def _metric_brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes Brier score (mean squared error of probability predictions)."""
    yp = np.clip(y_pred, 0.0, 1.0)
    return float(np.mean((yp - y_true) ** 2))


def _metric_nll(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes negative log-likelihood (binary cross-entropy loss)."""
    yp = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return float(log_loss(y_true, yp))


def _metric_f1_score(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = 0.5) -> float:
    """Computes F1 score at given decision threshold."""
    bin_pred = (y_pred >= threshold).astype(float)
    tp = np.sum((bin_pred == 1.0) & (y_true == 1.0))
    fp = np.sum((bin_pred == 1.0) & (y_true == 0.0))
    fn = np.sum((bin_pred == 0.0) & (y_true == 1.0))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2.0 * (precision * recall) / (precision + recall))


STANDARD_METRICS: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    "auroc": roc_auc_score,
    "auprc": average_precision_score,
    "sens_at_90_spec": _metric_sens_at_90_spec,
    "spec_at_90_sens": _metric_spec_at_90_sens,
    "ece_uniform": _metric_ece_uniform,
    "ece_adaptive": _metric_ece_adaptive,
    "brier_score": _metric_brier_score,
    "nll": _metric_nll,
    "f1_score": _metric_f1_score,
}


class PatientClusteredBootstrap:
    """High-performance patient-clustered bootstrap estimator."""

    def __init__(
        self,
        n_resamples: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> None:
        self.n_resamples = n_resamples
        self.confidence_level = confidence_level
        self.seed = seed

    def compute_ci(
        self,
        patient_ids: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "auroc",
    ) -> BootstrapCIResult:
        """Computes clustered bootstrap 95% confidence intervals for a metric.

        Args:
            patient_ids: 1D array of patient cluster identifiers.
            y_true: 1D array of ground truth binary labels.
            y_pred: 1D array of continuous predicted probabilities.
            metric: Metric name or callable function.

        Returns:
            BootstrapCIResult containing point estimate, CI bounds, and standard error.
        """
        metric_fn = STANDARD_METRICS.get(metric, metric) if isinstance(metric, str) else metric
        if not callable(metric_fn):
            raise ValueError(f"Metric must be callable or one of {list(STANDARD_METRICS.keys())}")

        p_ids = np.asarray(patient_ids)
        yt = np.asarray(y_true, dtype=float)
        yp = np.asarray(y_pred, dtype=float)

        unique_patients = np.unique(p_ids)
        point_estimate = float(metric_fn(yt, yp))

        # Build fast index lookup map
        patient_indices = {p: np.where(p_ids == p)[0] for p in unique_patients}
        rng = np.random.RandomState(self.seed)

        estimates: List[float] = []
        n_patients = len(unique_patients)

        for _ in range(self.n_resamples):
            sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
            idx = np.concatenate([patient_indices[p] for p in sampled_patients])

            sub_yt = yt[idx]
            sub_yp = yp[idx]

            # Require both positive and negative cases for classification metrics
            if len(np.unique(sub_yt)) < 2:
                continue

            try:
                score = float(metric_fn(sub_yt, sub_yp))
                if not np.isnan(score):
                    estimates.append(score)
            except Exception:
                continue

        if len(estimates) < 10:
            logger.warning("Fewer than 10 valid bootstrap iterations; returning point estimate as bounds.")
            return BootstrapCIResult(
                point_estimate=point_estimate,
                ci_lower=point_estimate,
                ci_upper=point_estimate,
                standard_error=0.0,
                n_resamples=len(estimates),
                confidence_level=self.confidence_level,
            )

        alpha = 1.0 - self.confidence_level
        ci_lower = float(np.percentile(estimates, 100 * (alpha / 2.0)))
        ci_upper = float(np.percentile(estimates, 100 * (1.0 - alpha / 2.0)))
        se = float(np.std(estimates, ddof=1))

        # Enforce mathematical bounding
        ci_lower = min(ci_lower, point_estimate)
        ci_upper = max(ci_upper, point_estimate)

        return BootstrapCIResult(
            point_estimate=point_estimate,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            standard_error=se,
            n_resamples=len(estimates),
            confidence_level=self.confidence_level,
        )

    def compute_paired_difference(
        self,
        patient_ids: np.ndarray,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
        metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "auroc",
    ) -> Dict[str, Any]:
        """Calculates paired bootstrap differences: Delta = Metric(A) - Metric(B) on the same cohort.

        Returns:
            Dict containing mean_diff, ci_lower, ci_upper, standard_error, p_value, is_significant.
        """
        metric_fn = STANDARD_METRICS.get(metric, metric) if isinstance(metric, str) else metric
        p_ids = np.asarray(patient_ids)
        yt = np.asarray(y_true, dtype=float)
        yp_a = np.asarray(y_pred_a, dtype=float)
        yp_b = np.asarray(y_pred_b, dtype=float)

        unique_patients = np.unique(p_ids)
        orig_diff = float(metric_fn(yt, yp_a) - metric_fn(yt, yp_b))

        patient_indices = {p: np.where(p_ids == p)[0] for p in unique_patients}
        rng = np.random.RandomState(self.seed)

        diffs: List[float] = []
        n_patients = len(unique_patients)

        for _ in range(self.n_resamples):
            sampled_patients = rng.choice(unique_patients, size=n_patients, replace=True)
            idx = np.concatenate([patient_indices[p] for p in sampled_patients])

            sub_yt = yt[idx]
            if len(np.unique(sub_yt)) < 2:
                continue

            try:
                score_a = float(metric_fn(sub_yt, yp_a[idx]))
                score_b = float(metric_fn(sub_yt, yp_b[idx]))
                if not (np.isnan(score_a) or np.isnan(score_b)):
                    diffs.append(score_a - score_b)
            except Exception:
                continue

        if not diffs:
            return {
                "mean_diff": orig_diff,
                "ci_lower": orig_diff,
                "ci_upper": orig_diff,
                "standard_error": 0.0,
                "p_value": 1.0,
                "is_significant": False,
            }

        alpha = 1.0 - self.confidence_level
        ci_lower = float(np.percentile(diffs, 100 * (alpha / 2.0)))
        ci_upper = float(np.percentile(diffs, 100 * (1.0 - alpha / 2.0)))
        se = float(np.std(diffs, ddof=1))

        # Two-sided empirical p-value for H0: difference == 0
        diff_arr = np.array(diffs)
        p_val = float(2.0 * min(np.mean(diff_arr <= 0.0), np.mean(diff_arr >= 0.0)))
        p_val = min(max(p_val, 1e-4), 1.0)

        is_sig = bool(ci_lower > 0.0 or ci_upper < 0.0)

        return {
            "mean_diff": orig_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "standard_error": se,
            "p_value": p_val,
            "is_significant": is_sig,
        }

    def compute_two_sample_difference(
        self,
        patient_ids_a: np.ndarray,
        y_true_a: np.ndarray,
        y_pred_a: np.ndarray,
        patient_ids_b: np.ndarray,
        y_true_b: np.ndarray,
        y_pred_b: np.ndarray,
        metric: Union[str, Callable[[np.ndarray, np.ndarray], float]] = "auroc",
    ) -> Dict[str, Any]:
        """Calculates independent two-sample clustered bootstrap difference: Delta = Metric(A) - Metric(B)."""
        metric_fn = STANDARD_METRICS.get(metric, metric) if isinstance(metric, str) else metric
        p_ids_a = np.asarray(patient_ids_a)
        yt_a = np.asarray(y_true_a, dtype=float)
        yp_a = np.asarray(y_pred_a, dtype=float)

        p_ids_b = np.asarray(patient_ids_b)
        yt_b = np.asarray(y_true_b, dtype=float)
        yp_b = np.asarray(y_pred_b, dtype=float)

        orig_diff = float(metric_fn(yt_a, yp_a) - metric_fn(yt_b, yp_b))

        unq_a = np.unique(p_ids_a)
        unq_b = np.unique(p_ids_b)
        idx_a = {p: np.where(p_ids_a == p)[0] for p in unq_a}
        idx_b = {p: np.where(p_ids_b == p)[0] for p in unq_b}

        rng = np.random.RandomState(self.seed)
        diffs: List[float] = []

        for _ in range(self.n_resamples):
            samp_a = rng.choice(unq_a, size=len(unq_a), replace=True)
            samp_b = rng.choice(unq_b, size=len(unq_b), replace=True)

            sel_a = np.concatenate([idx_a[p] for p in samp_a])
            sel_b = np.concatenate([idx_b[p] for p in samp_b])

            sub_yta, sub_ypa = yt_a[sel_a], yp_a[sel_a]
            sub_ytb, sub_ypb = yt_b[sel_b], yp_b[sel_b]

            if len(np.unique(sub_yta)) < 2 or len(np.unique(sub_ytb)) < 2:
                continue

            try:
                ma = float(metric_fn(sub_yta, sub_ypa))
                mb = float(metric_fn(sub_ytb, sub_ypb))
                if not (np.isnan(ma) or np.isnan(mb)):
                    diffs.append(ma - mb)
            except Exception:
                continue

        if not diffs:
            return {
                "mean_diff": orig_diff,
                "ci_lower": orig_diff,
                "ci_upper": orig_diff,
                "standard_error": 0.0,
                "p_value": 1.0,
                "is_significant": False,
            }

        alpha = 1.0 - self.confidence_level
        ci_lower = float(np.percentile(diffs, 100 * (alpha / 2.0)))
        ci_upper = float(np.percentile(diffs, 100 * (1.0 - alpha / 2.0)))
        se = float(np.std(diffs, ddof=1))

        diff_arr = np.array(diffs)
        p_val = float(2.0 * min(np.mean(diff_arr <= 0.0), np.mean(diff_arr >= 0.0)))
        p_val = min(max(p_val, 1e-4), 1.0)

        is_sig = bool(ci_lower > 0.0 or ci_upper < 0.0)

        return {
            "mean_diff": orig_diff,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "standard_error": se,
            "p_value": p_val,
            "is_significant": is_sig,
        }


def compute_cohort_bootstrap_summary(
    patient_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: Optional[List[str]] = None,
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Evaluates an array of metrics with patient-clustered 95% confidence intervals.

    Args:
        patient_ids: Patient identifiers for cluster resampling.
        y_true: Binary ground truth labels.
        y_pred: Continuous predicted probabilities.
        metrics: List of metric names to evaluate (default: AUROC, AUPRC, Sens@90Spec, ECE, Brier).
        n_resamples: Number of bootstrap iterations.
        confidence_level: Desired coverage probability.
        seed: Random seed.

    Returns:
        Dictionary mapping metric name to structured CI dictionary.
    """
    if metrics is None:
        metrics = ["auroc", "auprc", "sens_at_90_spec", "ece_uniform", "brier_score"]

    engine = PatientClusteredBootstrap(
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        seed=seed,
    )

    summary: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        try:
            ci_res = engine.compute_ci(patient_ids, y_true, y_pred, metric=m)
            summary[m] = ci_res.to_dict()
            summary[m]["formatted"] = ci_res.format_ci(precision=3)
        except Exception as e:
            logger.error("Failed to compute bootstrap CI for metric %s: %s", m, e)

    return summary


def save_bootstrap_report(
    bootstrap_results: Dict[str, Any],
    output_path: Union[str, Path] = "results/tables/master/bootstrap_summary.json",
) -> Path:
    """Saves bootstrap confidence interval results to disk as JSON."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(bootstrap_results, out_p)
    logger.info("Saved bootstrap summary to: %s", out_p)
    return out_p

