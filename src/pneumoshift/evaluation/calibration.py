"""Probability calibration evaluation, post-hoc temperature scaling, and oracle recalibration."""

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.calibration")


def compute_ece(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
    strategy: Literal["uniform", "adaptive"] = "uniform",
) -> Tuple[float, float, Dict[str, np.ndarray]]:
    """Computes Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_pred: Predicted probabilities in [0, 1].
        n_bins: Number of bins (standard = 15).
        strategy: 'uniform' (equal-width) or 'adaptive' (equal-frequency quantile).

    Returns:
        (ece, mce, bin_details_dict)
    """
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    yp = np.clip(y_pred[valid], 1e-7, 1.0 - 1e-7)

    n_samples = len(yt)
    if n_samples == 0:
        return 0.0, 0.0, {}

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(yp, bin_edges[1:-1])
    elif strategy == "adaptive":
        quantiles = np.linspace(0.0, 100.0, n_bins + 1)
        bin_edges = np.percentile(yp, quantiles)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(yp, bin_edges[1:-1])
    else:
        raise ValueError(f"Unknown bin strategy: {strategy}")

    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    mce = 0.0

    actual_bins = len(bin_edges) - 1
    for b in range(actual_bins):
        mask = bin_indices == b
        count = np.sum(mask)
        bin_counts.append(count)
        if count > 0:
            acc = float(np.mean(yt[mask]))
            conf = float(np.mean(yp[mask]))
            bin_accs.append(acc)
            bin_confs.append(conf)
            diff = abs(acc - conf)
            ece += (count / n_samples) * diff
            mce = max(mce, diff)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    details = {
        "bin_accs": np.array(bin_accs),
        "bin_confs": np.array(bin_confs),
        "bin_counts": np.array(bin_counts),
        "bin_edges": bin_edges,
    }
    return float(ece), float(mce), details


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
) -> Dict[str, float]:
    """Calculates ECE (uniform & adaptive), MCE, Brier score, NLL, and mean prevalence."""
    valid = (y_true == 0.0) | (y_true == 1.0)
    yt = y_true[valid]
    yp = np.clip(y_pred[valid], 1e-7, 1.0 - 1e-7)

    if len(yt) == 0:
        return {}

    ece_uniform, mce_uniform, _ = compute_ece(yt, yp, n_bins=n_bins, strategy="uniform")
    ece_adaptive, _, _ = compute_ece(yt, yp, n_bins=n_bins, strategy="adaptive")

    brier_score = float(np.mean((yp - yt) ** 2))
    nll = float(-np.mean(yt * np.log(yp) + (1.0 - yt) * np.log(1.0 - yp)))
    mean_pred = float(np.mean(yp))
    observed_prevalence = float(np.mean(yt))

    return {
        "ece_uniform": ece_uniform,
        "ece_adaptive": ece_adaptive,
        "mce": mce_uniform,
        "brier_score": brier_score,
        "nll": nll,
        "mean_prediction": mean_pred,
        "observed_prevalence": observed_prevalence,
        "prevalence_calibration_ratio": float(mean_pred / max(observed_prevalence, 1e-5)),
    }


class TemperatureScaling:
    """Post-hoc probability calibration via learned scalar or vector temperature."""

    def __init__(self) -> None:
        self.temperature: float = 1.0
        self.temperatures: Optional[np.ndarray] = None

    def _fit_scalar(self, z: np.ndarray, y: np.ndarray) -> float:
        valid = (y == 0.0) | (y == 1.0)
        z_v = z[valid]
        y_v = y[valid]

        if len(y_v) == 0 or len(np.unique(y_v)) < 2:
            return 1.0

        def nll_loss(t: np.ndarray) -> float:
            scaled_z = z_v / max(float(t[0]), 1e-4)
            max_z = np.maximum(scaled_z, 0)
            log_p = -max_z - np.log(np.exp(-max_z) + np.exp(scaled_z - max_z))
            log_1_p = scaled_z - max_z - np.log(np.exp(-max_z) + np.exp(scaled_z - max_z))
            loss = -(y_v * log_p + (1.0 - y_v) * log_1_p)
            return float(np.mean(loss))

        res = scipy.optimize.minimize(nll_loss, x0=np.array([1.0]), bounds=[(0.01, 10.0)], method="L-BFGS-B")
        return float(res.x[0])

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        """Fits temperature parameter T > 0 on validation set logits using NLL minimization.

        If logits is 2D, optimizes per-class temperature vectors for multi-label calibration.
        """
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[1] == 1):
            z = logits.flatten()
            y = labels.flatten()
            self.temperature = self._fit_scalar(z, y)
            self.temperatures = np.array([self.temperature])
        else:
            num_classes = logits.shape[1]
            temps = []
            for c in range(num_classes):
                t_c = self._fit_scalar(logits[:, c], labels[:, c])
                temps.append(t_c)
            self.temperatures = np.array(temps)
            self.temperature = float(np.mean(self.temperatures))

        logger.debug("Fitted temperature scaling: scalar=%.3f", self.temperature)
        return self

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Scales logits and returns calibrated probabilities in [0, 1]."""
        if self.temperatures is not None and logits.ndim == 2 and logits.shape[1] == len(self.temperatures):
            scaled_logits = logits / np.maximum(self.temperatures[None, :], 1e-4)
        else:
            scaled_logits = logits / max(self.temperature, 1e-4)
        scaled_logits = np.clip(scaled_logits, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-scaled_logits))


def compute_reliability_diagram_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
    strategy: Literal["uniform", "adaptive"] = "uniform",
) -> Dict[str, Any]:
    """Generates calibration accuracy vs. confidence binning data for reliability diagrams."""
    ece, mce, details = compute_ece(y_true, y_pred, n_bins=n_bins, strategy=strategy)
    return {
        "ece": ece,
        "mce": mce,
        "bin_accs": details.get("bin_accs", np.array([])).tolist(),
        "bin_confs": details.get("bin_confs", np.array([])).tolist(),
        "bin_counts": details.get("bin_counts", np.array([])).tolist(),
        "bin_edges": details.get("bin_edges", np.array([])).tolist(),
        "strategy": strategy,
        "n_bins": n_bins,
    }


def evaluate_calibration_breakdown(
    source_val_logits: np.ndarray,
    source_val_labels: np.ndarray,
    target_test_logits: np.ndarray,
    target_test_labels: np.ndarray,
    target_cal_logits: Optional[np.ndarray] = None,
    target_cal_labels: Optional[np.ndarray] = None,
    findings: Optional[List[str]] = None,
    n_bins: int = 15,
    shift_direction: str = "nih_to_chexpert",
) -> Dict[str, Any]:
    """Quantifies calibration breakdown across domains, evaluates source temperature scaling,

    and measures recoverable calibration error via oracle recalibration.

    Regimes evaluated:
        1. In-domain baseline (source validation set)
        2. Cross-domain uncalibrated (raw target test predictions)
        3. Cross-domain source-scaled (target test scaled by T_source fitted on source val)
        4. Cross-domain oracle-recalibrated (target test scaled by T_oracle fitted on 10% target_cal)

    Returns:
        Structured dict with per-class and macro calibration metrics, ECE deltas,
        recoverable error percentages, reliability diagram data, and Markdown table.
    """
    classes = findings or SHARED_CLASSES
    num_classes = len(classes)

    # 1. Fit Source Temperature Scaler
    source_scaler = TemperatureScaling()
    source_scaler.fit(source_val_logits, source_val_labels)

    # 2. Fit Oracle Recalibration Scaler on 10% target_cal (if provided)
    oracle_scaler: Optional[TemperatureScaling] = None
    if target_cal_logits is not None and target_cal_labels is not None:
        oracle_scaler = TemperatureScaling()
        oracle_scaler.fit(target_cal_logits, target_cal_labels)

    # 3. Generate probabilities across all regimes
    src_val_probs = 1.0 / (1.0 + np.exp(-np.clip(source_val_logits, -20.0, 20.0)))
    tgt_uncal_probs = 1.0 / (1.0 + np.exp(-np.clip(target_test_logits, -20.0, 20.0)))
    tgt_src_scaled_probs = source_scaler.calibrate_probs(target_test_logits)
    tgt_oracle_probs = (
        oracle_scaler.calibrate_probs(target_test_logits) if oracle_scaler else None
    )

    per_class_results: Dict[str, Dict[str, Any]] = {}
    table_rows = [
        f"### Calibration Degradation & Oracle Recalibration ({shift_direction.upper()})",
        "",
        "| Finding | In-Domain ECE | Target Uncalibrated ECE | Source-Scaled ECE | Oracle-Recalibrated ECE | ECE Reduction (%) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    macro_in_ece = []
    macro_uncal_ece = []
    macro_src_scaled_ece = []
    macro_oracle_ece = []

    for idx, f in enumerate(classes):
        # Extract 1D slices
        s_yt = source_val_labels[:, idx] if source_val_labels.ndim > 1 else source_val_labels
        s_yp = src_val_probs[:, idx] if src_val_probs.ndim > 1 else src_val_probs

        t_yt = target_test_labels[:, idx] if target_test_labels.ndim > 1 else target_test_labels
        t_yp_uncal = tgt_uncal_probs[:, idx] if tgt_uncal_probs.ndim > 1 else tgt_uncal_probs
        t_yp_src = tgt_src_scaled_probs[:, idx] if tgt_src_scaled_probs.ndim > 1 else tgt_src_scaled_probs
        t_yp_ora = (tgt_oracle_probs[:, idx] if tgt_oracle_probs.ndim > 1 else tgt_oracle_probs) if tgt_oracle_probs is not None else None

        # Calibration metrics
        in_cal = compute_calibration_metrics(s_yt, s_yp, n_bins=n_bins)
        uncal = compute_calibration_metrics(t_yt, t_yp_uncal, n_bins=n_bins)
        src_scaled = compute_calibration_metrics(t_yt, t_yp_src, n_bins=n_bins)
        oracle = compute_calibration_metrics(t_yt, t_yp_ora, n_bins=n_bins) if t_yp_ora is not None else {}

        in_ece = in_cal.get("ece_uniform", 0.0)
        uncal_ece = uncal.get("ece_uniform", 0.0)
        src_ece = src_scaled.get("ece_uniform", 0.0)
        ora_ece = oracle.get("ece_uniform", None)

        macro_in_ece.append(in_ece)
        macro_uncal_ece.append(uncal_ece)
        macro_src_scaled_ece.append(src_ece)
        if ora_ece is not None:
            macro_oracle_ece.append(ora_ece)

        # ECE reduction percentage under oracle recalibration
        ece_reduction_pct = None
        if ora_ece is not None and uncal_ece > 0:
            ece_reduction_pct = float(((uncal_ece - ora_ece) / uncal_ece) * 100.0)

        # Reliability diagram data
        rel_diag = {
            "uncalibrated": compute_reliability_diagram_data(t_yt, t_yp_uncal, n_bins=n_bins),
            "source_scaled": compute_reliability_diagram_data(t_yt, t_yp_src, n_bins=n_bins),
        }
        if t_yp_ora is not None:
            rel_diag["oracle_recalibrated"] = compute_reliability_diagram_data(t_yt, t_yp_ora, n_bins=n_bins)

        t_src_c = float(source_scaler.temperatures[idx]) if (source_scaler.temperatures is not None and idx < len(source_scaler.temperatures)) else source_scaler.temperature
        t_ora_c = (
            float(oracle_scaler.temperatures[idx])
            if (oracle_scaler and oracle_scaler.temperatures is not None and idx < len(oracle_scaler.temperatures))
            else (oracle_scaler.temperature if oracle_scaler else None)
        )

        per_class_results[f] = {
            "in_domain": in_cal,
            "target_uncalibrated": uncal,
            "target_source_scaled": src_scaled,
            "target_oracle_recalibrated": oracle,
            "ece_shift_delta": float(uncal_ece - in_ece),
            "ece_reduction_percent": ece_reduction_pct,
            "temperature_source": t_src_c,
            "temperature_oracle": t_ora_c,
            "reliability_diagram": rel_diag,
        }

        ora_str = f"{ora_ece:.4f}" if ora_ece is not None else "—"
        red_str = f"{ece_reduction_pct:+.1f}%" if ece_reduction_pct is not None else "—"
        table_rows.append(
            f"| **{f}** | {in_ece:.4f} | {uncal_ece:.4f} | {src_ece:.4f} | {ora_str} | {red_str} |"
        )

    # Macro averages
    m_in = float(np.mean(macro_in_ece)) if macro_in_ece else 0.0
    m_uncal = float(np.mean(macro_uncal_ece)) if macro_uncal_ece else 0.0
    m_src = float(np.mean(macro_src_scaled_ece)) if macro_src_scaled_ece else 0.0
    m_ora = float(np.mean(macro_oracle_ece)) if macro_oracle_ece else None
    m_red = float(((m_uncal - m_ora) / m_uncal) * 100.0) if (m_ora is not None and m_uncal > 0) else None

    table_rows.append(
        f"| **Macro Average** | {m_in:.4f} | {m_uncal:.4f} | {m_src:.4f} | "
        f"{f'{m_ora:.4f}' if m_ora is not None else '—'} | "
        f"{f'{m_red:+.1f}%' if m_red is not None else '—'} |"
    )

    markdown_report = "\n".join(table_rows)

    return {
        "shift_direction": shift_direction,
        "per_class": per_class_results,
        "headline_pneumonia": per_class_results.get("Pneumonia", {}),
        "macro_metrics": {
            "in_domain_ece": m_in,
            "target_uncalibrated_ece": m_uncal,
            "target_source_scaled_ece": m_src,
            "target_oracle_recalibrated_ece": m_ora,
            "macro_ece_reduction_percent": m_red,
            "source_mean_temperature": source_scaler.temperature,
            "oracle_mean_temperature": oracle_scaler.temperature if oracle_scaler else None,
        },
        "markdown_table": markdown_report,
    }

