"""Probability calibration evaluation and post-hoc temperature scaling."""

from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
import scipy.optimize
import torch
import torch.nn as nn


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
    """Post-hoc probability calibration via learned scalar temperature."""

    def __init__(self) -> None:
        self.temperature: float = 1.0

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> "TemperatureScaling":
        """Fits temperature parameter T > 0 on validation set logits using NLL minimization."""
        valid = (labels == 0.0) | (labels == 1.0)
        z = logits[valid]
        y = labels[valid]

        def nll_loss(t: np.ndarray) -> float:
            scaled_z = z / max(float(t[0]), 1e-4)
            # Binary cross entropy with logits
            max_z = np.maximum(scaled_z, 0)
            log_p = -max_z - np.log(np.exp(-max_z) + np.exp(scaled_z - max_z))
            log_1_p = scaled_z - max_z - np.log(np.exp(-max_z) + np.exp(scaled_z - max_z))
            loss = -(y * log_p + (1.0 - y) * log_1_p)
            return float(np.mean(loss))

        res = scipy.optimize.minimize(nll_loss, x0=np.array([1.0]), bounds=[(0.01, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self

    def calibrate_probs(self, logits: np.ndarray) -> np.ndarray:
        """Scales logits and returns calibrated probabilities."""
        scaled_logits = logits / max(self.temperature, 1e-4)
        return 1.0 / (1.0 + np.exp(-scaled_logits))

