"""Publication-grade visualizations: ROC, PR, calibration, and saliency heatmaps."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve
from pneumoshift.evaluation.calibration import compute_ece


def plot_roc_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "Receiver Operating Characteristic (ROC)",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plots multi-model ROC curves with AUROC values in legend."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, (y_true, y_pred, auroc) in curves_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auroc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)

    return fig


def plot_pr_curves(
    curves_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    title: str = "Precision-Recall Curve",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plots multi-model Precision-Recall curves with AUPRC values in legend."""
    fig, ax = plt.subplots(figsize=(7, 6))

    for label, (y_true, y_pred, auprc) in curves_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ax.plot(recall, precision, lw=2, label=f"{label} (AUPRC = {auprc:.3f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)

    return fig


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 15,
    title: str = "Reliability Diagram",
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plots calibration reliability bar chart with perfect calibration diagonal."""
    ece, mce, details = compute_ece(y_true, y_pred, n_bins=n_bins, strategy="uniform")

    bin_accs = details["bin_accs"]
    bin_confs = details["bin_confs"]
    bin_counts = details["bin_counts"]
    bin_edges = details["bin_edges"]

    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths = bin_edges[1:] - bin_edges[:-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar(centers, bin_accs, width=widths, edgecolor="black", alpha=0.6, label="Empirical Accuracy")
    ax.plot([0, 1], [0, 1], "r--", lw=2, label="Perfect Calibration")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("Predicted Confidence")
    ax.set_ylabel("Empirical Accuracy / Fraction of Positives")
    ax.set_title(f"{title} (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300)

    return fig


def plot_saliency_overlay(
    image: np.ndarray,
    saliency_map: np.ndarray,
    alpha: float = 0.5,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Overlays jet colormap Grad-CAM attribution onto grayscale chest X-ray."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    ax.imshow(saliency_map, cmap="jet", alpha=alpha)
    ax.axis("off")
    fig.tight_layout()

    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def plot_saliency_gallery(
    items: List[Dict[str, Any]],
    cols: int = 4,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plots a multi-panel gallery of Grad-CAM explanations."""
    n = len(items)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.5))
    axes = np.array(axes).reshape(-1)

    for idx, item in enumerate(items):
        ax = axes[idx]
        ax.imshow(item["image"], cmap="gray")
        ax.imshow(item["saliency"], cmap="jet", alpha=0.45)
        ax.set_title(item.get("title", ""), fontsize=9)
        ax.axis("off")

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    if output_path:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig

