"""Anatomical plausibility via lung field segmentation and attribution mass ratios."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
import torch


def generate_anatomical_lung_mask(
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Generates standard bilateral anatomical lung field mask heuristic for dry-run/smoke testing."""
    h, w = image_shape
    y, x = np.ogrid[:h, :w]
    # Elliptical anatomical approximations for left and right lung fields
    left_lung = ((x - w * 0.35) ** 2 / (w * 0.18) ** 2 + (y - h * 0.50) ** 2 / (h * 0.32) ** 2) <= 1.0
    right_lung = ((x - w * 0.65) ** 2 / (w * 0.18) ** 2 + (y - h * 0.50) ** 2 / (h * 0.32) ** 2) <= 1.0
    return (left_lung | right_lung).astype(np.float32)


def compute_in_lung_attribution_ratio(
    saliency_map: np.ndarray,
    lung_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculates the fraction of total attribution energy situated inside anatomical lung fields.

    Args:
        saliency_map: 2D heatmap in [0, 1].
        lung_mask: Binary or soft 2D lung mask. If None, uses standardized lung field heuristic.

    Returns:
        Dict with in_lung_ratio, out_lung_ratio.
    """
    if lung_mask is None:
        lung_mask = generate_anatomical_lung_mask(saliency_map.shape)

    total_mass = float(np.sum(saliency_map))
    if total_mass == 0.0:
        return {"in_lung_ratio": 1.0, "out_lung_ratio": 0.0}

    in_lung_mass = float(np.sum(saliency_map * lung_mask))
    in_lung_ratio = float(in_lung_mass / max(total_mass, 1e-7))

    return {
        "in_lung_ratio": in_lung_ratio,
        "out_lung_ratio": float(1.0 - in_lung_ratio),
    }

