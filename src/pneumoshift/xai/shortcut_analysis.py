"""Shortcut feature detection: peripheral border, corner, and lateral marker energy."""

from typing import Any, Dict, Tuple
import numpy as np


def compute_shortcut_energy_ratio(
    saliency_map: np.ndarray,
    border_fraction: float = 0.05,
    corner_fraction: float = 0.08,
) -> Dict[str, float]:
    """Calculates attribution energy localized in peripheral image borders and corners.

    High attribution in image margins frequently indicates spurious shortcut learning
    (e.g., radiopaque text markers, scanner collimation edges, or external lead wires).
    """
    h, w = saliency_map.shape
    total_energy = float(np.sum(saliency_map))
    if total_energy == 0.0:
        return {"border_energy_ratio": 0.0, "corner_energy_ratio": 0.0}

    # Border mask
    border_mask = np.zeros((h, w), dtype=bool)
    b_h = int(h * border_fraction)
    b_w = int(w * border_fraction)
    border_mask[:b_h, :] = True
    border_mask[-b_h:, :] = True
    border_mask[:, :b_w] = True
    border_mask[:, -b_w:] = True

    # Corner mask
    corner_mask = np.zeros((h, w), dtype=bool)
    c_h = int(h * corner_fraction)
    c_w = int(w * corner_fraction)
    corner_mask[:c_h, :c_w] = True
    corner_mask[:c_h, -c_w:] = True
    corner_mask[-c_h:, :c_w] = True
    corner_mask[-c_h:, -c_w:] = True

    border_energy = float(np.sum(saliency_map[border_mask]))
    corner_energy = float(np.sum(saliency_map[corner_mask]))

    return {
        "border_energy_ratio": float(border_energy / total_energy),
        "corner_energy_ratio": float(corner_energy / total_energy),
    }

