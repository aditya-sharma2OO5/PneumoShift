"""Localization benchmarking on ground-truth disease bounding boxes."""

from typing import Any, Dict, List, Tuple
import numpy as np


def compute_pointing_game_and_iou(
    saliency_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
    binarize_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates saliency localization metrics against a ground truth bounding box.

    Args:
        saliency_map: 2D attribution heatmap of shape (H, W) in [0, 1].
        bbox: (x_min, y_min, box_w, box_h) in pixel coordinates.
        binarize_threshold: Threshold to convert saliency into binary mask for IoU.

    Returns:
        Dict containing pointing_game_hit, iou, and energy_in_box.
    """
    h, w = saliency_map.shape
    x, y, bw, bh = bbox

    x_min = int(max(0, x))
    y_min = int(max(0, y))
    x_max = int(min(w, x + bw))
    y_max = int(min(h, y + bh))

    # Pointing game hit: maximum activation peak resides inside bbox
    peak_y, peak_x = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    hit = bool(x_min <= peak_x <= x_max and y_min <= peak_y <= y_max)

    # Energy in box
    total_energy = float(np.sum(saliency_map))
    box_energy = float(np.sum(saliency_map[y_min:y_max, x_min:x_max]))
    energy_ratio = float(box_energy / max(total_energy, 1e-7))

    # Binarized IoU
    binary_cam = saliency_map >= binarize_threshold
    ground_truth_mask = np.zeros((h, w), dtype=bool)
    ground_truth_mask[y_min:y_max, x_min:x_max] = True

    intersection = np.logical_and(binary_cam, ground_truth_mask).sum()
    union = np.logical_or(binary_cam, ground_truth_mask).sum()
    iou = float(intersection / max(union, 1))

    return {
        "pointing_game_hit": hit,
        "energy_in_box": energy_ratio,
        "iou": iou,
        "binarize_threshold": binarize_threshold,
    }

