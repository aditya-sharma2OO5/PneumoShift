"""Faithfulness evaluation: Deletion and Insertion curves/AUC."""

from typing import Any, Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn


def compute_deletion_insertion_curves(
    model: nn.Module,
    image_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_class_idx: int = 5,
    num_steps: int = 10,
    mask_value: float = 0.0,
) -> Dict[str, Any]:
    """Computes deletion and insertion curves and their Area Under the Curve (AUC).

    Args:
        model: PyTorch model in eval mode.
        image_tensor: Input image tensor of shape (1, C, H, W).
        saliency_map: 2D attribution heatmap of shape (H, W).
        target_class_idx: Index of target class (e.g. Pneumonia).
        num_steps: Number of masking intervals (e.g. 10 steps = 0% to 100% or 0% to 50%).
        mask_value: Baseline pixel replacement value.

    Returns:
        Dict with deletion_curve, insertion_curve, deletion_auc, insertion_auc.
    """
    model.eval()
    device = next(model.parameters()).device
    img = image_tensor.to(device).clone()

    h, w = saliency_map.shape
    total_pixels = h * w
    flat_indices = np.argsort(-saliency_map.ravel())  # Descending order of saliency

    percentages = np.linspace(0.0, 1.0, num_steps + 1)
    deletion_scores: List[float] = []
    insertion_scores: List[float] = []

    # Baseline blank image
    baseline_img = torch.full_like(img, fill_value=mask_value)

    with torch.no_grad():
        orig_prob = float(torch.sigmoid(model(img))[0, target_class_idx].item())
        base_prob = float(torch.sigmoid(model(baseline_img))[0, target_class_idx].item())

        for p in percentages:
            k = int(p * total_pixels)
            mask_idx = flat_indices[:k]

            # Deletion: mask top-k% pixels in original image
            del_img = img.clone()
            c_channels = del_img.shape[1]
            for c in range(c_channels):
                flat_channel = del_img[0, c].view(-1)
                flat_channel[mask_idx] = mask_value
            del_prob = float(torch.sigmoid(model(del_img))[0, target_class_idx].item())
            deletion_scores.append(del_prob)

            # Insertion: insert top-k% pixels into baseline blank image
            ins_img = baseline_img.clone()
            for c in range(c_channels):
                flat_base = ins_img[0, c].view(-1)
                flat_orig = img[0, c].view(-1)
                flat_base[mask_idx] = flat_orig[mask_idx]
            ins_prob = float(torch.sigmoid(model(ins_img))[0, target_class_idx].item())
            insertion_scores.append(ins_prob)

    # Compute numerical trapezoidal AUC
    deletion_auc = float(np.trapz(deletion_scores, percentages))
    insertion_auc = float(np.trapz(insertion_scores, percentages))

    return {
        "percentages": percentages.tolist(),
        "deletion_curve": deletion_scores,
        "insertion_curve": insertion_scores,
        "deletion_auc": deletion_auc,
        "insertion_auc": insertion_auc,
    }
