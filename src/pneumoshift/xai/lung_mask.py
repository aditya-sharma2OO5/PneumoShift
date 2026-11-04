"""Anatomical plausibility via lung field segmentation and attribution mass ratios."""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


def generate_anatomical_lung_mask(
    image_shape: Tuple[int, int],
) -> np.ndarray:
    """Generates standard bilateral anatomical lung field mask heuristic.

    Uses smooth anatomical elliptical envelopes defining left and right hemithoracic
    cavities, excluding mediastinal (cardiac/vertebral) and subdiaphragmatic regions.

    Args:
        image_shape: (H, W) spatial dimensions of target radiograph.

    Returns:
        Binary float32 mask of shape (H, W) where 1.0 denotes lung parenchyma and 0.0 outside.
    """
    h, w = image_shape
    y, x = np.ogrid[:h, :w]

    # Anatomical hemithoracic envelopes
    # Right lung (anatomical right, left side of standard PA radiograph view: x ~ 0.35)
    right_lung = ((x - w * 0.35) ** 2 / (w * 0.18) ** 2 + (y - h * 0.48) ** 2 / (h * 0.30) ** 2) <= 1.0

    # Left lung (anatomical left, right side of standard PA view: x ~ 0.65, cardiac notch accounted for)
    left_lung = ((x - w * 0.65) ** 2 / (w * 0.17) ** 2 + (y - h * 0.46) ** 2 / (h * 0.28) ** 2) <= 1.0

    # Exclude central mediastinal strip (cardiac silhouette + spine)
    mediastinum_strip = np.abs(x - w * 0.50) < (w * 0.07)
    left_cardiac_notch = ((x - w * 0.52) ** 2 / (w * 0.12) ** 2 + (y - h * 0.58) ** 2 / (h * 0.14) ** 2) <= 1.0

    mask = (right_lung | left_lung) & (~mediastinum_strip) & (~left_cardiac_notch)
    return mask.astype(np.float32)


def extract_adaptive_lung_mask(
    image: Union[np.ndarray, torch.Tensor],
) -> np.ndarray:
    """Extracts an adaptive lung field mask from image grayscale intensity dynamics.

    Uses intensity histogram thresholding to separate darker aerated lung tissue from
    dense bone/soft tissue, constrained by the standard anatomical thoracic bounding envelope.

    Args:
        image: Radiograph as 2D numpy array (H, W) or PyTorch tensor (1, C, H, W) or (C, H, W).

    Returns:
        Binary float32 mask of shape (H, W).
    """
    if isinstance(image, torch.Tensor):
        arr = image.squeeze().detach().cpu().numpy()
        if arr.ndim == 3:
            arr = np.mean(arr, axis=0)
    else:
        arr = np.array(image, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.mean(arr, axis=0)

    h, w = arr.shape
    base_mask = generate_anatomical_lung_mask((h, w))

    # Standardize image to [0, 1]
    denom = arr.max() - arr.min()
    if denom > 1e-6:
        norm_img = (arr - arr.min()) / denom
    else:
        norm_img = arr

    # Aerated lungs are typically radiolucent (darker, lower pixel intensities)
    # inside the anatomical ribcage envelope
    inside_vals = norm_img[base_mask > 0.5]
    if len(inside_vals) > 0 and np.std(inside_vals) > 1e-3:
        threshold = float(np.percentile(inside_vals, 85))
        adaptive_mask = (norm_img <= threshold) & (base_mask > 0.5)
        # If adaptive segmentation covers at least 40% of baseline, use it
        if adaptive_mask.sum() >= 0.4 * base_mask.sum():
            return adaptive_mask.astype(np.float32)

    return base_mask


def compute_in_lung_attribution_ratio(
    saliency_map: np.ndarray,
    lung_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calculates the fraction of total attribution energy situated inside anatomical lung fields.

    Args:
        saliency_map: 2D heatmap in [0, 1] of shape (H, W).
        lung_mask: Binary or soft 2D lung mask. If None, uses standardized lung field heuristic.

    Returns:
        Dict with in_lung_ratio, out_lung_ratio, mediastinum_ratio, and subdiaphragmatic_ratio.
    """
    h, w = saliency_map.shape
    if lung_mask is None:
        lung_mask = generate_anatomical_lung_mask((h, w))

    total_mass = float(np.sum(saliency_map))
    if total_mass < 1e-8:
        return {
            "in_lung_ratio": 1.0,
            "out_lung_ratio": 0.0,
            "mediastinum_ratio": 0.0,
            "subdiaphragmatic_ratio": 0.0,
        }

    in_lung_mass = float(np.sum(saliency_map * lung_mask))
    in_lung_ratio = float(in_lung_mass / total_mass)

    # Mediastinal central strip (x in [0.42 W, 0.58 W], y in [0.2 H, 0.7 H])
    y, x = np.ogrid[:h, :w]
    mediastinum_mask = (np.abs(x - w * 0.50) <= (w * 0.08)) & (y >= h * 0.20) & (y <= h * 0.75)
    mediastinum_mass = float(np.sum(saliency_map[mediastinum_mask]))
    mediastinum_ratio = float(mediastinum_mass / total_mass)

    # Subdiaphragmatic abdominal region (y > 0.80 H)
    sub_h = int(round(h * 0.80))
    subdiaphragmatic_mass = float(np.sum(saliency_map[sub_h:, :]))
    subdiaphragmatic_ratio = float(subdiaphragmatic_mass / total_mass)

    return {
        "in_lung_ratio": in_lung_ratio,
        "out_lung_ratio": float(max(0.0, 1.0 - in_lung_ratio)),
        "mediastinum_ratio": mediastinum_ratio,
        "subdiaphragmatic_ratio": subdiaphragmatic_ratio,
    }


def evaluate_cohort_lung_plausibility(
    saliency_maps: List[np.ndarray],
    lung_masks: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Evaluates anatomical lung attribution plausibility across a cohort of radiographs.

    Args:
        saliency_maps: List of 2D attribution heatmaps.
        lung_masks: Optional list of corresponding lung masks.

    Returns:
        Dict with mean and std of in_lung_ratio, mediastinum_ratio, and subdiaphragmatic_ratio.
    """
    if not saliency_maps:
        raise ValueError("saliency_maps list cannot be empty")

    in_lung_list: List[float] = []
    out_lung_list: List[float] = []
    med_list: List[float] = []
    sub_list: List[float] = []

    for idx, cam in enumerate(saliency_maps):
        mask = lung_masks[idx] if lung_masks is not None and idx < len(lung_masks) else None
        res = compute_in_lung_attribution_ratio(cam, lung_mask=mask)
        in_lung_list.append(res["in_lung_ratio"])
        out_lung_list.append(res["out_lung_ratio"])
        med_list.append(res["mediastinum_ratio"])
        sub_list.append(res["subdiaphragmatic_ratio"])

    return {
        "num_samples": len(saliency_maps),
        "mean_in_lung_ratio": float(np.mean(in_lung_list)),
        "std_in_lung_ratio": float(np.std(in_lung_list)),
        "mean_out_lung_ratio": float(np.mean(out_lung_list)),
        "std_out_lung_ratio": float(np.std(out_lung_list)),
        "mean_mediastinum_ratio": float(np.mean(med_list)),
        "mean_subdiaphragmatic_ratio": float(np.mean(sub_list)),
        "in_lung_ratios": in_lung_list,
    }
