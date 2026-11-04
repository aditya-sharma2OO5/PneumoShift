"""Localization benchmarking on ground-truth disease bounding boxes (NIH ChestX-ray14).

Implements Pointing Game peak containment hit rates, Energy-in-BBox ratios,
multi-threshold binarized IoU benchmarking, cohort aggregation, and cross-domain
localization collapse statistical hypothesis testing.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats
import torch

from pneumoshift.utils.logging import get_logger
from pneumoshift.xai.gradcam import GradCAMExplainer

logger = get_logger("pneumoshift.xai.localization")


def compute_pointing_game_and_iou(
    saliency_map: np.ndarray,
    bbox: Tuple[int, int, int, int],
    binarize_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates saliency localization metrics against a single ground truth bounding box.

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
    if y_max > y_min and x_max > x_min:
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


def evaluate_image_localization(
    saliency_map: np.ndarray,
    bboxes: List[Tuple[int, int, int, int]],
    binarize_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Evaluates localization metrics against one or more ground-truth bounding boxes.

    Computes pointing game peak containment (hit if the attribution peak falls within
    any lesion bounding box), energy-in-box ratio over the union of all boxes, and
    binarized IoU across multiple activation thresholds (e.g., 0.3, 0.5, 0.7).

    Args:
        saliency_map: 2D attribution heatmap of shape (H, W) normalized in [0, 1].
        bboxes: List of (x, y, w, h) bounding box coordinates.
        binarize_thresholds: List of thresholds for IoU computation (default: [0.3, 0.5, 0.7]).

    Returns:
        Dict containing pointing_game_hit, energy_in_box, iou_by_threshold, mean_iou,
        and peak_coords.
    """
    if binarize_thresholds is None:
        binarize_thresholds = [0.3, 0.5, 0.7]

    h, w = saliency_map.shape
    peak_y, peak_x = np.unravel_index(np.argmax(saliency_map), saliency_map.shape)
    peak_coords = (int(peak_x), int(peak_y))

    if not bboxes:
        return {
            "pointing_game_hit": False,
            "energy_in_box": 0.0,
            "iou_by_threshold": {f"iou_{tau}": 0.0 for tau in binarize_thresholds},
            "mean_iou": 0.0,
            "num_bboxes": 0,
            "peak_coords": peak_coords,
        }

    # 1. Pointing Game Hit: does peak fall inside ANY bounding box?
    hit = False
    gt_union_mask = np.zeros((h, w), dtype=bool)

    for x, y, bw, bh in bboxes:
        x_min = int(max(0, x))
        y_min = int(max(0, y))
        x_max = int(min(w, x + bw))
        y_max = int(min(h, y + bh))

        if x_min <= peak_x <= x_max and y_min <= peak_y <= y_max:
            hit = True

        if y_max > y_min and x_max > x_min:
            gt_union_mask[y_min:y_max, x_min:x_max] = True

    # 2. Energy in Box Union
    total_energy = float(np.sum(saliency_map))
    box_energy = float(np.sum(saliency_map[gt_union_mask]))
    energy_in_box = float(box_energy / max(total_energy, 1e-8))

    # 3. Multi-threshold IoU
    iou_by_threshold: Dict[str, float] = {}
    for tau in binarize_thresholds:
        binary_cam = saliency_map >= tau
        intersection = np.logical_and(binary_cam, gt_union_mask).sum()
        union = np.logical_or(binary_cam, gt_union_mask).sum()
        iou = float(intersection / max(union, 1))
        iou_by_threshold[f"iou_{tau}"] = iou

    mean_iou = float(np.mean(list(iou_by_threshold.values()))) if iou_by_threshold else 0.0

    return {
        "pointing_game_hit": hit,
        "energy_in_box": energy_in_box,
        "iou_by_threshold": iou_by_threshold,
        "mean_iou": mean_iou,
        "num_bboxes": len(bboxes),
        "peak_coords": peak_coords,
    }


def evaluate_cohort_localization(
    saliency_maps: Union[List[np.ndarray], Dict[str, np.ndarray]],
    bboxes_per_image: Union[List[List[Tuple[int, int, int, int]]], Dict[str, List[Tuple[int, int, int, int]]]],
    binarize_thresholds: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """Evaluates localization metrics over a cohort of test radiographs with ground-truth boxes.

    Args:
        saliency_maps: List or dict of 2D attribution heatmaps.
        bboxes_per_image: List or dict of bounding box lists corresponding to each image.
        binarize_thresholds: Thresholds for IoU calculation (default: [0.3, 0.5, 0.7]).

    Returns:
        Consolidated dictionary of cohort localization statistics.
    """
    if binarize_thresholds is None:
        binarize_thresholds = [0.3, 0.5, 0.7]

    # Align keys/indices
    if isinstance(saliency_maps, dict) and isinstance(bboxes_per_image, dict):
        common_keys = [k for k in saliency_maps.keys() if k in bboxes_per_image]
        map_list = [saliency_maps[k] for k in common_keys]
        box_list = [bboxes_per_image[k] for k in common_keys]
    elif isinstance(saliency_maps, list) and isinstance(bboxes_per_image, list):
        if len(saliency_maps) != len(bboxes_per_image):
            raise ValueError(f"Length mismatch: {len(saliency_maps)} maps vs {len(bboxes_per_image)} box lists")
        map_list = saliency_maps
        box_list = bboxes_per_image
    else:
        raise TypeError("saliency_maps and bboxes_per_image must both be lists or both be dicts")

    if not map_list:
        raise ValueError("Cohort cannot be empty")

    sample_results: List[Dict[str, Any]] = []
    hits = 0
    energy_ratios: List[float] = []
    threshold_ious: Dict[float, List[float]] = {tau: [] for tau in binarize_thresholds}
    mean_ious: List[float] = []

    for cam, bboxes in zip(map_list, box_list):
        res = evaluate_image_localization(cam, bboxes, binarize_thresholds=binarize_thresholds)
        sample_results.append(res)

        if res["pointing_game_hit"]:
            hits += 1
        energy_ratios.append(res["energy_in_box"])
        mean_ious.append(res["mean_iou"])
        for tau in binarize_thresholds:
            threshold_ious[tau].append(res["iou_by_threshold"][f"iou_{tau}"])

    total_samples = len(map_list)
    hit_rate = float(hits / total_samples)
    mean_energy = float(np.mean(energy_ratios))
    std_energy = float(np.std(energy_ratios))
    overall_mean_iou = float(np.mean(mean_ious))
    std_mean_iou = float(np.std(mean_ious))

    mean_iou_by_threshold = {
        f"iou_{tau}": float(np.mean(vals)) for tau, vals in threshold_ious.items()
    }

    return {
        "num_samples": total_samples,
        "pointing_game_hits": hits,
        "pointing_game_hit_rate": hit_rate,
        "mean_energy_in_box": mean_energy,
        "std_energy_in_box": std_energy,
        "overall_mean_iou": overall_mean_iou,
        "std_mean_iou": std_mean_iou,
        "mean_iou_by_threshold": mean_iou_by_threshold,
        "sample_results": sample_results,
    }


def compare_cross_domain_localization(
    in_domain_results: Dict[str, Any],
    transferred_results: Dict[str, Any],
) -> Dict[str, Any]:
    """Tests Hypothesis 5D: Clinical Localization Collapse Under Cross-Domain Distribution Shift.

    Hypothesis 5D states: In-domain models achieve significantly higher Pointing Game hit rates
    and Energy-in-BBox ratios on official NIH radiologist-annotated bounding boxes than
    cross-domain transferred models (p < 0.05).

    Args:
        in_domain_results: Cohort localization results for in-domain trained model.
        transferred_results: Cohort localization results for transferred model.

    Returns:
        Statistical test results including Fisher's exact test for Pointing Game hit rate,
        Mann-Whitney U / Wilcoxon for Energy in Box and IoU, and hypothesis verdict.
    """
    n_in = in_domain_results["num_samples"]
    n_tr = transferred_results["num_samples"]

    hits_in = in_domain_results["pointing_game_hits"]
    miss_in = n_in - hits_in
    hits_tr = transferred_results["pointing_game_hits"]
    miss_tr = n_tr - hits_tr

    # 1. Fisher's Exact Test on Pointing Game Hit / Miss contingency table
    table = [[hits_in, miss_in], [hits_tr, miss_tr]]
    try:
        odds_ratio, p_pointing = scipy.stats.fisher_exact(table, alternative="greater")
    except Exception:
        odds_ratio, p_pointing = 1.0, 1.0

    # 2. Energy in Box comparison
    in_energies = [s["energy_in_box"] for s in in_domain_results.get("sample_results", [])]
    tr_energies = [s["energy_in_box"] for s in transferred_results.get("sample_results", [])]

    if in_energies and tr_energies and (in_energies != tr_energies):
        try:
            stat_energy, p_energy = scipy.stats.mannwhitneyu(in_energies, tr_energies, alternative="greater")
        except Exception:
            stat_energy, p_energy = 0.0, 1.0
    else:
        stat_energy, p_energy = 0.0, 1.0

    # 3. IoU comparison
    in_ious = [s["mean_iou"] for s in in_domain_results.get("sample_results", [])]
    tr_ious = [s["mean_iou"] for s in transferred_results.get("sample_results", [])]

    if in_ious and tr_ious and (in_ious != tr_ious):
        try:
            stat_iou, p_iou = scipy.stats.mannwhitneyu(in_ious, tr_ious, alternative="greater")
        except Exception:
            stat_iou, p_iou = 0.0, 1.0
    else:
        stat_iou, p_iou = 0.0, 1.0

    pointing_rate_in = in_domain_results["pointing_game_hit_rate"]
    pointing_rate_tr = transferred_results["pointing_game_hit_rate"]
    energy_in = in_domain_results["mean_energy_in_box"]
    energy_tr = transferred_results["mean_energy_in_box"]

    significant_pointing = bool(p_pointing < 0.05 and pointing_rate_in > pointing_rate_tr)
    significant_energy = bool(p_energy < 0.05 and energy_in > energy_tr)

    hypothesis_5d_supported = significant_pointing or significant_energy

    if hypothesis_5d_supported:
        verdict = (
            f"SUPPORTED: In-domain model demonstrates statistically superior lesion localization "
            f"(Pointing Game Hit Rate: {pointing_rate_in:.1%} vs {pointing_rate_tr:.1%}, p={p_pointing:.4f}; "
            f"Energy-in-Box: {energy_in:.3f} vs {energy_tr:.3f}, p={p_energy:.4f}), "
            f"confirming out-of-domain explainability collapse."
        )
    else:
        verdict = (
            f"INCONCLUSIVE / NOT SIGNIFICANT: Pointing Game hit rate ({pointing_rate_in:.1%} vs {pointing_rate_tr:.1%}, "
            f"p={p_pointing:.4f}) and Energy-in-Box ({energy_in:.3f} vs {energy_tr:.3f}, p={p_energy:.4f}) "
            f"did not achieve p < 0.05 significance threshold."
        )

    return {
        "pointing_game_in_domain": pointing_rate_in,
        "pointing_game_transferred": pointing_rate_tr,
        "pointing_game_odds_ratio": float(odds_ratio),
        "pointing_game_p_value": float(p_pointing),
        "significant_pointing_collapse": significant_pointing,
        "energy_in_box_in_domain": energy_in,
        "energy_in_box_transferred": energy_tr,
        "energy_in_box_p_value": float(p_energy),
        "significant_energy_collapse": significant_energy,
        "mean_iou_in_domain": in_domain_results["overall_mean_iou"],
        "mean_iou_transferred": transferred_results["overall_mean_iou"],
        "iou_p_value": float(p_iou),
        "hypothesis_5d_supported": hypothesis_5d_supported,
        "verdict": verdict,
    }
