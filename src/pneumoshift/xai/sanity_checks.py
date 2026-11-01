"""Sanity checks for visual explanations (Adebayo et al., 2018)."""

import copy
from typing import Any, Dict, List
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from pneumoshift.models.factory import get_target_layer_for_xai
from pneumoshift.xai.gradcam import GradCAMExplainer


def compute_map_similarity(map_a: np.ndarray, map_b: np.ndarray) -> Dict[str, float]:
    """Computes similarity metrics between two attribution heatmaps."""
    flat_a = map_a.ravel()
    flat_b = map_b.ravel()

    # Spearman rank correlation
    spearman_corr, _ = scipy.stats.spearmanr(flat_a, flat_b)
    # Pearson correlation
    pearson_corr, _ = scipy.stats.pearsonr(flat_a, flat_b)
    # Mean Squared Error
    mse = float(np.mean((flat_a - flat_b) ** 2))

    return {
        "spearman_rank_correlation": float(spearman_corr) if not np.isnan(spearman_corr) else 0.0,
        "pearson_correlation": float(pearson_corr) if not np.isnan(pearson_corr) else 0.0,
        "mean_squared_error": mse,
    }


def run_parameter_randomization_test(
    model: nn.Module,
    architecture: str,
    image_tensor: torch.Tensor,
    original_cam: np.ndarray,
    target_class_idx: int = 5,
    randomize_mode: str = "all",
) -> Dict[str, Any]:
    """Performs model parameter randomization sanity test (Adebayo et al.).

    Args:
        model: Original trained model.
        architecture: Architecture string ('densenet121' or 'efficientnet_b0').
        image_tensor: Image tensor of shape (1, C, H, W).
        original_cam: Heatmap from the unrandomized trained model.
        target_class_idx: Target disease class index.
        randomize_mode: 'classifier', 'cascading', or 'all'.

    Returns:
        Dict with similarity metrics against original attribution.
    """
    random_model = copy.deepcopy(model)
    random_model.eval()

    # Reinitialize weights randomly
    if randomize_mode in ("classifier", "all"):
        if hasattr(random_model, "head"):
            for layer in random_model.head.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

    if randomize_mode in ("cascading", "all"):
        if hasattr(random_model, "backbone"):
            for layer in random_model.backbone.modules():
                if isinstance(layer, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

    target_layer = get_target_layer_for_xai(random_model, architecture)
    explainer = GradCAMExplainer(random_model, target_layer)
    random_cam = explainer.generate_cam(image_tensor, target_class_idx=target_class_idx)
    explainer.release_hooks()

    sim = compute_map_similarity(original_cam, random_cam)
    sim["randomize_mode"] = randomize_mode
    return sim

