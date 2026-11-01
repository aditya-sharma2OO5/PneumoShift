"""Attribution stability analysis across seeds, perturbations, and distribution shift."""

from typing import Any, Dict, List
import numpy as np
import scipy.stats
import torch
import torchvision.transforms.functional as TF
from pneumoshift.xai.gradcam import GradCAMExplainer


def compute_attribution_stability(
    explainer: GradCAMExplainer,
    image_tensor: torch.Tensor,
    target_class_idx: int = 5,
    num_perturbations: int = 5,
    max_rotation_deg: float = 5.0,
    brightness_factor: float = 0.05,
) -> Dict[str, float]:
    """Measures attribution stability under minor geometric and photometric perturbations."""
    base_cam = explainer.generate_cam(image_tensor, target_class_idx=target_class_idx)
    flat_base = base_cam.ravel()

    correlations: List[float] = []

    for _ in range(num_perturbations):
        rot = float(np.random.uniform(-max_rotation_deg, max_rotation_deg))
        bf = float(np.random.uniform(1.0 - brightness_factor, 1.0 + brightness_factor))

        perturbed = TF.rotate(image_tensor, rot)
        perturbed = TF.adjust_brightness(perturbed, bf)

        pert_cam = explainer.generate_cam(perturbed, target_class_idx=target_class_idx)
        # Un-rotate perturbed cam back to align
        pert_cam_tensor = torch.tensor(pert_cam).unsqueeze(0).unsqueeze(0)
        pert_cam_aligned = TF.rotate(pert_cam_tensor, -rot).squeeze().numpy()

        r, _ = scipy.stats.spearmanr(flat_base, pert_cam_aligned.ravel())
        if not np.isnan(r):
            correlations.append(float(r))

    return {
        "mean_perturbation_stability": float(np.mean(correlations)) if correlations else 1.0,
        "std_perturbation_stability": float(np.std(correlations)) if correlations else 0.0,
    }

