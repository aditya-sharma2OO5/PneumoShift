"""Explainability (XAI) suite: Grad-CAM, faithfulness, sanity checks, stability, localization, and shortcuts."""

from pneumoshift.xai.gradcam import GradCAMExplainer
from pneumoshift.xai.faithfulness import compute_deletion_insertion_curves
from pneumoshift.xai.sanity_checks import run_parameter_randomization_test
from pneumoshift.xai.stability import compute_attribution_stability
from pneumoshift.xai.localization import compute_pointing_game_and_iou
from pneumoshift.xai.lung_mask import compute_in_lung_attribution_ratio
from pneumoshift.xai.shortcut_analysis import compute_shortcut_energy_ratio

__all__ = [
    "GradCAMExplainer",
    "compute_deletion_insertion_curves",
    "run_parameter_randomization_test",
    "compute_attribution_stability",
    "compute_pointing_game_and_iou",
    "compute_in_lung_attribution_ratio",
    "compute_shortcut_energy_ratio",
]

