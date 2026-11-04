"""Explainability (XAI) suite: Grad-CAM, faithfulness, sanity checks, stability, localization, and shortcuts."""

from pneumoshift.xai.gradcam import (
    GradCAMExplainer,
    overlay_cam_on_image,
    save_cam_visualization,
    explain_image_with_model,
)
from pneumoshift.xai.faithfulness import compute_deletion_insertion_curves
from pneumoshift.xai.sanity_checks import run_parameter_randomization_test
from pneumoshift.xai.stability import compute_attribution_stability
from pneumoshift.xai.localization import compute_pointing_game_and_iou
from pneumoshift.xai.lung_mask import compute_in_lung_attribution_ratio
from pneumoshift.xai.shortcut_analysis import compute_shortcut_energy_ratio
from pneumoshift.xai.config_schemas import (
    validate_xai_config,
    validate_faithfulness_config,
    validate_sanity_checks_config,
    validate_localization_config,
)

__all__ = [
    "GradCAMExplainer",
    "overlay_cam_on_image",
    "save_cam_visualization",
    "explain_image_with_model",
    "compute_deletion_insertion_curves",
    "run_parameter_randomization_test",
    "compute_attribution_stability",
    "compute_pointing_game_and_iou",
    "compute_in_lung_attribution_ratio",
    "compute_shortcut_energy_ratio",
    "validate_xai_config",
    "validate_faithfulness_config",
    "validate_sanity_checks_config",
    "validate_localization_config",
]

