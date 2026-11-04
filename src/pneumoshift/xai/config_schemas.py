"""Configuration validation schemas and parameter assertions for Phase 5 Explainability (XAI) experiments.

Validates master XAI configurations, faithfulness dynamics, Adebayo parameter
randomization sanity checks, and bounding box localization & shortcut analysis.
"""

from typing import Any, Dict, List, Optional
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.xai.config_schemas")

VALID_XAI_METHODS: List[str] = [
    "gradcam",
    "gradcam_plus_plus",
]

VALID_BASELINE_MODES: List[str] = [
    "zero",
    "blur",
    "mean",
]

VALID_RANDOMIZE_MODES: List[str] = [
    "classifier",
    "cascading",
    "all",
]

VALID_CORRELATION_METRICS: List[str] = [
    "spearman",
    "pearson",
    "mse",
    "ssim",
]


def validate_xai_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates global/master XAI configuration.

    Args:
        cfg: Raw configuration dictionary loaded from YAML.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If mandatory parameters are missing or out of valid bounds.
    """
    method = str(cfg.get("method", "gradcam")).lower()
    if method not in VALID_XAI_METHODS:
        raise ValueError(f"Invalid XAI method: '{method}'. Expected one of {VALID_XAI_METHODS}")

    target_class_idx = cfg.get("target_class_idx", 5)
    if not isinstance(target_class_idx, int) or target_class_idx < 0:
        raise ValueError(f"target_class_idx must be a non-negative integer, got {target_class_idx}")

    target_layers = cfg.get("target_layers", {})
    if not isinstance(target_layers, dict) or not target_layers:
        raise ValueError("target_layers must be a non-empty mapping of architecture to layer path")

    overlay = cfg.get("overlay", {})
    alpha = overlay.get("alpha", 0.5)
    if not isinstance(alpha, (int, float)) or not (0.0 <= alpha <= 1.0):
        raise ValueError(f"overlay.alpha must be in range [0.0, 1.0], got {alpha}")

    # Validate sub-sections if present
    if "faithfulness" in cfg:
        validate_faithfulness_config(cfg["faithfulness"], standalone=False)
    if "stability" in cfg:
        validate_stability_config(cfg["stability"], standalone=False)
    if "sanity_checks" in cfg:
        validate_sanity_checks_config(cfg["sanity_checks"], standalone=False)
    if "localization" in cfg:
        validate_localization_config(cfg["localization"], standalone=False)

    return cfg


def validate_faithfulness_config(cfg: Dict[str, Any], standalone: bool = True) -> Dict[str, Any]:
    """Validates faithfulness dynamics configuration (deletion & insertion dynamics).

    Args:
        cfg: Dictionary containing faithfulness parameters.
        standalone: Whether this is loaded as an independent file (checks experiment_name/models).

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If parameters are invalid or out of bounds.
    """
    if standalone and "experiment_name" not in cfg:
        raise ValueError("Faithfulness config missing mandatory key: 'experiment_name'")

    num_steps = cfg.get("num_steps", 10)
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError(f"num_steps must be a positive integer, got {num_steps}")

    step_size = cfg.get("step_size_percent", 5.0)
    if not isinstance(step_size, (int, float)) or not (0.0 < step_size <= 50.0):
        raise ValueError(f"step_size_percent must be in range (0.0, 50.0], got {step_size}")

    max_mask = cfg.get("max_mask_percent", 50.0)
    if not isinstance(max_mask, (int, float)) or not (0.0 < max_mask <= 100.0):
        raise ValueError(f"max_mask_percent must be in range (0.0, 100.0], got {max_mask}")

    if step_size > max_mask:
        raise ValueError(f"step_size_percent ({step_size}) cannot exceed max_mask_percent ({max_mask})")

    baseline_modes = cfg.get("baseline_modes", ["zero"])
    if not isinstance(baseline_modes, list) or not baseline_modes:
        raise ValueError("baseline_modes must be a non-empty list")
    for mode in baseline_modes:
        if str(mode).lower() not in VALID_BASELINE_MODES:
            raise ValueError(f"Invalid baseline mode: '{mode}'. Expected one of {VALID_BASELINE_MODES}")

    blur_kernel = cfg.get("blur_kernel_size", 11)
    if not isinstance(blur_kernel, int) or blur_kernel <= 0 or blur_kernel % 2 == 0:
        raise ValueError(f"blur_kernel_size must be a positive odd integer, got {blur_kernel}")

    blur_sigma = cfg.get("blur_sigma", 5.0)
    if not isinstance(blur_sigma, (int, float)) or blur_sigma <= 0:
        raise ValueError(f"blur_sigma must be positive float, got {blur_sigma}")

    return cfg


def validate_sanity_checks_config(cfg: Dict[str, Any], standalone: bool = True) -> Dict[str, Any]:
    """Validates parameter randomization sanity check configuration.

    Args:
        cfg: Dictionary containing sanity check parameters.
        standalone: Whether this is loaded as an independent file.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If parameters are invalid or out of bounds.
    """
    if standalone and "experiment_name" not in cfg:
        raise ValueError("Sanity checks config missing mandatory key: 'experiment_name'")

    randomize_modes = cfg.get("randomize_modes", ["classifier", "cascading", "all"])
    if not isinstance(randomize_modes, list) or not randomize_modes:
        raise ValueError("randomize_modes must be a non-empty list")
    for mode in randomize_modes:
        if str(mode).lower() not in VALID_RANDOMIZE_MODES:
            raise ValueError(f"Invalid randomize_mode: '{mode}'. Expected one of {VALID_RANDOMIZE_MODES}")

    corr_metrics = cfg.get("correlation_metrics", ["spearman", "pearson"])
    if not isinstance(corr_metrics, list) or not corr_metrics:
        raise ValueError("correlation_metrics must be a non-empty list")
    for m in corr_metrics:
        if str(m).lower() not in VALID_CORRELATION_METRICS:
            raise ValueError(f"Invalid correlation metric: '{m}'. Expected one of {VALID_CORRELATION_METRICS}")

    threshold_drop = cfg.get("threshold_drop_percent", 75.0)
    if not isinstance(threshold_drop, (int, float)) or not (0.0 <= threshold_drop <= 100.0):
        raise ValueError(f"threshold_drop_percent must be in range [0.0, 100.0], got {threshold_drop}")

    return cfg


def validate_localization_config(cfg: Dict[str, Any], standalone: bool = True) -> Dict[str, Any]:
    """Validates bounding box localization and shortcut analysis configuration.

    Args:
        cfg: Dictionary containing localization parameters.
        standalone: Whether this is loaded as an independent file.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If parameters are invalid or out of bounds.
    """
    if standalone and "experiment_name" not in cfg:
        raise ValueError("Localization config missing mandatory key: 'experiment_name'")

    iou_thresholds = cfg.get("iou_thresholds", [0.1, 0.3, 0.5])
    if not isinstance(iou_thresholds, list) or not iou_thresholds:
        raise ValueError("iou_thresholds must be a non-empty list of floats in (0.0, 1.0]")
    for iou in iou_thresholds:
        if not isinstance(iou, (int, float)) or not (0.0 < iou <= 1.0):
            raise ValueError(f"Each iou_threshold must be in range (0.0, 1.0], got {iou}")

    binarize_thresholds = cfg.get("binarize_thresholds", [0.3, 0.5, 0.7])
    if not isinstance(binarize_thresholds, list) or not binarize_thresholds:
        raise ValueError("binarize_thresholds must be a non-empty list of floats in (0.0, 1.0)")
    for b in binarize_thresholds:
        if not isinstance(b, (int, float)) or not (0.0 < b < 1.0):
            raise ValueError(f"Each binarize_threshold must be in range (0.0, 1.0), got {b}")

    shortcut_cfg = cfg.get("shortcut_analysis", {})
    border_frac = shortcut_cfg.get("border_margin_fraction", cfg.get("border_margin_fraction", 0.05))
    if not isinstance(border_frac, (int, float)) or not (0.0 < border_frac < 0.5):
        raise ValueError(f"border_margin_fraction must be in range (0.0, 0.5), got {border_frac}")

    corner_frac = shortcut_cfg.get("corner_margin_fraction", cfg.get("corner_margin_fraction", 0.08))
    if not isinstance(corner_frac, (int, float)) or not (0.0 < corner_frac < 0.5):
        raise ValueError(f"corner_margin_fraction must be in range (0.0, 0.5), got {corner_frac}")

    return cfg


def validate_stability_config(cfg: Dict[str, Any], standalone: bool = True) -> Dict[str, Any]:
    """Validates attribution stability and reproducibility configuration.

    Args:
        cfg: Dictionary containing stability parameters.
        standalone: Whether this is loaded as an independent file.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If parameters are invalid or out of bounds.
    """
    if standalone and "experiment_name" not in cfg:
        raise ValueError("Stability config missing mandatory key: 'experiment_name'")

    # Perturbations section (support either nested under 'perturbations' or flat in cfg)
    pert = cfg.get("perturbations", cfg)
    num_pert = pert.get("num_perturbations", 5)
    if not isinstance(num_pert, int) or num_pert <= 0:
        raise ValueError(f"num_perturbations must be a positive integer, got {num_pert}")

    rot = pert.get("rotation_degrees", 5.0)
    if not isinstance(rot, (int, float)) or not (0.0 <= rot <= 45.0):
        raise ValueError(f"rotation_degrees must be in range [0.0, 45.0], got {rot}")

    bf = pert.get("brightness_factor", 0.05)
    if not isinstance(bf, (int, float)) or not (0.0 <= bf < 1.0):
        raise ValueError(f"brightness_factor must be in range [0.0, 1.0), got {bf}")

    tf = pert.get("translation_percent", 0.05)
    if not isinstance(tf, (int, float)) or not (0.0 <= tf < 0.5):
        raise ValueError(f"translation_percent must be in range [0.0, 0.5), got {tf}")

    inter_seed = cfg.get("inter_seed", {})
    if isinstance(inter_seed, dict) and inter_seed:
        seeds = inter_seed.get("seeds", [42, 1337, 2024])
        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            raise ValueError(f"seeds must be a list of integers, got {seeds}")
        bin_thresh = inter_seed.get("binarize_threshold", 0.5)
        if not isinstance(bin_thresh, (int, float)) or not (0.0 < bin_thresh < 1.0):
            raise ValueError(f"binarize_threshold must be in range (0.0, 1.0), got {bin_thresh}")

    return cfg


