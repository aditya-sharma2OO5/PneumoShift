"""Configuration validation schemas and parameter assertions for Phase 4 experiments.

Validates training configurations, model configurations, architecture comparison
evaluations, and uncertainty policy ablations.
"""

from typing import Any, Dict, List, Optional
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.config_schemas")

VALID_ARCHITECTURES: List[str] = [
    "densenet121",
    "efficientnet_b0",
    "efficientnet_b3",
]

VALID_UNCERTAINTY_POLICIES: List[str] = [
    "binary",
    "u_zeros",
    "u_ones",
    "u_ignore",
]

VALID_DATASET_NAMES: List[str] = [
    "nih",
    "chexpert",
    "synthetic",
]

VALID_OPTIMIZERS: List[str] = [
    "adam",
    "adamw",
    "sgd",
]

VALID_SHIFT_DIRECTIONS: List[str] = [
    "nih_to_chexpert",
    "chexpert_to_nih",
    "synthetic_to_synthetic",
]

VALID_THRESHOLD_METRICS: List[str] = [
    "youden",
    "f1",
    "sens90",
]


def validate_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates standalone model configuration.

    Args:
        cfg: Dictionary containing model parameters.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If mandatory parameters are missing or out of valid bounds.
    """
    if "architecture" not in cfg:
        raise ValueError("Model config missing mandatory key: 'architecture'")

    arch = str(cfg["architecture"]).lower()
    if arch not in VALID_ARCHITECTURES:
        raise ValueError(f"Invalid architecture: '{arch}'. Expected one of {VALID_ARCHITECTURES}")

    num_classes = cfg.get("num_classes", 8)
    if not isinstance(num_classes, int) or num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer, got {num_classes}")

    dropout = cfg.get("dropout", 0.2)
    if not isinstance(dropout, (int, float)) or not (0.0 <= dropout < 1.0):
        raise ValueError(f"dropout must be in range [0.0, 1.0), got {dropout}")

    channels = cfg.get("channels", 3)
    if not isinstance(channels, int) or channels <= 0:
        raise ValueError(f"channels must be a positive integer, got {channels}")

    in_features = cfg.get("in_features", cfg.get("feature_dim"))
    if in_features is not None:
        if not isinstance(in_features, int) or in_features <= 0:
            raise ValueError(f"in_features / feature_dim must be positive integer, got {in_features}")

    return cfg


def validate_train_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates full training configuration dictionary.

    Args:
        cfg: Raw configuration loaded from YAML.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If mandatory sections or invalid parameter values are encountered.
    """
    for sec in ["experiment_name", "data", "model", "train"]:
        if sec not in cfg:
            raise ValueError(f"Training config missing mandatory section: '{sec}'")

    data = cfg["data"]
    ds_name = str(data.get("dataset_name", "")).lower()
    if ds_name not in VALID_DATASET_NAMES:
        raise ValueError(f"Invalid dataset_name: '{ds_name}'. Expected one of {VALID_DATASET_NAMES}")

    uncertainty_policy = str(data.get("uncertainty_policy", "binary")).lower()
    if uncertainty_policy not in VALID_UNCERTAINTY_POLICIES:
        raise ValueError(
            f"Invalid uncertainty_policy: '{uncertainty_policy}'. Expected one of {VALID_UNCERTAINTY_POLICIES}"
        )

    img_size = data.get("image_size", 224)
    if not isinstance(img_size, int) or img_size <= 0:
        raise ValueError(f"image_size must be positive integer, got {img_size}")

    batch_size = data.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be positive integer, got {batch_size}")

    model = cfg["model"]
    validate_model_config(model)

    train = cfg["train"]
    epochs = train.get("epochs", 20)
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"epochs must be positive integer, got {epochs}")

    lr = train.get("learning_rate", 1e-4)
    if not isinstance(lr, (int, float)) or lr <= 0:
        raise ValueError(f"learning_rate must be positive float, got {lr}")

    wd = train.get("weight_decay", 1e-5)
    if not isinstance(wd, (int, float)) or wd < 0:
        raise ValueError(f"weight_decay must be non-negative float, got {wd}")

    opt = str(train.get("optimizer", "adamw")).lower()
    if opt not in VALID_OPTIMIZERS:
        raise ValueError(f"Invalid optimizer: '{opt}'. Expected one of {VALID_OPTIMIZERS}")

    return cfg


def validate_arch_comparison_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates configuration for architecture robustness comparison.

    Args:
        cfg: Configuration dictionary loaded from YAML.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If mandatory sections or invalid architecture specs are encountered.
    """
    for sec in ["experiment_name", "architectures", "shift_directions", "eval_protocol", "outputs"]:
        if sec not in cfg:
            raise ValueError(f"Architecture comparison config missing mandatory section: '{sec}'")

    archs = cfg["architectures"]
    if not isinstance(archs, list) or len(archs) < 2:
        raise ValueError("architectures must be a list containing at least 2 architectures to compare")

    seen_names = set()
    for arch_item in archs:
        if not isinstance(arch_item, dict) or "name" not in arch_item:
            raise ValueError("Each architecture entry must be a dictionary with a 'name' field")
        name = str(arch_item["name"]).lower()
        if name not in VALID_ARCHITECTURES:
            raise ValueError(f"Architecture '{name}' is not in valid architectures: {VALID_ARCHITECTURES}")
        if name in seen_names:
            raise ValueError(f"Duplicate architecture specified: '{name}'")
        seen_names.add(name)

    shifts = cfg["shift_directions"]
    if not isinstance(shifts, list) or len(shifts) == 0:
        raise ValueError("shift_directions must be a non-empty list of shift directions")
    for s in shifts:
        if s not in VALID_SHIFT_DIRECTIONS:
            raise ValueError(f"Invalid shift_direction: '{s}'. Expected one of {VALID_SHIFT_DIRECTIONS}")

    proto = cfg["eval_protocol"]
    bs = proto.get("bootstrap_samples", 1000)
    if not isinstance(bs, int) or bs <= 0:
        raise ValueError(f"bootstrap_samples must be positive integer, got {bs}")

    cl = proto.get("confidence_level", 0.95)
    if not isinstance(cl, (int, float)) or not (0.5 < cl < 1.0):
        raise ValueError(f"confidence_level must be in (0.5, 1.0), got {cl}")

    fdr = proto.get("fdr_alpha", 0.05)
    if not isinstance(fdr, (int, float)) or not (0.0 < fdr < 0.5):
        raise ValueError(f"fdr_alpha must be in (0.0, 0.5), got {fdr}")

    outputs = cfg["outputs"]
    if "output_dir" not in outputs or not outputs["output_dir"]:
        raise ValueError("outputs must contain a non-empty 'output_dir'")

    return cfg


def validate_uncertainty_ablation_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates configuration for CheXpert uncertainty policy ablation.

    Args:
        cfg: Configuration dictionary loaded from YAML.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If mandatory sections or invalid policy specs are encountered.
    """
    for sec in ["experiment_name", "policies", "target_data", "eval_protocol", "outputs"]:
        if sec not in cfg:
            raise ValueError(f"Uncertainty ablation config missing mandatory section: '{sec}'")

    policies = cfg["policies"]
    if not isinstance(policies, list) or len(policies) < 2:
        raise ValueError("policies must be a list containing at least 2 uncertainty policies to compare")

    valid_policy_names = ["u_zeros", "u_ones", "u_ignore"]
    seen_policies = set()
    for pol in policies:
        if not isinstance(pol, dict) or "name" not in pol:
            raise ValueError("Each policy entry must be a dictionary with a 'name' field")
        name = str(pol["name"]).lower()
        if name not in valid_policy_names:
            raise ValueError(f"Policy '{name}' is not supported. Expected one of {valid_policy_names}")
        if name in seen_policies:
            raise ValueError(f"Duplicate policy specified: '{name}'")
        seen_policies.add(name)

    proto = cfg["eval_protocol"]
    bs = proto.get("bootstrap_samples", 1000)
    if not isinstance(bs, int) or bs <= 0:
        raise ValueError(f"bootstrap_samples must be positive integer, got {bs}")

    cl = proto.get("confidence_level", 0.95)
    if not isinstance(cl, (int, float)) or not (0.5 < cl < 1.0):
        raise ValueError(f"confidence_level must be in (0.5, 1.0), got {cl}")

    t_metrics = proto.get("threshold_metrics", ["youden", "f1", "sens90"])
    if not isinstance(t_metrics, list) or len(t_metrics) == 0:
        raise ValueError("threshold_metrics must be a non-empty list")
    for tm in t_metrics:
        if tm not in VALID_THRESHOLD_METRICS:
            raise ValueError(f"Invalid threshold metric: '{tm}'. Expected one of {VALID_THRESHOLD_METRICS}")

    outputs = cfg["outputs"]
    if "output_dir" not in outputs or not outputs["output_dir"]:
        raise ValueError("outputs must contain a non-empty 'output_dir'")

    return cfg

