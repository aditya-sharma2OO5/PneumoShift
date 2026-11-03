"""Cross-dataset evaluation harness, configuration validation, and path resolution."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.data.splits import create_synthetic_dataset
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.cross_domain")

VALID_SHIFT_DIRECTIONS: List[str] = [
    "nih_to_chexpert",
    "chexpert_to_nih",
    "synthetic_to_synthetic",
]

VALID_UNCERTAINTY_MODES: List[str] = [
    "uncertain_as_negative",
    "uncertain_excluded",
]

VALID_THRESHOLD_METRICS: List[str] = [
    "youden",
    "f1",
    "sens90",
]


def validate_cross_domain_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates schema, parameters, and constraints for cross-dataset evaluation.

    Args:
        cfg: Raw configuration dictionary loaded from YAML.

    Returns:
        Validated and normalized configuration dictionary.

    Raises:
        ValueError: If mandatory sections or invalid parameter values are encountered.
    """
    # 1. Mandatory top-level sections
    for sec in ["source_model", "target_data", "eval_protocol", "outputs"]:
        if sec not in cfg:
            raise ValueError(f"Missing mandatory configuration section: '{sec}'")

    shift_dir = cfg.get("shift_direction", "")
    if shift_dir not in VALID_SHIFT_DIRECTIONS:
        raise ValueError(
            f"Invalid shift_direction: '{shift_dir}'. Expected one of {VALID_SHIFT_DIRECTIONS}"
        )

    # 2. Source model validation
    src_model = cfg["source_model"]
    if "architecture" not in src_model:
        raise ValueError("source_model missing mandatory 'architecture'")
    if "checkpoint_path" not in src_model:
        raise ValueError("source_model missing mandatory 'checkpoint_path'")

    # 3. Target data validation
    tgt_data = cfg["target_data"]
    if "dataset_name" not in tgt_data:
        raise ValueError("target_data missing mandatory 'dataset_name'")
    if "splits_dir" not in tgt_data:
        raise ValueError("target_data missing mandatory 'splits_dir'")

    uncertainty_modes = tgt_data.get("eval_uncertainty_modes", ["uncertain_as_negative"])
    for mode in uncertainty_modes:
        if mode not in VALID_UNCERTAINTY_MODES:
            raise ValueError(
                f"Invalid uncertainty mode '{mode}'. Expected one of {VALID_UNCERTAINTY_MODES}"
            )

    # 4. Evaluation protocol validation
    protocol = cfg["eval_protocol"]
    cal_ratio = float(protocol.get("target_calibration_ratio", 0.10))
    if not (0.0 < cal_ratio < 1.0):
        raise ValueError(
            f"target_calibration_ratio must be in (0, 1), got {cal_ratio}"
        )

    conf_level = float(protocol.get("confidence_level", 0.95))
    if not (0.5 < conf_level < 1.0):
        raise ValueError(f"confidence_level must be in (0.5, 1.0), got {conf_level}")

    boot_samples = int(protocol.get("bootstrap_samples", 1000))
    if boot_samples < 10:
        raise ValueError(f"bootstrap_samples must be >= 10, got {boot_samples}")

    thresh_metrics = protocol.get("threshold_metrics", ["youden", "f1", "sens90"])
    for tm in thresh_metrics:
        if tm not in VALID_THRESHOLD_METRICS:
            raise ValueError(
                f"Invalid threshold metric '{tm}'. Expected one of {VALID_THRESHOLD_METRICS}"
            )

    # 5. Output locations
    out_dir = cfg["outputs"].get("output_dir")
    if not out_dir:
        raise ValueError("outputs section missing mandatory 'output_dir'")

    logger.debug("Cross-domain configuration validated successfully for shift: %s", shift_dir)
    return cfg


def resolve_cross_domain_paths(
    cfg: Dict[str, Any],
    smoke_test: bool = False,
) -> Dict[str, Path]:
    """Resolves all filesystem paths for source model, target splits, and output directories.

    In dry-run or smoke-test mode, automatically falls back to synthetic datasets and mock weights.

    Args:
        cfg: Validated cross-domain configuration dictionary.
        smoke_test: Whether to force synthetic smoke test paths.

    Returns:
        Dict mapping logical keys to resolved Paths.
    """
    src_model = cfg["source_model"]
    tgt_data = cfg["target_data"]
    tgt_ds = tgt_data["dataset_name"]
    src_ds = src_model.get("source_dataset", "nih")

    ckpt_path = Path(src_model["checkpoint_path"])
    src_val_csv = Path(src_model.get("source_val_csv", f"data/splits/{src_ds}/{src_ds}_val.csv"))
    src_metrics_path = Path(src_model.get("source_test_metrics", f"results/runs/{src_ds}_densenet121/test_metrics_densenet121_42.json"))

    splits_dir = Path(tgt_data["splits_dir"])
    test_split_name = tgt_data.get("test_split", "test")
    cal_split_name = tgt_data.get("target_cal_split", "target_cal")
    off_val_split_name = tgt_data.get("official_val_split", "official_val")

    target_test_csv = splits_dir / f"{tgt_ds}_{test_split_name}.csv"
    target_cal_csv = splits_dir / f"{tgt_ds}_{cal_split_name}.csv"
    target_off_val_csv = splits_dir / f"{tgt_ds}_{off_val_split_name}.csv"
    images_dir = Path(tgt_data.get("images_dir", ""))

    out_dir = Path(cfg["outputs"]["output_dir"])
    figures_dir = Path(cfg["outputs"].get("figures_dir", out_dir / "figures"))

    # Fallback to synthetic cohort if smoke_test requested or real files missing
    use_synthetic = smoke_test or (not target_test_csv.is_file() and not ckpt_path.is_file())
    if use_synthetic:
        logger.info("Using synthetic smoke test environment for cross-domain evaluation.")
        synth_dir = Path("data/synthetic")
        if not (synth_dir / "splits" / "synthetic_test.csv").is_file():
            create_synthetic_dataset(output_dir=synth_dir, num_samples=40, image_size=64)

        src_val_csv = synth_dir / "splits" / "synthetic_val.csv"
        target_test_csv = synth_dir / "splits" / "synthetic_test.csv"
        target_cal_csv = synth_dir / "splits" / "synthetic_target_cal.csv"
        target_off_val_csv = synth_dir / "splits" / "synthetic_official_val.csv"
        images_dir = synth_dir / "images"

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "checkpoint_path": ckpt_path,
        "source_val_csv": src_val_csv,
        "source_test_metrics": src_metrics_path,
        "target_test_csv": target_test_csv,
        "target_cal_csv": target_cal_csv,
        "target_off_val_csv": target_off_val_csv,
        "images_dir": images_dir,
        "output_dir": out_dir,
        "figures_dir": figures_dir,
        "is_synthetic": use_synthetic,
    }
    return resolved

