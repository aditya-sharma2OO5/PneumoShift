"""Unit tests for cross-domain evaluation configuration validation and path resolution."""

from pathlib import Path
import pytest

from pneumoshift.evaluation.cross_domain import (
    VALID_SHIFT_DIRECTIONS,
    VALID_THRESHOLD_METRICS,
    VALID_UNCERTAINTY_MODES,
    resolve_cross_domain_paths,
    validate_cross_domain_config,
)
from pneumoshift.utils.io import load_yaml_config


class TestCrossDomainConfig:
    def test_nih_to_chexpert_config_valid(self):
        cfg = load_yaml_config("configs/eval/nih_to_chexpert.yaml")
        validated = validate_cross_domain_config(cfg)
        assert validated["shift_direction"] == "nih_to_chexpert"
        assert validated["source_model"]["source_dataset"] == "nih"
        assert validated["target_data"]["dataset_name"] == "chexpert"
        assert validated["eval_protocol"]["target_calibration_ratio"] == 0.10
        assert "uncertain_as_negative" in validated["target_data"]["eval_uncertainty_modes"]

    def test_chexpert_to_nih_config_valid(self):
        cfg = load_yaml_config("configs/eval/chexpert_to_nih.yaml")
        validated = validate_cross_domain_config(cfg)
        assert validated["shift_direction"] == "chexpert_to_nih"
        assert validated["source_model"]["source_dataset"] == "chexpert"
        assert validated["target_data"]["dataset_name"] == "nih"
        assert validated["eval_protocol"]["target_calibration_ratio"] == 0.10

    def test_missing_mandatory_section_raises(self):
        cfg = {"shift_direction": "nih_to_chexpert"}
        with pytest.raises(ValueError, match="Missing mandatory configuration section"):
            validate_cross_domain_config(cfg)

    def test_invalid_shift_direction_raises(self):
        cfg = load_yaml_config("configs/eval/nih_to_chexpert.yaml")
        cfg["shift_direction"] = "invalid_shift"
        with pytest.raises(ValueError, match="Invalid shift_direction"):
            validate_cross_domain_config(cfg)

    def test_invalid_target_calibration_ratio_raises(self):
        cfg = load_yaml_config("configs/eval/nih_to_chexpert.yaml")
        cfg["eval_protocol"]["target_calibration_ratio"] = 1.5
        with pytest.raises(ValueError, match="target_calibration_ratio must be in"):
            validate_cross_domain_config(cfg)

    def test_invalid_uncertainty_mode_raises(self):
        cfg = load_yaml_config("configs/eval/nih_to_chexpert.yaml")
        cfg["target_data"]["eval_uncertainty_modes"] = ["invalid_mode"]
        with pytest.raises(ValueError, match="Invalid uncertainty mode"):
            validate_cross_domain_config(cfg)

    def test_invalid_confidence_level_raises(self):
        cfg = load_yaml_config("configs/eval/nih_to_chexpert.yaml")
        cfg["eval_protocol"]["confidence_level"] = 0.2
        with pytest.raises(ValueError, match="confidence_level must be in"):
            validate_cross_domain_config(cfg)

    def test_resolve_cross_domain_paths_smoke_test(self, tmp_path: Path):
        cfg = load_yaml_config("configs/eval/nih_to_chexpert.yaml")
        cfg["outputs"]["output_dir"] = str(tmp_path / "cross_eval")
        cfg["outputs"]["figures_dir"] = str(tmp_path / "cross_eval" / "figures")

        resolved = resolve_cross_domain_paths(cfg, smoke_test=True)
        assert resolved["is_synthetic"] is True
        assert resolved["output_dir"].exists()
        assert resolved["figures_dir"].exists()
        assert resolved["target_test_csv"].name == "synthetic_test.csv"
        assert resolved["target_cal_csv"].name == "synthetic_target_cal.csv"

