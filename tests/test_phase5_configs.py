"""Unit tests for Phase 5 Explainability (XAI) configuration schemas."""

from pathlib import Path
import pytest
import yaml

from pneumoshift.xai.config_schemas import (
    validate_xai_config,
    validate_faithfulness_config,
    validate_sanity_checks_config,
    validate_localization_config,
    VALID_XAI_METHODS,
    VALID_BASELINE_MODES,
    VALID_RANDOMIZE_MODES,
)


CONFIGS_DIR = Path("configs/xai")


class TestPhase5ConfigLoading:
    """Verifies that all Phase 5 YAML files in configs/xai/ pass schema validation."""

    def test_default_xai_yaml_valid(self):
        cfg_path = CONFIGS_DIR / "default_xai.yaml"
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        validated = validate_xai_config(cfg)
        assert validated["method"] in VALID_XAI_METHODS
        assert "target_layers" in validated

    def test_faithfulness_yaml_valid(self):
        cfg_path = CONFIGS_DIR / "faithfulness.yaml"
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        validated = validate_faithfulness_config(cfg)
        assert validated["num_steps"] > 0
        assert all(m in VALID_BASELINE_MODES for m in validated["baseline_modes"])

    def test_sanity_checks_yaml_valid(self):
        cfg_path = CONFIGS_DIR / "sanity_checks.yaml"
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        validated = validate_sanity_checks_config(cfg)
        assert all(m in VALID_RANDOMIZE_MODES for m in validated["randomize_modes"])
        assert 0.0 <= validated["threshold_drop_percent"] <= 100.0

    def test_localization_yaml_valid(self):
        cfg_path = CONFIGS_DIR / "localization.yaml"
        assert cfg_path.is_file(), f"Missing config file: {cfg_path}"
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        validated = validate_localization_config(cfg)
        assert len(validated["iou_thresholds"]) > 0
        assert len(validated["binarize_thresholds"]) > 0


class TestXAIConfigValidation:
    """Tests boundary assertions and validation logic for global XAI configs."""

    def test_valid_minimal_config(self):
        cfg = {
            "method": "gradcam",
            "target_class_idx": 5,
            "target_layers": {"densenet121": "features.denseblock4"},
            "overlay": {"alpha": 0.4},
        }
        res = validate_xai_config(cfg)
        assert res["method"] == "gradcam"

    def test_invalid_method_raises(self):
        cfg = {"method": "invalid_method", "target_layers": {"densenet121": "x"}}
        with pytest.raises(ValueError, match="Invalid XAI method"):
            validate_xai_config(cfg)

    def test_negative_target_class_idx_raises(self):
        cfg = {"method": "gradcam", "target_class_idx": -1, "target_layers": {"d": "x"}}
        with pytest.raises(ValueError, match="target_class_idx must be a non-negative integer"):
            validate_xai_config(cfg)

    def test_empty_target_layers_raises(self):
        cfg = {"method": "gradcam", "target_layers": {}}
        with pytest.raises(ValueError, match="target_layers must be a non-empty mapping"):
            validate_xai_config(cfg)

    def test_invalid_overlay_alpha_raises(self):
        cfg = {"target_layers": {"d": "x"}, "overlay": {"alpha": 1.5}}
        with pytest.raises(ValueError, match="overlay.alpha must be in range"):
            validate_xai_config(cfg)


class TestFaithfulnessConfigValidation:
    """Tests boundary assertions for faithfulness dynamics configurations."""

    def test_valid_faithfulness_config(self):
        cfg = {
            "experiment_name": "faith_test",
            "num_steps": 10,
            "step_size_percent": 5.0,
            "max_mask_percent": 50.0,
            "baseline_modes": ["zero", "blur"],
            "blur_kernel_size": 11,
            "blur_sigma": 5.0,
        }
        res = validate_faithfulness_config(cfg)
        assert res["num_steps"] == 10

    def test_step_size_exceeding_max_mask_raises(self):
        cfg = {
            "experiment_name": "faith_test",
            "num_steps": 10,
            "step_size_percent": 40.0,
            "max_mask_percent": 30.0,
            "baseline_modes": ["zero"],
        }
        with pytest.raises(ValueError, match="cannot exceed max_mask_percent"):
            validate_faithfulness_config(cfg)

    def test_invalid_baseline_mode_raises(self):
        cfg = {
            "experiment_name": "faith_test",
            "baseline_modes": ["noise_not_supported"],
        }
        with pytest.raises(ValueError, match="Invalid baseline mode"):
            validate_faithfulness_config(cfg)

    def test_even_blur_kernel_size_raises(self):
        cfg = {
            "experiment_name": "faith_test",
            "baseline_modes": ["blur"],
            "blur_kernel_size": 10,  # Must be odd
        }
        with pytest.raises(ValueError, match="blur_kernel_size must be a positive odd integer"):
            validate_faithfulness_config(cfg)


class TestSanityChecksConfigValidation:
    """Tests boundary assertions for parameter randomization sanity checks."""

    def test_valid_sanity_config(self):
        cfg = {
            "experiment_name": "sanity_test",
            "randomize_modes": ["classifier", "all"],
            "correlation_metrics": ["spearman", "pearson"],
            "threshold_drop_percent": 75.0,
        }
        res = validate_sanity_checks_config(cfg)
        assert res["threshold_drop_percent"] == 75.0

    def test_missing_experiment_name_standalone_raises(self):
        cfg = {"randomize_modes": ["all"]}
        with pytest.raises(ValueError, match="missing mandatory key: 'experiment_name'"):
            validate_sanity_checks_config(cfg, standalone=True)

    def test_invalid_randomize_mode_raises(self):
        cfg = {"experiment_name": "t", "randomize_modes": ["unknown_mode"]}
        with pytest.raises(ValueError, match="Invalid randomize_mode"):
            validate_sanity_checks_config(cfg)

    def test_invalid_threshold_drop_raises(self):
        cfg = {"experiment_name": "t", "threshold_drop_percent": 150.0}
        with pytest.raises(ValueError, match="threshold_drop_percent must be in range"):
            validate_sanity_checks_config(cfg)


class TestLocalizationConfigValidation:
    """Tests boundary assertions for localization and shortcut configs."""

    def test_valid_localization_config(self):
        cfg = {
            "experiment_name": "loc_test",
            "iou_thresholds": [0.1, 0.3, 0.5],
            "binarize_thresholds": [0.3, 0.5, 0.7],
            "shortcut_analysis": {
                "border_margin_fraction": 0.05,
                "corner_margin_fraction": 0.08,
            },
        }
        res = validate_localization_config(cfg)
        assert len(res["iou_thresholds"]) == 3

    def test_invalid_iou_threshold_raises(self):
        cfg = {"experiment_name": "t", "iou_thresholds": [1.5]}
        with pytest.raises(ValueError, match="Each iou_threshold must be in range"):
            validate_localization_config(cfg)

    def test_invalid_binarize_threshold_raises(self):
        cfg = {"experiment_name": "t", "binarize_thresholds": [0.0]}
        with pytest.raises(ValueError, match="Each binarize_threshold must be in range"):
            validate_localization_config(cfg)

    def test_invalid_border_fraction_raises(self):
        cfg = {"experiment_name": "t", "shortcut_analysis": {"border_margin_fraction": 0.6}}
        with pytest.raises(ValueError, match="border_margin_fraction must be in range"):
            validate_localization_config(cfg)
