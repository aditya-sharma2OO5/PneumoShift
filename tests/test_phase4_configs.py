"""Unit tests for Phase 4 configuration files, schema validations, and parameter assertions."""

from pathlib import Path
import pytest

from pneumoshift.evaluation.config_schemas import (
    VALID_ARCHITECTURES,
    VALID_DATASET_NAMES,
    VALID_OPTIMIZERS,
    VALID_SHIFT_DIRECTIONS,
    VALID_THRESHOLD_METRICS,
    VALID_UNCERTAINTY_POLICIES,
    validate_arch_comparison_config,
    validate_model_config,
    validate_train_config,
    validate_uncertainty_ablation_config,
)
from pneumoshift.utils.io import load_yaml_config


class TestModelConfigs:
    """Tests for model backbone YAML configurations and schema validation."""

    def test_efficientnet_b0_config_valid(self):
        cfg = load_yaml_config("configs/model/efficientnet_b0.yaml")
        validated = validate_model_config(cfg)
        assert validated["architecture"] == "efficientnet_b0"
        assert validated["in_features"] == 1280
        assert validated["dropout"] == 0.2
        assert validated["channels"] == 3
        assert validated["num_classes"] == 8

    def test_densenet121_config_valid(self):
        cfg = load_yaml_config("configs/model/densenet121.yaml")
        validated = validate_model_config(cfg)
        assert validated["architecture"] == "densenet121"
        assert validated["feature_dim"] == 1024
        assert validated["dropout"] == 0.2

    def test_invalid_architecture_raises(self):
        cfg = {"architecture": "resnet50", "num_classes": 8}
        with pytest.raises(ValueError, match="Invalid architecture"):
            validate_model_config(cfg)

    def test_missing_architecture_raises(self):
        cfg = {"num_classes": 8, "dropout": 0.2}
        with pytest.raises(ValueError, match="missing mandatory key: 'architecture'"):
            validate_model_config(cfg)

    def test_invalid_dropout_raises(self):
        cfg = {"architecture": "efficientnet_b0", "dropout": 1.5}
        with pytest.raises(ValueError, match="dropout must be in range"):
            validate_model_config(cfg)

        cfg_neg = {"architecture": "efficientnet_b0", "dropout": -0.1}
        with pytest.raises(ValueError, match="dropout must be in range"):
            validate_model_config(cfg_neg)


class TestTrainingConfigs:
    """Tests for training YAML configurations and parameter assertions."""

    def test_nih_efficientnet_b0_config_valid(self):
        cfg = load_yaml_config("configs/train/nih_efficientnet_b0.yaml")
        validated = validate_train_config(cfg)
        assert validated["data"]["dataset_name"] == "nih"
        assert validated["model"]["architecture"] == "efficientnet_b0"
        assert validated["data"]["uncertainty_policy"] == "binary"
        assert validated["train"]["optimizer"] == "adamw"

    def test_chexpert_efficientnet_b0_config_valid(self):
        cfg = load_yaml_config("configs/train/chexpert_efficientnet_b0.yaml")
        validated = validate_train_config(cfg)
        assert validated["data"]["dataset_name"] == "chexpert"
        assert validated["model"]["architecture"] == "efficientnet_b0"
        assert validated["data"]["uncertainty_policy"] == "u_zeros"

    def test_chexpert_densenet121_uones_config_valid(self):
        cfg = load_yaml_config("configs/train/chexpert_densenet121_uones.yaml")
        validated = validate_train_config(cfg)
        assert validated["data"]["dataset_name"] == "chexpert"
        assert validated["model"]["architecture"] == "densenet121"
        assert validated["data"]["uncertainty_policy"] == "u_ones"

    def test_chexpert_densenet121_uignore_config_valid(self):
        cfg = load_yaml_config("configs/train/chexpert_densenet121_uignore.yaml")
        validated = validate_train_config(cfg)
        assert validated["data"]["dataset_name"] == "chexpert"
        assert validated["model"]["architecture"] == "densenet121"
        assert validated["data"]["uncertainty_policy"] == "u_ignore"

    def test_missing_train_section_raises(self):
        cfg = {"experiment_name": "test", "data": {}, "model": {}}
        with pytest.raises(ValueError, match="Training config missing mandatory section"):
            validate_train_config(cfg)

    def test_invalid_dataset_name_raises(self):
        cfg = load_yaml_config("configs/train/nih_efficientnet_b0.yaml")
        cfg["data"]["dataset_name"] = "mimic_cxr"
        with pytest.raises(ValueError, match="Invalid dataset_name"):
            validate_train_config(cfg)

    def test_invalid_uncertainty_policy_raises(self):
        cfg = load_yaml_config("configs/train/nih_efficientnet_b0.yaml")
        cfg["data"]["uncertainty_policy"] = "u_random"
        with pytest.raises(ValueError, match="Invalid uncertainty_policy"):
            validate_train_config(cfg)

    def test_invalid_learning_rate_raises(self):
        cfg = load_yaml_config("configs/train/nih_efficientnet_b0.yaml")
        cfg["train"]["learning_rate"] = -0.001
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_train_config(cfg)


class TestArchitectureComparisonConfig:
    """Tests for architecture robustness comparison YAML configuration."""

    def test_arch_comparison_config_valid(self):
        cfg = load_yaml_config("configs/eval/arch_comparison.yaml")
        validated = validate_arch_comparison_config(cfg)
        arch_names = [a["name"] for a in validated["architectures"]]
        assert "densenet121" in arch_names
        assert "efficientnet_b0" in arch_names
        assert "nih_to_chexpert" in validated["shift_directions"]
        assert "chexpert_to_nih" in validated["shift_directions"]
        assert validated["eval_protocol"]["confidence_level"] == 0.95

    def test_fewer_than_two_archs_raises(self):
        cfg = load_yaml_config("configs/eval/arch_comparison.yaml")
        cfg["architectures"] = [{"name": "densenet121"}]
        with pytest.raises(ValueError, match="at least 2 architectures to compare"):
            validate_arch_comparison_config(cfg)

    def test_duplicate_architectures_raises(self):
        cfg = load_yaml_config("configs/eval/arch_comparison.yaml")
        cfg["architectures"] = [{"name": "densenet121"}, {"name": "densenet121"}]
        with pytest.raises(ValueError, match="Duplicate architecture specified"):
            validate_arch_comparison_config(cfg)

    def test_invalid_shift_direction_raises(self):
        cfg = load_yaml_config("configs/eval/arch_comparison.yaml")
        cfg["shift_directions"] = ["nih_to_mimic"]
        with pytest.raises(ValueError, match="Invalid shift_direction"):
            validate_arch_comparison_config(cfg)

    def test_invalid_confidence_level_raises(self):
        cfg = load_yaml_config("configs/eval/arch_comparison.yaml")
        cfg["eval_protocol"]["confidence_level"] = 0.2
        with pytest.raises(ValueError, match="confidence_level must be in"):
            validate_arch_comparison_config(cfg)


class TestUncertaintyAblationConfig:
    """Tests for CheXpert uncertainty policy ablation YAML configuration."""

    def test_uncertainty_ablation_config_valid(self):
        cfg = load_yaml_config("configs/eval/uncertainty_ablation.yaml")
        validated = validate_uncertainty_ablation_config(cfg)
        policy_names = [p["name"] for p in validated["policies"]]
        assert "u_zeros" in policy_names
        assert "u_ones" in policy_names
        assert "u_ignore" in policy_names
        assert validated["target_data"]["dataset_name"] == "nih"
        assert "youden" in validated["eval_protocol"]["threshold_metrics"]

    def test_fewer_than_two_policies_raises(self):
        cfg = load_yaml_config("configs/eval/uncertainty_ablation.yaml")
        cfg["policies"] = [{"name": "u_zeros"}]
        with pytest.raises(ValueError, match="at least 2 uncertainty policies to compare"):
            validate_uncertainty_ablation_config(cfg)

    def test_invalid_policy_name_raises(self):
        cfg = load_yaml_config("configs/eval/uncertainty_ablation.yaml")
        cfg["policies"] = [{"name": "u_zeros"}, {"name": "u_multiclass"}]
        with pytest.raises(ValueError, match="Policy 'u_multiclass' is not supported"):
            validate_uncertainty_ablation_config(cfg)

    def test_duplicate_policy_raises(self):
        cfg = load_yaml_config("configs/eval/uncertainty_ablation.yaml")
        cfg["policies"] = [{"name": "u_zeros"}, {"name": "u_zeros"}]
        with pytest.raises(ValueError, match="Duplicate policy specified"):
            validate_uncertainty_ablation_config(cfg)

    def test_invalid_threshold_metric_raises(self):
        cfg = load_yaml_config("configs/eval/uncertainty_ablation.yaml")
        cfg["eval_protocol"]["threshold_metrics"] = ["invalid_metric"]
        with pytest.raises(ValueError, match="Invalid threshold metric"):
            validate_uncertainty_ablation_config(cfg)

