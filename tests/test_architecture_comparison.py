"""Unit tests for cross-dataset architecture robustness comparison engine (DenseNet vs EfficientNet)."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.architecture_comparison import (
    compare_architecture_calibration,
    compare_architecture_domain_divergence,
    compare_architecture_generalization_gaps,
    generate_architecture_comparison_tables,
    run_architecture_comparison_suite,
)


@pytest.fixture
def mock_densenet_in_domain() -> dict:
    return {
        "macro_auroc": 0.85,
        "per_class": {
            "Pneumonia": {"auroc": 0.82, "auprc": 0.45},
            "Atelectasis": {"auroc": 0.84, "auprc": 0.50},
            "Cardiomegaly": {"auroc": 0.90, "auprc": 0.65},
            "Consolidation": {"auroc": 0.81, "auprc": 0.40},
            "Edema": {"auroc": 0.88, "auprc": 0.55},
            "Pleural Effusion": {"auroc": 0.89, "auprc": 0.60},
            "Pneumothorax": {"auroc": 0.86, "auprc": 0.48},
            "No Finding": {"auroc": 0.80, "auprc": 0.70},
        },
        "ece_uniform": 0.045,
        "mce": 0.120,
        "brier_score": 0.080,
    }


@pytest.fixture
def mock_densenet_cross_domain() -> dict:
    # Severe drop on cross-domain
    return {
        "macro_auroc": 0.75,
        "per_class": {
            "Pneumonia": {"auroc": 0.70, "auprc": 0.30},  # Gap: 0.12
            "Atelectasis": {"auroc": 0.72, "auprc": 0.35},  # Gap: 0.12
            "Cardiomegaly": {"auroc": 0.82, "auprc": 0.50},  # Gap: 0.08
            "Consolidation": {"auroc": 0.71, "auprc": 0.28},  # Gap: 0.10
            "Edema": {"auroc": 0.80, "auprc": 0.42},  # Gap: 0.08
            "Pleural Effusion": {"auroc": 0.81, "auprc": 0.45},  # Gap: 0.08
            "Pneumothorax": {"auroc": 0.76, "auprc": 0.32},  # Gap: 0.10
            "No Finding": {"auroc": 0.68, "auprc": 0.50},  # Gap: 0.12
        },
        "ece_uniform": 0.110,
        "mce": 0.240,
        "brier_score": 0.145,
    }


@pytest.fixture
def mock_efficientnet_in_domain() -> dict:
    return {
        "macro_auroc": 0.84,
        "per_class": {
            "Pneumonia": {"auroc": 0.81, "auprc": 0.43},
            "Atelectasis": {"auroc": 0.83, "auprc": 0.48},
            "Cardiomegaly": {"auroc": 0.89, "auprc": 0.62},
            "Consolidation": {"auroc": 0.80, "auprc": 0.38},
            "Edema": {"auroc": 0.87, "auprc": 0.53},
            "Pleural Effusion": {"auroc": 0.88, "auprc": 0.58},
            "Pneumothorax": {"auroc": 0.85, "auprc": 0.46},
            "No Finding": {"auroc": 0.79, "auprc": 0.68},
        },
        "ece_uniform": 0.040,
        "mce": 0.115,
        "brier_score": 0.078,
    }


@pytest.fixture
def mock_efficientnet_cross_domain() -> dict:
    # EfficientNet retains more performance under shift (narrower gap)
    return {
        "macro_auroc": 0.79,
        "per_class": {
            "Pneumonia": {"auroc": 0.75, "auprc": 0.36},  # Gap: 0.06 (vs DN 0.12)
            "Atelectasis": {"auroc": 0.77, "auprc": 0.40},  # Gap: 0.06 (vs DN 0.12)
            "Cardiomegaly": {"auroc": 0.85, "auprc": 0.55},  # Gap: 0.04 (vs DN 0.08)
            "Consolidation": {"auroc": 0.75, "auprc": 0.32},  # Gap: 0.05 (vs DN 0.10)
            "Edema": {"auroc": 0.83, "auprc": 0.48},  # Gap: 0.04 (vs DN 0.08)
            "Pleural Effusion": {"auroc": 0.84, "auprc": 0.51},  # Gap: 0.04 (vs DN 0.08)
            "Pneumothorax": {"auroc": 0.80, "auprc": 0.39},  # Gap: 0.05 (vs DN 0.10)
            "No Finding": {"auroc": 0.73, "auprc": 0.58},  # Gap: 0.06 (vs DN 0.12)
        },
        "ece_uniform": 0.080,
        "mce": 0.180,
        "brier_score": 0.110,
    }


class TestArchitectureGapComparison:
    """Verifies generalization gap arithmetic and robustness winner determination."""

    def test_gap_comparison_identifies_narrower_gap(
        self,
        mock_densenet_in_domain,
        mock_densenet_cross_domain,
        mock_efficientnet_in_domain,
        mock_efficientnet_cross_domain,
    ):
        report = compare_architecture_generalization_gaps(
            densenet_in_domain=mock_densenet_in_domain,
            densenet_cross_domain=mock_densenet_cross_domain,
            efficientnet_in_domain=mock_efficientnet_in_domain,
            efficientnet_cross_domain=mock_efficientnet_cross_domain,
            shift_direction="nih_to_chexpert",
        )

        assert report["shift_direction"] == "nih_to_chexpert"
        assert report["macro_densenet_gap_auroc"] is not None
        assert report["macro_efficientnet_gap_auroc"] is not None
        assert report["macro_densenet_gap_auroc"] > report["macro_efficientnet_gap_auroc"]
        assert report["overall_robust_architecture"] == "EfficientNet-B0"

        # Check pneumonia class gap
        pneu = report["per_class"]["Pneumonia"]
        assert pytest.approx(pneu["densenet_gap_auroc"], abs=1e-3) == 0.12
        assert pytest.approx(pneu["efficientnet_gap_auroc"], abs=1e-3) == 0.06
        assert pytest.approx(pneu["gap_difference"], abs=1e-3) == 0.06
        assert pneu["more_robust_architecture"] == "EfficientNet-B0"


class TestArchitectureCalibrationComparison:
    """Verifies calibration degradation comparison between architectures."""

    def test_calibration_degradation_metrics(
        self,
        mock_densenet_in_domain,
        mock_densenet_cross_domain,
        mock_efficientnet_in_domain,
        mock_efficientnet_cross_domain,
    ):
        cal_report = compare_architecture_calibration(
            densenet_in_domain=mock_densenet_in_domain,
            densenet_cross_domain=mock_densenet_cross_domain,
            efficientnet_in_domain=mock_efficientnet_in_domain,
            efficientnet_cross_domain=mock_efficientnet_cross_domain,
        )

        dn = cal_report["densenet121"]
        eff = cal_report["efficientnet_b0"]

        assert pytest.approx(dn["ece_degradation"], abs=1e-3) == 0.065
        assert pytest.approx(eff["ece_degradation"], abs=1e-3) == 0.040
        assert dn["cross_domain_brier"] > eff["cross_domain_brier"]


class TestArchitectureFeatureSpaceProbing:
    """Verifies domain probe and MMD on penultimate embeddings (D=1024 vs D=1280)."""

    def test_probe_and_mmd_comparison(self):
        np.random.seed(42)
        # DenseNet: D=1024
        dn_source = np.random.randn(30, 1024)
        dn_target = np.random.randn(30, 1024) + 1.0  # Shifted

        # EfficientNet: D=1280
        eff_source = np.random.randn(30, 1280)
        eff_target = np.random.randn(30, 1280) + 0.3  # Less shifted

        div_report = compare_architecture_domain_divergence(
            densenet_source_features=dn_source,
            densenet_target_features=dn_target,
            efficientnet_source_features=eff_source,
            efficientnet_target_features=eff_target,
            seed=42,
        )

        assert div_report["densenet121"]["feature_dim"] == 1024
        assert div_report["efficientnet_b0"]["feature_dim"] == 1280
        assert "domain_probe_auroc" in div_report["densenet121"]
        assert "domain_probe_auroc" in div_report["efficientnet_b0"]
        assert "mmd_distance" in div_report["densenet121"]
        assert "mmd_distance" in div_report["efficientnet_b0"]
        assert div_report["less_separable_feature_space"] in ["EfficientNet-B0", "DenseNet121"]


class TestArchitectureComparisonSuite:
    """Verifies end-to-end execution, table markdown generation, and file saving."""

    def test_suite_execution_and_markdown(
        self,
        tmp_path: Path,
        mock_densenet_in_domain,
        mock_densenet_cross_domain,
        mock_efficientnet_in_domain,
        mock_efficientnet_cross_domain,
    ):
        np.random.seed(42)
        dn_src = np.random.randn(20, 1024)
        dn_tgt = np.random.randn(20, 1024)
        eff_src = np.random.randn(20, 1280)
        eff_tgt = np.random.randn(20, 1280)

        results = run_architecture_comparison_suite(
            densenet_in_domain=mock_densenet_in_domain,
            densenet_cross_domain=mock_densenet_cross_domain,
            efficientnet_in_domain=mock_efficientnet_in_domain,
            efficientnet_cross_domain=mock_efficientnet_cross_domain,
            shift_direction="nih_to_chexpert",
            densenet_source_features=dn_src,
            densenet_target_features=dn_tgt,
            efficientnet_source_features=eff_src,
            efficientnet_target_features=eff_tgt,
            output_dir=tmp_path,
        )

        assert "table_markdown" in results
        md = results["table_markdown"]
        assert "DenseNet121 vs. EfficientNet-B0" in md
        assert "Pneumonia" in md
        assert "Macro Average" in md
        assert "Penultimate Representation Space Domain Probing" in md

        # Verify disk outputs
        json_file = tmp_path / "architecture_comparison_nih_to_chexpert.json"
        md_file = tmp_path / "architecture_comparison_nih_to_chexpert.md"
        assert json_file.is_file()
        assert md_file.is_file()

