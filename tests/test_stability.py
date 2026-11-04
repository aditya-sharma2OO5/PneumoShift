"""Unit tests for attribution stability, perturbation invariance, and inter-seed reproducibility."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from pneumoshift.models.factory import build_model, get_target_layer_for_xai
from pneumoshift.utils.io import load_json
from pneumoshift.xai.gradcam import GradCAMExplainer
from pneumoshift.xai.stability import (
    compute_binarized_iou,
    compute_similarity_metrics,
    compute_attribution_stability,
    compare_inter_seed_stability,
    compare_inter_architecture_stability,
    evaluate_cohort_stability,
    format_stability_markdown_table,
    save_stability_report,
)


@pytest.fixture
def densenet_model():
    model = build_model(architecture="densenet121", num_classes=8, pretrained=False)
    model.eval()
    return model


@pytest.fixture
def efficientnet_model():
    model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
    model.eval()
    return model


@pytest.fixture
def densenet_explainer(densenet_model):
    target_layer = get_target_layer_for_xai(densenet_model, "densenet121")
    explainer = GradCAMExplainer(densenet_model, target_layer)
    yield explainer
    explainer.release_hooks()


@pytest.fixture
def efficientnet_explainer(efficientnet_model):
    target_layer = get_target_layer_for_xai(efficientnet_model, "efficientnet_b0")
    explainer = GradCAMExplainer(efficientnet_model, target_layer)
    yield explainer
    explainer.release_hooks()


class TestBinarizedIoUAndMetrics:
    """Verifies IoU computation, map resizing, and similarity metric calculations."""

    def test_binarized_iou_identical_maps(self):
        m = np.zeros((32, 32), dtype=np.float32)
        m[10:20, 10:20] = 0.8
        iou = compute_binarized_iou(m, m, threshold=0.5)
        assert iou == pytest.approx(1.0)

    def test_binarized_iou_disjoint_maps(self):
        m1 = np.zeros((32, 32), dtype=np.float32)
        m2 = np.zeros((32, 32), dtype=np.float32)
        m1[0:10, 0:10] = 0.9
        m2[20:30, 20:30] = 0.9
        iou = compute_binarized_iou(m1, m2, threshold=0.5)
        assert iou == pytest.approx(0.0)

    def test_binarized_iou_inactive_maps(self):
        m1 = np.zeros((32, 32), dtype=np.float32)
        m2 = np.zeros((32, 32), dtype=np.float32)
        iou = compute_binarized_iou(m1, m2, threshold=0.5)
        assert iou == pytest.approx(1.0)

    def test_binarized_iou_different_resolutions(self):
        m1 = np.zeros((32, 32), dtype=np.float32)
        m2 = np.zeros((64, 64), dtype=np.float32)
        m1[10:20, 10:20] = 0.8
        m2[20:40, 20:40] = 0.8
        iou = compute_binarized_iou(m1, m2, threshold=0.5)
        assert 0.8 <= iou <= 1.0

    def test_similarity_metrics_identical(self):
        m = np.random.rand(32, 32).astype(np.float32)
        metrics = compute_similarity_metrics(m, m, binarize_threshold=0.5)
        assert metrics["spearman"] == pytest.approx(1.0)
        assert metrics["pearson"] == pytest.approx(1.0)
        assert metrics["cosine_similarity"] == pytest.approx(1.0)
        assert metrics["mse"] == pytest.approx(0.0)
        assert metrics["iou"] == pytest.approx(1.0)

    def test_similarity_metrics_constant_zero(self):
        z = np.zeros((32, 32), dtype=np.float32)
        r = np.random.rand(32, 32).astype(np.float32)
        metrics = compute_similarity_metrics(z, r, binarize_threshold=0.5)
        assert metrics["spearman"] == pytest.approx(0.0)
        assert metrics["pearson"] == pytest.approx(0.0)
        assert metrics["cosine_similarity"] == pytest.approx(0.0)


class TestPerturbationStability:
    """Verifies geometric and photometric perturbation invariance evaluation."""

    def test_attribution_stability_basic(self, densenet_explainer):
        dummy_input = torch.randn(1, 3, 64, 64)
        res = compute_attribution_stability(
            explainer=densenet_explainer,
            image_tensor=dummy_input,
            target_class_idx=5,
            num_perturbations=3,
            max_rotation_deg=3.0,
            brightness_factor=0.03,
            translation_percent=0.03,
        )

        assert "mean_spearman" in res
        assert "mean_perturbation_stability" in res
        assert "std_spearman" in res
        assert "mean_pearson" in res
        assert "mean_cosine" in res
        assert "mean_iou" in res
        assert "mean_mse" in res
        assert len(res["trials"]) == 3

        # Stability correlations must be in [-1, 1]
        assert -1.0 <= res["mean_spearman"] <= 1.0
        assert -1.0 <= res["mean_pearson"] <= 1.0
        assert 0.0 <= res["mean_iou"] <= 1.0

    def test_attribution_stability_zero_perturbations(self, densenet_explainer):
        dummy_input = torch.randn(1, 3, 64, 64)
        res = compute_attribution_stability(
            explainer=densenet_explainer,
            image_tensor=dummy_input,
            num_perturbations=0,
        )
        assert res["mean_spearman"] == 1.0
        assert res["std_spearman"] == 0.0
        assert res["trials"] == []

    def test_attribution_stability_3d_input(self, densenet_explainer):
        # Shape (C, H, W)
        dummy_input = torch.randn(3, 64, 64)
        res = compute_attribution_stability(
            explainer=densenet_explainer,
            image_tensor=dummy_input,
            num_perturbations=2,
        )
        assert len(res["trials"]) == 2


class TestInterSeedStability:
    """Verifies reproducibility across models trained with different random seeds."""

    def test_inter_seed_stability_multiple_seeds(self, densenet_model):
        # Simulate seed models with the fixture and a clone
        model_seed_42 = densenet_model
        model_seed_1337 = build_model(architecture="densenet121", num_classes=8, pretrained=False)
        model_seed_1337.eval()

        models_by_seed = {
            42: model_seed_42,
            1337: model_seed_1337,
        }

        dummy_input = torch.randn(1, 3, 64, 64)
        res = compare_inter_seed_stability(
            models_or_explainers=models_by_seed,
            image_tensor=dummy_input,
            architecture="densenet121",
            target_class_idx=5,
        )

        assert "seeds" in res
        assert res["seeds"] == ["42", "1337"]
        assert "seed_42_vs_seed_1337" in res["pairwise_comparisons"]
        assert -1.0 <= res["mean_inter_seed_spearman"] <= 1.0
        assert 0.0 <= res["mean_inter_seed_iou"] <= 1.0

    def test_inter_seed_stability_single_seed_graceful(self, densenet_explainer):
        dummy_input = torch.randn(1, 3, 64, 64)
        res = compare_inter_seed_stability(
            models_or_explainers={42: densenet_explainer},
            image_tensor=dummy_input,
        )
        assert res["mean_inter_seed_spearman"] == 1.0
        assert res["pairwise_comparisons"] == {}

    def test_inter_seed_stability_with_explainers(self, densenet_explainer):
        dummy_input = torch.randn(1, 3, 64, 64)
        res = compare_inter_seed_stability(
            models_or_explainers={"seed_a": densenet_explainer, "seed_b": densenet_explainer},
            image_tensor=dummy_input,
        )
        assert "seed_seed_a_vs_seed_seed_b" in res["pairwise_comparisons"]
        # Comparing with itself should yield perfect correlation
        assert res["mean_inter_seed_spearman"] == pytest.approx(1.0)
        assert res["mean_inter_seed_iou"] == pytest.approx(1.0)


class TestInterArchitectureStability:
    """Verifies attribution concordance between DenseNet121 and EfficientNet-B0."""

    def test_inter_architecture_stability_densenet_vs_efficientnet(self, densenet_model, efficientnet_model):
        dummy_input = torch.randn(1, 3, 64, 64)
        res = compare_inter_architecture_stability(
            model_or_explainer_a=densenet_model,
            model_or_explainer_b=efficientnet_model,
            arch_a="densenet121",
            arch_b="efficientnet_b0",
            image_tensor=dummy_input,
            target_class_idx=5,
        )

        assert res["architecture_a"] == "densenet121"
        assert res["architecture_b"] == "efficientnet_b0"
        assert -1.0 <= res["spearman"] <= 1.0
        assert -1.0 <= res["pearson"] <= 1.0
        assert 0.0 <= res["iou"] <= 1.0
        assert res["mse"] >= 0.0

    def test_inter_architecture_stability_none_input_raises(self, densenet_model, efficientnet_model):
        with pytest.raises(ValueError, match="image_tensor cannot be None"):
            compare_inter_architecture_stability(
                model_or_explainer_a=densenet_model,
                model_or_explainer_b=efficientnet_model,
                image_tensor=None,
            )


class TestCohortStabilityEvaluation:
    """Verifies cohort-level stability evaluation across multiple images."""

    def test_evaluate_cohort_stability(self, densenet_explainer):
        images = [torch.randn(1, 3, 64, 64) for _ in range(3)]
        models = {"DenseNet121": densenet_explainer}

        res = evaluate_cohort_stability(
            models=models,
            images=images,
            target_class_idx=5,
            num_perturbations=2,
        )

        assert res["num_cohort_samples"] == 3
        assert "DenseNet121" in res["perturbation_stability"]
        dn_metrics = res["perturbation_stability"]["DenseNet121"]
        assert "mean_spearman" in dn_metrics
        assert "mean_iou" in dn_metrics

    def test_evaluate_cohort_empty_images_raises(self, densenet_explainer):
        with pytest.raises(ValueError, match="Cohort images list cannot be empty"):
            evaluate_cohort_stability(
                models={"DenseNet121": densenet_explainer},
                images=[],
            )


class TestStabilityReporting:
    """Verifies formatting and export of Markdown and JSON stability reports."""

    def test_format_stability_markdown_table(self):
        sample_results = {
            "perturbation_stability": {
                "DenseNet121": {
                    "mean_spearman": 0.884,
                    "std_spearman": 0.031,
                    "mean_pearson": 0.902,
                    "std_pearson": 0.025,
                    "mean_cosine": 0.941,
                    "std_cosine": 0.018,
                    "mean_iou": 0.725,
                    "std_iou": 0.042,
                    "mean_mse": 0.012345,
                }
            },
            "inter_seed_reproducibility": {
                "seeds": [42, 1337, 2024],
                "cohort_mean_spearman": 0.822,
                "cohort_std_spearman": 0.008,
                "cohort_mean_pearson": 0.842,
                "cohort_std_pearson": 0.006,
                "cohort_mean_cosine": 0.898,
                "cohort_std_cosine": 0.005,
                "cohort_mean_iou": 0.665,
                "cohort_std_iou": 0.008,
                "cohort_mean_mse": 0.021,
                "cohort_std_mse": 0.002,
                "pairwise_comparisons": {
                    "seed_42_vs_seed_1337": {
                        "spearman": 0.812,
                        "pearson": 0.835,
                        "cosine_similarity": 0.892,
                        "iou": 0.654,
                        "mse": 0.022,
                    }
                },
            },
            "inter_architecture_consistency": {
                "architecture_a": "densenet121",
                "architecture_b": "efficientnet_b0",
                "spearman": 0.745,
                "pearson": 0.772,
                "cosine_similarity": 0.845,
                "iou": 0.589,
                "mse": 0.035,
            },
        }

        md = format_stability_markdown_table(sample_results)
        assert "# Attribution Stability & Inter-Seed Reproducibility Analysis" in md
        assert "## 1. Geometric & Photometric Perturbation Invariance" in md
        assert "`DenseNet121`" in md
        assert "## 2. Multi-Seed Attribution Reproducibility" in md
        assert "seed_42_vs_seed_1337" in md
        assert "## 3. Inter-Architecture Attribution Consistency" in md

    def test_save_stability_report_tmp_path(self, tmp_path):
        sample_results = {
            "perturbation_stability": {
                "DenseNet121": {
                    "mean_spearman": 0.85,
                    "std_spearman": 0.02,
                    "mean_pearson": 0.87,
                    "std_pearson": 0.02,
                    "mean_cosine": 0.92,
                    "std_cosine": 0.01,
                    "mean_iou": 0.70,
                    "std_iou": 0.03,
                    "mean_mse": 0.015,
                }
            }
        }

        json_out = tmp_path / "test_stability.json"
        saved_json, saved_md = save_stability_report(sample_results, json_out)

        assert saved_json.is_file()
        assert saved_md.is_file()

        loaded = load_json(saved_json)
        assert "perturbation_stability" in loaded
        assert "DenseNet121" in loaded["perturbation_stability"]

        md_text = saved_md.read_text(encoding="utf-8")
        assert "Perturbation Invariance" in md_text

