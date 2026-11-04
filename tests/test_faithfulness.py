"""Unit tests for quantitative faithfulness evaluation (deletion/insertion dynamics)."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from pneumoshift.models.factory import build_model
from pneumoshift.xai.faithfulness import (
    compute_deletion_insertion_curves,
    evaluate_cohort_faithfulness,
    compare_cross_domain_faithfulness,
    format_faithfulness_markdown_table,
    save_faithfulness_report,
)


@pytest.fixture
def dummy_model():
    """A lightweight mock model returning predictable outputs."""
    model = nn.Sequential(
        nn.Conv2d(3, 8, kernel_size=3, padding=1),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(8, 8),
    )
    model.eval()
    return model


class TestDeletionInsertionBaselines:
    """Verifies single-image deletion/insertion curve calculation across baseline imputation modes."""

    def test_zero_baseline_curve_generation(self, dummy_model):
        img = torch.randn(1, 3, 32, 32)
        cam = np.random.rand(32, 32).astype(np.float32)

        res = compute_deletion_insertion_curves(
            model=dummy_model,
            image_tensor=img,
            saliency_map=cam,
            target_class_idx=5,
            num_steps=10,
            max_mask_percent=50.0,
            baseline_mode="zero",
        )

        assert "percentages" in res
        assert len(res["percentages"]) == 11
        assert res["percentages"][0] == 0.0
        assert res["percentages"][-1] == 50.0

        assert len(res["deletion_curve"]) == 11
        assert len(res["insertion_curve"]) == 11
        assert 0.0 <= res["deletion_auc"] <= 1.0
        assert 0.0 <= res["insertion_auc"] <= 1.0
        assert res["faithfulness_ratio"] >= 0.0
        assert res["baseline_mode"] == "zero"

    def test_blur_baseline_mode(self, dummy_model):
        img = torch.randn(1, 3, 32, 32)
        cam = np.random.rand(32, 32).astype(np.float32)

        res = compute_deletion_insertion_curves(
            model=dummy_model,
            image_tensor=img,
            saliency_map=cam,
            target_class_idx=5,
            num_steps=5,
            max_mask_percent=30.0,
            baseline_mode="blur",
            blur_kernel_size=5,
            blur_sigma=2.0,
        )

        assert len(res["percentages"]) == 6
        assert res["baseline_mode"] == "blur"
        assert res["deletion_auc"] >= 0.0

    def test_mean_baseline_mode(self, dummy_model):
        img = torch.randn(3, 32, 32)  # 3D tensor without batch dim
        cam = np.random.rand(32, 32).astype(np.float32)

        res = compute_deletion_insertion_curves(
            model=dummy_model,
            image_tensor=img,
            saliency_map=cam,
            target_class_idx=5,
            num_steps=4,
            max_mask_percent=40.0,
            baseline_mode="mean",
        )

        assert len(res["percentages"]) == 5
        assert res["baseline_mode"] == "mean"
        assert res["insertion_auc"] >= 0.0

    def test_invalid_baseline_raises(self, dummy_model):
        img = torch.randn(1, 3, 32, 32)
        cam = np.random.rand(32, 32).astype(np.float32)

        with pytest.raises(ValueError, match="Unsupported baseline mode"):
            compute_deletion_insertion_curves(
                model=dummy_model,
                image_tensor=img,
                saliency_map=cam,
                baseline_mode="unknown_mode",
            )


class TestCohortFaithfulness:
    """Verifies cohort-level aggregated statistics and curve averaging."""

    def test_evaluate_cohort_statistics(self, dummy_model):
        n = 4
        images = [torch.randn(1, 3, 24, 24) for _ in range(n)]
        cams = [np.random.rand(24, 24).astype(np.float32) for _ in range(n)]

        cohort_res = evaluate_cohort_faithfulness(
            model=dummy_model,
            images=images,
            cams=cams,
            target_class_idx=5,
            num_steps=5,
            max_mask_percent=50.0,
            baseline_mode="zero",
        )

        assert cohort_res["num_samples"] == n
        assert cohort_res["mean_deletion_auc"] >= 0.0
        assert cohort_res["mean_insertion_auc"] >= 0.0
        assert cohort_res["mean_faithfulness_ratio"] >= 0.0
        assert len(cohort_res["avg_deletion_curve"]) == 6
        assert len(cohort_res["avg_insertion_curve"]) == 6


class TestCrossDomainComparisonAndReporting:
    """Tests cross-domain degradation calculation and publication report emission."""

    def test_compare_cross_domain_metrics(self):
        in_domain = {
            "mean_deletion_auc": 0.25,
            "mean_insertion_auc": 0.70,
            "mean_faithfulness_ratio": 2.80,
            "baseline_mode": "zero",
        }
        cross_domain = {
            "mean_deletion_auc": 0.40,  # Flatter deletion (worse)
            "mean_insertion_auc": 0.55,  # Slower recovery (worse)
            "mean_faithfulness_ratio": 1.375,
            "baseline_mode": "zero",
        }

        comp = compare_cross_domain_faithfulness(
            in_domain_results=in_domain,
            cross_domain_results=cross_domain,
            model_name="DenseNet121",
            shift_name="NIH_to_CheXpert",
        )

        assert comp["model_name"] == "DenseNet121"
        assert comp["deltas"]["delta_deletion_auc"] == pytest.approx(0.15)
        assert comp["deltas"]["delta_insertion_auc"] == pytest.approx(-0.15)
        assert comp["deltas"]["delta_faithfulness_ratio"] < 0.0
        assert comp["deltas"]["faithfulness_degradation_percent"] > 0.0

    def test_markdown_and_json_report_export(self, tmp_path: Path):
        comparison = {
            "model_name": "EfficientNet-B0",
            "shift_name": "CheXpert_to_NIH",
            "baseline_mode": "blur",
            "in_domain": {"deletion_auc": 0.22, "insertion_auc": 0.68, "faithfulness_ratio": 3.09},
            "cross_domain": {"deletion_auc": 0.35, "insertion_auc": 0.52, "faithfulness_ratio": 1.48},
            "deltas": {
                "delta_deletion_auc": 0.13,
                "delta_insertion_auc": -0.16,
                "delta_faithfulness_ratio": -1.61,
                "faithfulness_degradation_percent": 52.1,
            },
        }

        md_text = format_faithfulness_markdown_table(comparison)
        assert "| **EfficientNet-B0** |" in md_text
        assert "52.1%" in md_text

        json_out, md_out = save_faithfulness_report(comparison, output_dir=tmp_path / "xai")
        assert json_out.is_file()
        assert md_out.is_file()
        assert md_out.read_text(encoding="utf-8") == md_text

