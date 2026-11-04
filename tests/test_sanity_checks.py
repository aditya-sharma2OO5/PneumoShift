"""Unit tests for model parameter randomization sanity checks (Adebayo et al., 2018)."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn

from pneumoshift.models.factory import build_model
from pneumoshift.xai.sanity_checks import (
    compute_map_similarity,
    randomize_model_weights,
    run_parameter_randomization_test,
    evaluate_cohort_sanity_checks,
    format_sanity_check_markdown_table,
    save_sanity_check_report,
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


class TestMapSimilarityMetrics:
    """Verifies correlation, distance, and edge-case handling for heatmap similarity."""

    def test_identical_maps_similarity(self):
        m = np.random.rand(32, 32).astype(np.float32)
        res = compute_map_similarity(m, m)

        assert res["spearman_rank_correlation"] == pytest.approx(1.0)
        assert res["pearson_correlation"] == pytest.approx(1.0)
        assert res["mean_squared_error"] == pytest.approx(0.0)
        assert res["cosine_similarity"] == pytest.approx(1.0)

    def test_distinct_maps_bounded(self):
        m1 = np.random.rand(32, 32).astype(np.float32)
        m2 = np.random.rand(32, 32).astype(np.float32)
        res = compute_map_similarity(m1, m2)

        assert -1.0 <= res["spearman_rank_correlation"] <= 1.0
        assert -1.0 <= res["pearson_correlation"] <= 1.0
        assert res["mean_squared_error"] >= 0.0

    def test_constant_zero_maps_handled_gracefully(self):
        zero_map = np.zeros((32, 32), dtype=np.float32)
        random_map = np.random.rand(32, 32).astype(np.float32)

        # One constant, one variable
        res1 = compute_map_similarity(zero_map, random_map)
        assert res1["spearman_rank_correlation"] == 0.0
        assert res1["pearson_correlation"] == 0.0

        # Both constant identical
        res2 = compute_map_similarity(zero_map, zero_map)
        assert res2["spearman_rank_correlation"] == 1.0
        assert res2["mean_squared_error"] == 0.0


class TestRandomizeModelWeights:
    """Verifies selective layer re-initialization according to Adebayo scopes."""

    def test_classifier_mode_preserves_backbone(self, densenet_model):
        orig_backbone_p = [p.clone() for p in densenet_model.backbone.parameters()]
        orig_head_p = [p.clone() for p in densenet_model.head.parameters()]

        rand_model = randomize_model_weights(densenet_model, "densenet121", randomize_mode="classifier", seed=42)

        # Backbone weights must remain identical
        for orig, rand in zip(orig_backbone_p, rand_model.backbone.parameters()):
            assert torch.equal(orig, rand)

        # Head weights must differ
        diff_found = any(
            not torch.equal(orig, rand)
            for orig, rand in zip(orig_head_p, rand_model.head.parameters())
        )
        assert diff_found

    def test_all_mode_randomizes_network(self, densenet_model):
        orig_backbone_p = [p.clone() for p in densenet_model.backbone.parameters()]
        rand_model = randomize_model_weights(densenet_model, "densenet121", randomize_mode="all", seed=42)

        diff_found = any(
            not torch.equal(orig, rand)
            for orig, rand in zip(orig_backbone_p, rand_model.backbone.parameters())
        )
        assert diff_found

    def test_invalid_mode_raises(self, densenet_model):
        with pytest.raises(ValueError, match="Unsupported randomize_mode"):
            randomize_model_weights(densenet_model, "densenet121", randomize_mode="invalid_scope")


class TestParameterRandomizationExecution:
    """Tests single-image and cohort Adebayo parameter randomization checks."""

    def test_run_sanity_check_densenet(self, densenet_model):
        img = torch.randn(1, 3, 32, 32)
        orig_cam = np.random.rand(32, 32).astype(np.float32)

        res = run_parameter_randomization_test(
            model=densenet_model,
            architecture="densenet121",
            image_tensor=img,
            original_cam=orig_cam,
            target_class_idx=5,
            randomize_mode="classifier",
            threshold_drop_percent=75.0,
            seed=42,
        )

        assert "spearman_rank_correlation" in res
        assert "sanity_passed" in res
        assert res["randomize_mode"] == "classifier"
        assert res["architecture"] == "densenet121"

    def test_run_sanity_check_efficientnet(self, efficientnet_model):
        img = torch.randn(1, 3, 32, 32)
        orig_cam = np.random.rand(32, 32).astype(np.float32)

        res = run_parameter_randomization_test(
            model=efficientnet_model,
            architecture="efficientnet_b0",
            image_tensor=img,
            original_cam=orig_cam,
            target_class_idx=5,
            randomize_mode="cascading",
            threshold_drop_percent=75.0,
            seed=42,
        )

        assert "spearman_rank_correlation" in res
        assert res["randomize_mode"] == "cascading"

    def test_cohort_sanity_checks_and_report_export(self, densenet_model, tmp_path: Path):
        images = [torch.randn(1, 3, 32, 32) for _ in range(2)]
        cams = [np.random.rand(32, 32).astype(np.float32) for _ in range(2)]

        cohort_res = evaluate_cohort_sanity_checks(
            model=densenet_model,
            architecture="densenet121",
            images=images,
            cams=cams,
            target_class_idx=5,
            randomize_modes=["classifier", "all"],
            threshold_drop_percent=75.0,
            seed=42,
        )

        assert cohort_res["num_samples"] == 2
        assert "classifier" in cohort_res["modes"]
        assert "all" in cohort_res["modes"]
        assert "verdict" in cohort_res["modes"]["classifier"]

        # Markdown format
        md_text = format_sanity_check_markdown_table(cohort_res)
        assert "| **densenet121** |" in md_text
        assert "`classifier`" in md_text

        # Disk export
        json_out, md_out = save_sanity_check_report(cohort_res, output_dir=tmp_path / "xai")
        assert json_out.is_file()
        assert md_out.is_file()

