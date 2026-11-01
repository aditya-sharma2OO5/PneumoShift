"""Unit tests for single-task vs multi-task pneumonia ablation comparison."""

from pathlib import Path
import numpy as np
import pytest
import torch

from pneumoshift.evaluation.ablation import compare_single_vs_multitask
from pneumoshift.models.factory import build_model
from pneumoshift.utils.io import load_yaml_config


# ---------------------------------------------------------------------------
# Test: compare_single_vs_multitask
# ---------------------------------------------------------------------------

class TestAblationComparison:
    def test_compare_metrics_deltas(self):
        multi_metrics = {
            "val_pneumonia_auroc": 0.740,
            "val_pneumonia_auprc": 0.410,
            "ece": 0.040,
        }
        single_metrics = {
            "val_pneumonia_auroc": 0.710,
            "val_pneumonia_auprc": 0.380,
            "ece": 0.055,
        }

        res = compare_single_vs_multitask(
            multitask_results=multi_metrics,
            singletask_results=single_metrics,
            cohort_name="NIH ChestX-ray14",
        )

        assert res["multi_auroc"] == pytest.approx(0.740)
        assert res["single_auroc"] == pytest.approx(0.710)
        assert res["delta_auroc"] == pytest.approx(0.030)
        assert res["delta_auprc"] == pytest.approx(0.030)
        assert res["delta_ece"] == pytest.approx(-0.015)
        assert "table_markdown" in res
        assert "Multi-label regularizes" in res["table_markdown"]

    def test_paired_bootstrap_significance_in_comparison(self):
        rng = np.random.RandomState(42)
        n = 40
        y_true = np.array([1.0 if i < 10 else 0.0 for i in range(n)])
        # Multi-task is slightly better on positive cases
        y_pred_multi = np.clip(y_true * 0.7 + rng.rand(n) * 0.3, 0.0, 1.0)
        y_pred_single = np.clip(y_true * 0.5 + rng.rand(n) * 0.5, 0.0, 1.0)
        patient_ids = np.array([f"p_{i // 2:02d}" for i in range(n)])

        res = compare_single_vs_multitask(
            multitask_results={"auroc_pneumonia": 0.75},
            singletask_results={"auroc_pneumonia": 0.68},
            cohort_name="NIH",
            patient_ids=patient_ids,
            y_true=y_true,
            y_pred_multi=y_pred_multi,
            y_pred_single=y_pred_single,
            n_resamples=100,
        )

        assert res["significance_test"] is not None
        assert "p_value" in res["significance_test"]
        assert "mean_diff" in res["significance_test"]

    def test_feature_space_analysis(self):
        feats_multi = np.random.randn(20, 16)
        feats_single = np.random.randn(20, 16) + 0.5

        res = compare_single_vs_multitask(
            multitask_results={"auroc_pneumonia": 0.74},
            singletask_results={"auroc_pneumonia": 0.71},
            feats_multi=feats_multi,
            feats_single=feats_single,
        )

        assert res["feature_analysis"] is not None
        assert "mmd_distance" in res["feature_analysis"]
        assert res["feature_analysis"]["mmd_distance"] >= 0.0


# ---------------------------------------------------------------------------
# Test: Single-Label Ablation Model Instantiation
# ---------------------------------------------------------------------------

class TestSingleLabelModel:
    def test_pneumonia_only_nih_config(self):
        cfg = load_yaml_config("configs/train/pneumonia_only_nih.yaml")
        assert cfg["model"]["num_classes"] == 1
        assert cfg["data"]["headline_only"] is True
        assert cfg["data"]["shared_classes"] == ["Pneumonia"]

        model = build_model(
            architecture=cfg["model"]["architecture"],
            num_classes=cfg["model"]["num_classes"],
            pretrained=False,
        )
        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (2, 1)

    def test_pneumonia_only_chexpert_config(self):
        cfg = load_yaml_config("configs/train/pneumonia_only_chexpert.yaml")
        assert cfg["model"]["num_classes"] == 1
        assert cfg["data"]["dataset_name"] == "chexpert"
        assert cfg["data"]["headline_only"] is True
        assert cfg["data"]["uncertainty_policy"] == "u_zeros"

