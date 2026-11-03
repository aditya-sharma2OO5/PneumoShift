"""Unit tests for cross-domain evaluation engine and generalization gap calculation."""

from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.evaluation.bootstrap import two_sample_bootstrap_diff
from pneumoshift.evaluation.cross_domain import (
    compute_generalization_gap_report,
    extract_cross_domain_predictions,
)


# ---------------------------------------------------------------------------
# Dummy Model
# ---------------------------------------------------------------------------

class DummyCrossModel(nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits = self.fc(x)
        if return_features:
            return logits, x
        return logits


# ---------------------------------------------------------------------------
# Tests: Two-Sample Clustered Bootstrap Difference
# ---------------------------------------------------------------------------

class TestTwoSampleBootstrapDiff:
    def test_two_sample_bootstrap_calculation(self):
        rng = np.random.RandomState(42)
        n_a = 40
        n_b = 50

        # Cohort A (Source - higher performance)
        y_true_a = np.array([1.0 if i < 15 else 0.0 for i in range(n_a)])
        y_pred_a = np.clip(y_true_a * 0.8 + rng.rand(n_a) * 0.2, 0.0, 1.0)
        pids_a = np.array([f"src_p_{i // 2:02d}" for i in range(n_a)])

        # Cohort B (Target - degraded performance)
        y_true_b = np.array([1.0 if i < 15 else 0.0 for i in range(n_b)])
        y_pred_b = np.clip(y_true_b * 0.4 + rng.rand(n_b) * 0.6, 0.0, 1.0)
        pids_b = np.array([f"tgt_p_{i // 2:02d}" for i in range(n_b)])

        res = two_sample_bootstrap_diff(
            patient_ids_a=pids_a,
            y_true_a=y_true_a,
            y_pred_a=y_pred_a,
            patient_ids_b=pids_b,
            y_true_b=y_true_b,
            y_pred_b=y_pred_b,
            n_resamples=100,
            seed=42,
        )

        assert "mean_diff" in res
        assert "ci_lower" in res
        assert "ci_upper" in res
        assert "p_value" in res
        assert res["mean_diff"] > 0  # Cohort A AUROC should exceed Cohort B
        assert 0.0 <= res["p_value"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: Forward Prediction & Feature Extraction
# ---------------------------------------------------------------------------

class TestCrossDomainInference:
    def test_extract_cross_domain_predictions(self):
        num_samples = 25
        model = DummyCrossModel(in_features=4, num_classes=8)
        dataset = TensorDataset(torch.randn(num_samples, 4), torch.zeros(num_samples, 8))
        loader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=lambda b: {
                "image": torch.stack([x[0] for x in b]),
                "target": torch.stack([x[1] for x in b]),
            },
        )
        device = torch.device("cpu")

        logits, probs, targets, features = extract_cross_domain_predictions(model, loader, device)

        assert logits.shape == (num_samples, 8)
        assert probs.shape == (num_samples, 8)
        assert targets.shape == (num_samples, 8)
        assert features.shape == (num_samples, 4)
        assert np.all((probs >= 0.0) & (probs <= 1.0))


# ---------------------------------------------------------------------------
# Tests: Generalization Gap Report Engine
# ---------------------------------------------------------------------------

class TestGeneralizationGapReport:
    def test_compute_generalization_gap_metrics_and_markdown(self):
        in_domain = {
            "auroc_pneumonia": 0.735,
            "auprc_pneumonia": 0.380,
            "auroc_atelectasis": 0.830,
            "auprc_atelectasis": 0.520,
            "val_macro_auroc": 0.820,
            "val_macro_auprc": 0.500,
        }
        cross_domain = {
            "auroc_pneumonia": 0.620,
            "auprc_pneumonia": 0.290,
            "auroc_atelectasis": 0.740,
            "auprc_atelectasis": 0.410,
            "val_macro_auroc": 0.730,
            "val_macro_auprc": 0.400,
        }

        res = compute_generalization_gap_report(
            in_domain_metrics=in_domain,
            cross_domain_metrics=cross_domain,
            shift_direction="nih_to_chexpert",
            findings=["Pneumonia", "Atelectasis"],
        )

        assert res["shift_direction"] == "nih_to_chexpert"
        assert res["macro_gap_auroc"] == pytest.approx(0.090)
        assert res["macro_gap_auprc"] == pytest.approx(0.100)

        pneu_gap = res["headline_pneumonia_gap"]
        assert pneu_gap["in_domain_auroc"] == pytest.approx(0.735)
        assert pneu_gap["cross_domain_auroc"] == pytest.approx(0.620)
        assert pneu_gap["gap_auroc"] == pytest.approx(0.115)
        assert pneu_gap["gap_auprc"] == pytest.approx(0.090)

        # Markdown table verification
        assert "markdown_table" in res
        assert "| **Pneumonia** |" in res["markdown_table"]
        assert "| **Atelectasis** |" in res["markdown_table"]
        assert "| **Macro Average** |" in res["markdown_table"]

    def test_generalization_gap_with_hypothesis_testing(self):
        rng = np.random.RandomState(42)
        n = 30

        pids = np.array([f"p_{i // 2:02d}" for i in range(n)])
        targets = np.zeros((n, 2))
        targets[:10, :] = 1.0

        # In-domain predictions (stronger)
        in_preds = np.clip(targets * 0.8 + rng.rand(n, 2) * 0.2, 0.0, 1.0)
        # Cross-domain predictions (weaker)
        cross_preds = np.clip(targets * 0.4 + rng.rand(n, 2) * 0.6, 0.0, 1.0)

        in_domain = {"auroc_pneumonia": 0.78, "auroc_cardiomegaly": 0.85}
        cross_domain = {"auroc_pneumonia": 0.64, "auroc_cardiomegaly": 0.72}

        res = compute_generalization_gap_report(
            in_domain_metrics=in_domain,
            cross_domain_metrics=cross_domain,
            shift_direction="nih_to_chexpert",
            findings=["Pneumonia", "Cardiomegaly"],
            target_patients=pids,
            target_targets=targets,
            target_preds=cross_preds,
            target_in_domain_preds=in_preds,
            n_resamples=50,
        )

        pneu = res["per_class_gaps"]["Pneumonia"]
        assert pneu["p_value"] is not None
        assert pneu["fdr_adjusted_p_value"] is not None
        assert isinstance(pneu["fdr_significant"], bool)

