"""Unit tests for literature benchmark comparison and in-domain evaluation engine."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.evaluation.benchmarks import (
    ARASTEH_2023_CHEXPERT,
    ARASTEH_2023_NIH,
    IRVIN_2019_CHEXPERT_UZEROS,
    PAN_2019_NIH,
    compare_against_benchmarks,
)
from scripts.evaluate import evaluate_split, extract_features_and_predictions


# ---------------------------------------------------------------------------
# Test: Literature Benchmarks
# ---------------------------------------------------------------------------

class TestBenchmarks:
    def test_pan_2019_nih_targets(self):
        assert PAN_2019_NIH["Pneumonia"] == 0.734
        assert PAN_2019_NIH["Atelectasis"] == 0.833
        assert PAN_2019_NIH["Cardiomegaly"] == 0.920
        assert len(PAN_2019_NIH) == 8

    def test_irvin_2019_chexpert_targets(self):
        assert IRVIN_2019_CHEXPERT_UZEROS["Atelectasis"] == 0.811
        assert IRVIN_2019_CHEXPERT_UZEROS["Consolidation"] == 0.932
        assert IRVIN_2019_CHEXPERT_UZEROS["Pleural Effusion"] == 0.931

    def test_arasteh_2023_targets(self):
        assert ARASTEH_2023_NIH["Pneumonia"]["mean"] == 0.688
        assert ARASTEH_2023_CHEXPERT["Pneumonia"]["mean"] == 0.764

    def test_compare_against_benchmarks_aligned(self):
        metrics = {
            "auroc_pneumonia": 0.730,
            "auroc_atelectasis": 0.830,
            "auroc_cardiomegaly": 0.915,
        }
        res = compare_against_benchmarks(metrics, benchmark_name="pan_2019_nih")

        assert res["headline_sanity_passed"] is True
        assert res["comparisons"]["Pneumonia"]["status"] == "Aligned"
        assert abs(res["comparisons"]["Pneumonia"]["difference"]) <= 0.01
        assert "table_markdown" in res
        assert "| **Pneumonia** |" in res["table_markdown"]

    def test_compare_against_benchmarks_sanity_bounds_flag(self):
        # Abnormally low pneumonia AUROC
        metrics = {"auroc_pneumonia": 0.52}
        res = compare_against_benchmarks(metrics, benchmark_name="pan_2019_nih", sanity_bounds=(0.65, 0.85))

        assert res["headline_sanity_passed"] is False
        assert "outside sanity bounds" in res["headline_sanity_message"]

    def test_invalid_benchmark_name_raises(self):
        with pytest.raises(ValueError, match="Unknown benchmark"):
            compare_against_benchmarks({}, benchmark_name="nonexistent_bench")


# ---------------------------------------------------------------------------
# Test: In-Domain Evaluation Engine
# ---------------------------------------------------------------------------

class DummyEvalModel(nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False):
        logits = self.fc(x)
        if return_features:
            return logits, x
        return logits


class TestEvaluationEngine:
    def test_extract_features_and_predictions(self):
        num_samples = 20
        model = DummyEvalModel(in_features=4, num_classes=8)
        dataset = TensorDataset(torch.randn(num_samples, 4), torch.zeros(num_samples, 8))
        loader = DataLoader(
            dataset,
            batch_size=5,
            collate_fn=lambda b: {
                "image": torch.stack([x[0] for x in b]),
                "target": torch.stack([x[1] for x in b]),
            },
        )

        device = torch.device("cpu")
        logits, probs, targets, features = extract_features_and_predictions(model, loader, device)

        assert logits.shape == (num_samples, 8)
        assert probs.shape == (num_samples, 8)
        assert targets.shape == (num_samples, 8)
        assert features.shape == (num_samples, 4)
        assert np.all((probs >= 0.0) & (probs <= 1.0))

    def test_evaluate_split(self):
        num_samples = 30
        model = DummyEvalModel(in_features=4, num_classes=8)
        dataset = TensorDataset(torch.randn(num_samples, 4), torch.zeros(num_samples, 8))
        loader = DataLoader(
            dataset,
            batch_size=10,
            collate_fn=lambda b: {
                "image": torch.stack([x[0] for x in b]),
                "target": torch.stack([x[1] for x in b]),
            },
        )

        df = pd.DataFrame({
            "image_id": [f"img_{i:03d}.png" for i in range(num_samples)],
            "patient_id": [f"p_{i % 5:02d}" for i in range(num_samples)],
        })

        device = torch.device("cpu")
        results = evaluate_split(
            model=model,
            loader=loader,
            df=df,
            label_cols=SHARED_CLASSES,
            device=device,
            compute_bootstrap=False,
        )

        assert results["num_samples"] == num_samples
        assert results["num_patients"] == 5
        assert "per_class" in results
        assert "macro_auroc" in results
        assert "macro_auprc" in results
        assert "headline_pneumonia" in results
        assert "Pneumonia" in results["per_class"]

        pneu = results["per_class"]["Pneumonia"]
        assert "auroc" in pneu
        assert "auprc" in pneu
        assert "ece" in pneu
        assert "mce" in pneu

    def test_chexpert_uncertainty_modes_in_evaluation(self):
        # 10 samples with uncertain (-1.0) ground truth
        targets = torch.tensor([
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],  # uncertain sample
            [1.0, 1.0],
            [0.0, 0.0],
            [-1.0, -1.0], # uncertain sample
        ])
        num_samples = len(targets)
        images = torch.randn(num_samples, 4)

        dataset = TensorDataset(images, targets)
        loader = DataLoader(
            dataset,
            batch_size=num_samples,
            collate_fn=lambda b: {
                "image": torch.stack([x[0] for x in b]),
                "target": torch.stack([x[1] for x in b]),
            },
        )
        df = pd.DataFrame({
            "image_id": [f"img_{i:02d}.png" for i in range(num_samples)],
            "patient_id": [f"p_{i:02d}" for i in range(num_samples)],
        })
        model = DummyEvalModel(in_features=4, num_classes=2)
        device = torch.device("cpu")

        # Mode 1: uncertain_as_negative
        res_neg = evaluate_split(
            model=model,
            loader=loader,
            df=df,
            label_cols=["Pneumonia", "Atelectasis"],
            device=device,
            compute_bootstrap=False,
            uncertainty_mode="uncertain_as_negative",
        )
        assert res_neg["uncertainty_mode"] == "uncertain_as_negative"
        assert not np.isnan(res_neg["per_class"]["Pneumonia"]["auroc"])

        # Mode 2: uncertain_excluded
        res_exc = evaluate_split(
            model=model,
            loader=loader,
            df=df,
            label_cols=["Pneumonia", "Atelectasis"],
            device=device,
            compute_bootstrap=False,
            uncertainty_mode="uncertain_excluded",
        )
        assert res_exc["uncertainty_mode"] == "uncertain_excluded"
        assert not np.isnan(res_exc["per_class"]["Pneumonia"]["auroc"])

    def test_chexpert_official_val_benchmark_comparison(self):
        # Comparison with Irvin et al. (2019) official validation targets
        metrics = {
            "auroc_atelectasis": 0.815,
            "auroc_cardiomegaly": 0.845,
            "auroc_consolidation": 0.928,
            "auroc_edema": 0.930,
            "auroc_pleural_effusion": 0.933,
            "auroc_pneumonia": 0.760,
        }
        res = compare_against_benchmarks(metrics, benchmark_name="irvin_2019_chexpert")
        assert res["comparisons"]["Atelectasis"]["benchmark"] == 0.811
        assert res["comparisons"]["Consolidation"]["benchmark"] == 0.932
        assert res["comparisons"]["Pleural Effusion"]["benchmark"] == 0.931
        assert res["comparisons"]["Atelectasis"]["status"] == "Aligned"
        assert res["headline_sanity_passed"] is True

    def test_chexpert_test_benchmark_comparison(self):
        # Comparison with Tayebi Arasteh et al. (2023) CheXpert targets
        metrics = {
            "auroc_pneumonia": 0.765,
            "auroc_atelectasis": 0.685,
            "auroc_cardiomegaly": 0.870,
        }
        res = compare_against_benchmarks(metrics, benchmark_name="arasteh_2023_chexpert")
        assert res["comparisons"]["Pneumonia"]["benchmark"] == 0.764
        assert res["comparisons"]["Pneumonia"]["status"] == "Aligned"
        assert res["headline_sanity_passed"] is True

