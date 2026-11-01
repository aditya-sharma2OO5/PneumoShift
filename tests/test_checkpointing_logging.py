"""Unit tests for Task 2.3: Checkpointing, Git Hashing, and Multi-Metric Logging."""

import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.training.callbacks import EarlyStopping, ModelCheckpoint, get_git_hash
from pneumoshift.training.metrics_logger import EpochMetricsLogger
from pneumoshift.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers & Dummy Models
# ---------------------------------------------------------------------------

class DummyLinearModel(nn.Module):
    def __init__(self, in_features: int = 4, num_classes: int = 8) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor, return_features: bool = False) -> torch.Tensor:
        out = self.fc(x)
        if return_features:
            return out, x
        return out


# ---------------------------------------------------------------------------
# Test: Git Hash Helper
# ---------------------------------------------------------------------------

def test_get_git_hash():
    git_hash = get_git_hash()
    assert isinstance(git_hash, str)
    assert len(git_hash) > 0


# ---------------------------------------------------------------------------
# Test: ModelCheckpoint
# ---------------------------------------------------------------------------

class TestModelCheckpoint:
    def test_checkpoint_latest_saved(self, tmp_path: Path):
        model = DummyLinearModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cb = ModelCheckpoint(checkpoint_dir=tmp_path, monitor="val_macro_auroc")

        metrics = {"val_macro_auroc": 0.75, "val_pneumonia_auroc": 0.70}
        cfg = {"experiment_name": "test_run"}

        saved_path = cb.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics=metrics,
            config=cfg,
            is_best=False,
        )

        assert saved_path == tmp_path / "checkpoint_latest.pt"
        assert saved_path.exists()

        checkpoint = torch.load(saved_path)
        assert checkpoint["epoch"] == 1
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert checkpoint["metrics"] == metrics
        assert checkpoint["config"] == cfg
        assert "git_hash" in checkpoint
        assert isinstance(checkpoint["git_hash"], str)

    def test_checkpoint_best_primary_saved(self, tmp_path: Path):
        model = DummyLinearModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cb = ModelCheckpoint(checkpoint_dir=tmp_path, monitor="val_macro_auroc")

        metrics = {"val_macro_auroc": 0.80, "val_pneumonia_auroc": 0.72}
        saved_path = cb.save(
            model=model,
            optimizer=optimizer,
            epoch=2,
            metrics=metrics,
            config={},
            is_best=True,
        )

        assert saved_path == tmp_path / "checkpoint_best.pt"
        assert (tmp_path / "checkpoint_best.pt").exists()
        assert (tmp_path / "checkpoint_latest.pt").exists()

        best_ckpt = torch.load(tmp_path / "checkpoint_best.pt")
        assert best_ckpt["epoch"] == 2
        assert best_ckpt["metrics"]["val_macro_auroc"] == 0.80

    def test_checkpoint_secondary_pneumonia_saved(self, tmp_path: Path):
        model = DummyLinearModel()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        cb = ModelCheckpoint(
            checkpoint_dir=tmp_path,
            monitor="val_macro_auroc",
            secondary_monitor="val_pneumonia_auroc",
        )

        # Epoch 1: Good macro, mediocre pneumonia
        cb.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            metrics={"val_macro_auroc": 0.85, "val_pneumonia_auroc": 0.70},
            config={},
            is_best=True,
        )
        assert (tmp_path / "checkpoint_best.pt").exists()
        assert (tmp_path / "checkpoint_best_pneumonia.pt").exists()

        pneu_ckpt_1 = torch.load(tmp_path / "checkpoint_best_pneumonia.pt")
        assert pneu_ckpt_1["epoch"] == 1
        assert pneu_ckpt_1["metrics"]["val_pneumonia_auroc"] == 0.70

        # Epoch 2: Worse macro, but BETTER pneumonia
        cb.save(
            model=model,
            optimizer=optimizer,
            epoch=2,
            metrics={"val_macro_auroc": 0.80, "val_pneumonia_auroc": 0.78},
            config={},
            is_best=False,  # macro didn't improve
        )

        # Primary best should still be epoch 1
        best_ckpt = torch.load(tmp_path / "checkpoint_best.pt")
        assert best_ckpt["epoch"] == 1

        # Secondary best should be updated to epoch 2
        pneu_ckpt_2 = torch.load(tmp_path / "checkpoint_best_pneumonia.pt")
        assert pneu_ckpt_2["epoch"] == 2
        assert pneu_ckpt_2["metrics"]["val_pneumonia_auroc"] == 0.78


# ---------------------------------------------------------------------------
# Test: EpochMetricsLogger
# ---------------------------------------------------------------------------

class TestEpochMetricsLogger:
    def test_csv_and_json_logging(self, tmp_path: Path):
        logger = EpochMetricsLogger(output_dir=tmp_path)
        csv_file = tmp_path / "epoch_metrics.csv"
        json_file = tmp_path / "epoch_metrics.json"

        # Epoch 1
        logger.log(
            epoch=1,
            metrics={
                "train_loss": 0.54,
                "loss": 0.50,
                "val_macro_auroc": 0.71,
                "val_macro_auprc": 0.42,
                "val_pneumonia_auroc": 0.69,
                "val_pneumonia_auprc": 0.38,
            },
        )

        assert csv_file.exists()
        assert json_file.exists()

        # Epoch 2
        logger.log(
            epoch=2,
            metrics={
                "train_loss": 0.48,
                "loss": 0.45,
                "val_macro_auroc": 0.74,
                "val_macro_auprc": 0.45,
                "val_pneumonia_auroc": 0.72,
                "val_pneumonia_auprc": 0.41,
            },
        )

        # Validate CSV contents
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert reader[0]["epoch"] == "1"
            assert float(reader[0]["train_loss"]) == pytest.approx(0.54)
            assert float(reader[1]["val_pneumonia_auroc"]) == pytest.approx(0.72)

        # Validate JSON contents
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert len(data) == 2
            assert data[0]["epoch"] == 1
            assert data[1]["epoch"] == 2
            assert data[1]["val_macro_auroc"] == pytest.approx(0.74)

        assert len(logger.history) == 2

    def test_dynamic_columns_addition(self, tmp_path: Path):
        logger = EpochMetricsLogger(output_dir=tmp_path)
        csv_file = tmp_path / "epoch_metrics.csv"

        logger.log(epoch=1, metrics={"train_loss": 0.5, "val_loss": 0.4})
        # Epoch 2 introduces new column
        logger.log(epoch=2, metrics={"train_loss": 0.4, "val_loss": 0.3, "extra_metric": 0.99})

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            assert len(reader) == 2
            assert "extra_metric" in reader[0]
            assert reader[1]["extra_metric"] == "0.99"


# ---------------------------------------------------------------------------
# Test: Trainer Multi-Metric Calculation (AUROC & AUPRC per class)
# ---------------------------------------------------------------------------

class TestTrainerEvaluationMetrics:
    def test_evaluate_computes_auroc_and_auprc(self, tmp_path: Path):
        # Create synthetic dataset with known distribution
        num_samples = 40
        num_classes = len(SHARED_CLASSES)

        torch.manual_seed(42)
        images = torch.randn(num_samples, 4)
        targets = torch.zeros(num_samples, num_classes)
        # Ensure at least one positive and negative for each class
        targets[:20, :] = 1.0

        dataset = TensorDataset(images, targets)
        loader = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=lambda batch: {
                "image": torch.stack([b[0] for b in batch]),
                "target": torch.stack([b[1] for b in batch]),
            },
        )

        model = DummyLinearModel(in_features=4, num_classes=num_classes)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        cfg = {
            "train": {
                "checkpoint_dir": str(tmp_path / "checkpoints"),
                "run_output_dir": str(tmp_path / "runs"),
                "mixed_precision": False,
            }
        }

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            config=cfg,
            label_cols=SHARED_CLASSES,
        )

        metrics = trainer.evaluate(loader)

        # Verify loss and macro metrics exist
        assert "loss" in metrics
        assert "val_macro_auroc" in metrics
        assert "val_macro_auprc" in metrics
        assert "val_pneumonia_auroc" in metrics
        assert "val_pneumonia_auprc" in metrics

        # Verify all 8 shared classes have individual auroc and auprc
        for col in SHARED_CLASSES:
            key = col.lower().replace(" ", "_")
            assert f"auroc_{key}" in metrics, f"Missing auroc_{key}"
            assert f"auprc_{key}" in metrics, f"Missing auprc_{key}"
            assert 0.0 <= metrics[f"auroc_{key}"] <= 1.0
            assert 0.0 <= metrics[f"auprc_{key}"] <= 1.0

