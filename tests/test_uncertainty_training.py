"""Unit tests for CheXpert uncertainty label policies (U-Zeros, U-Ones, U-Ignore),
batch loss masking, and dual-mode official validation evaluation.
"""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.harmonize_labels import (
    HEADLINE_CLASS,
    SHARED_CLASSES,
    harmonize_chexpert_labels,
    get_evaluation_targets,
)
from pneumoshift.models.factory import build_model
from pneumoshift.training.losses import MaskedBCEWithLogitsLoss
from pneumoshift.training.trainer import Trainer


class TestUncertaintyLabelHarmonization:
    """Verifies label mapping logic across U-Zeros, U-Ones, and U-Ignore policies."""

    @pytest.fixture
    def raw_chexpert_df(self) -> pd.DataFrame:
        data = {
            "Path": [
                "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
                "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
                "CheXpert-v1.0-small/train/patient00003/study1/view1_lateral.jpg",
            ],
            "Frontal/Lateral": ["Frontal", "Frontal", "Lateral"],
            "Sex": ["Male", "Female", "Male"],
            "Age": [65, 45, 70],
            "AP/PA": ["AP", "PA", "LL"],
            "Atelectasis": [1.0, -1.0, 0.0],
            "Cardiomegaly": [0.0, 1.0, 1.0],
            "Consolidation": [-1.0, 0.0, 1.0],
            "Edema": [0.0, -1.0, 0.0],
            "Pleural Effusion": [1.0, 1.0, -1.0],
            "Pneumonia": [-1.0, 1.0, 0.0],
            "Pneumothorax": [0.0, 0.0, 1.0],
            "No Finding": [np.nan, 0.0, 0.0],
        }
        return pd.DataFrame(data)

    def test_u_zeros_mapping(self, raw_chexpert_df: pd.DataFrame):
        harmonized = harmonize_chexpert_labels(raw_chexpert_df, uncertainty_policy="u_zeros", drop_lateral=True)
        assert len(harmonized) == 2  # Lateral dropped
        # In patient00002: Atelectasis was -1.0 -> should now be 0.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00002", "Atelectasis"].values[0] == 0.0
        # In patient00001: Pneumonia was -1.0 -> should now be 0.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00001", "Pneumonia"].values[0] == 0.0
        # Positive labels remain 1.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00001", "Atelectasis"].values[0] == 1.0

    def test_u_ones_mapping(self, raw_chexpert_df: pd.DataFrame):
        harmonized = harmonize_chexpert_labels(raw_chexpert_df, uncertainty_policy="u_ones", drop_lateral=True)
        assert len(harmonized) == 2
        # In patient00002: Atelectasis was -1.0 -> mapped to 1.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00002", "Atelectasis"].values[0] == 1.0
        # In patient00001: Pneumonia was -1.0 -> mapped to 1.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00001", "Pneumonia"].values[0] == 1.0
        # In patient00001: Cardiomegaly was 0.0 -> remains 0.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00001", "Cardiomegaly"].values[0] == 0.0

    def test_u_ignore_preserves_negative_one(self, raw_chexpert_df: pd.DataFrame):
        harmonized = harmonize_chexpert_labels(raw_chexpert_df, uncertainty_policy="u_ignore", drop_lateral=True)
        assert len(harmonized) == 2
        # -1.0 must be strictly preserved for gradient masking
        assert harmonized.loc[harmonized["patient_id"] == "patient00002", "Atelectasis"].values[0] == -1.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00001", "Pneumonia"].values[0] == -1.0
        assert harmonized.loc[harmonized["patient_id"] == "patient00001", "Atelectasis"].values[0] == 1.0


class TestBatchLossMasking:
    """Verifies batch binary loss mask generation and gradient backprop isolation."""

    def test_dataset_emits_correct_loss_mask(self, tmp_path: Path):
        img_file = tmp_path / "img1.png"
        from PIL import Image
        Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(img_file)

        df = pd.DataFrame([{
            "image_path": str(img_file),
            "patient_id": "p1",
            "Atelectasis": 1.0,
            "Cardiomegaly": -1.0,  # Uncertain
            "Consolidation": 0.0,
            "Edema": -1.0,        # Uncertain
            "Pleural Effusion": 1.0,
            "Pneumonia": 0.0,
            "Pneumothorax": -1.0,  # Uncertain
            "No Finding": 0.0,
        }])

        dataset = ChestXrayDataset(df, images_dir=tmp_path)
        sample = dataset[0]

        assert "mask" in sample
        mask = sample["mask"]
        target = sample["target"]

        # Mask should be 0.0 where target == -1.0, and 1.0 elsewhere
        expected_mask = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])
        assert torch.equal(mask, expected_mask)

    def test_masked_loss_zeroes_gradients_on_ignored_positions(self):
        criterion = MaskedBCEWithLogitsLoss()
        logits = torch.randn(2, 4, requires_grad=True)
        targets = torch.tensor([
            [1.0, -1.0, 0.0, 1.0],
            [0.0,  1.0, -1.0, 0.0],
        ])
        mask = (targets != -1.0).float()

        loss = criterion(logits, targets, mask)
        loss.backward()

        assert logits.grad is not None
        # Positions with mask == 0.0 must have exactly zero gradient
        assert logits.grad[0, 1].item() == 0.0
        assert logits.grad[1, 2].item() == 0.0
        # Valid positions must have non-zero gradients
        assert logits.grad[0, 0].item() != 0.0
        assert logits.grad[1, 1].item() != 0.0


class TestDualModeEvaluation:
    """Verifies evaluation under both uncertain_as_negative and uncertain_excluded policies."""

    class MockDataset(Dataset):
        def __init__(self):
            self.images = torch.randn(10, 3, 32, 32)
            # 10 samples, 2 classes: class 0 has uncertain values, class 1 is clean
            # Targets: 1, 0, -1
            self.targets = torch.tensor([
                [1.0, 1.0],
                [1.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [-1.0, 1.0],  # Uncertain in class 0
                [-1.0, 0.0],  # Uncertain in class 0
                [1.0, 1.0],
                [0.0, 0.0],
                [1.0, 1.0],
                [0.0, 0.0],
            ])
            self.masks = (self.targets != -1.0).float()

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return {
                "image": self.images[idx],
                "target": self.targets[idx],
                "mask": self.masks[idx],
            }

    def test_evaluation_under_both_modes(self):
        model = build_model(architecture="densenet121", num_classes=2, pretrained=False)
        dataset = self.MockDataset()
        loader = DataLoader(dataset, batch_size=5, shuffle=False)
        criterion = MaskedBCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            criterion=criterion,
            optimizer=optimizer,
            label_cols=["Atelectasis", "Pneumonia"],
        )

        metrics_neg = trainer.evaluate(loader, uncertainty_mode="uncertain_as_negative")
        assert metrics_neg["uncertainty_mode"] == "uncertain_as_negative"
        assert "auroc_atelectasis" in metrics_neg
        assert "auroc_pneumonia" in metrics_neg

        metrics_excl = trainer.evaluate(loader, uncertainty_mode="uncertain_excluded")
        assert metrics_excl["uncertainty_mode"] == "uncertain_excluded"
        assert "auroc_atelectasis" in metrics_excl
        assert "auroc_pneumonia" in metrics_excl

    def test_get_evaluation_targets_helper(self):
        y_raw = np.array([1.0, -1.0, 0.0, -1.0, 1.0])
        y_neg = get_evaluation_targets(y_raw, mode="uncertain_as_negative")
        assert np.array_equal(y_neg, np.array([1.0, 0.0, 0.0, 0.0, 1.0]))

        y_excl = get_evaluation_targets(y_raw, mode="uncertain_excluded")
        assert np.array_equal(y_excl, y_raw)

