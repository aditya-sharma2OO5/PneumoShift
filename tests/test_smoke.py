"""End-to-end smoke test validating pipeline on synthetic data without real datasets."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.splits import create_synthetic_dataset, verify_patient_leakage
from pneumoshift.data.transforms import get_transforms
from pneumoshift.evaluation.metrics import compute_discrimination_metrics
from pneumoshift.models.factory import build_model, get_target_layer_for_xai
from pneumoshift.training.losses import MaskedBCEWithLogitsLoss
from pneumoshift.xai.faithfulness import compute_deletion_insertion_curves
from pneumoshift.xai.gradcam import GradCAMExplainer


def test_end_to_end_synthetic_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # 1. Generate synthetic dataset
        split_paths = create_synthetic_dataset(output_dir=tmp_path, num_samples=15, image_size=32, seed=42)

        train_df = pd.read_csv(split_paths["train"])
        val_df = pd.read_csv(split_paths["val"])
        test_df = pd.read_csv(split_paths["test"])

        # 2. Leakage check
        verify_patient_leakage({"train": train_df, "val": val_df, "test": test_df})

        # 3. Loaders
        transform = get_transforms(split="train", image_size=32)
        dataset = ChestXrayDataset(train_df, images_dir=tmp_path / "images", transform=transform)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)

        batch = next(iter(loader))
        images = batch["image"]
        targets = batch["target"]
        assert images.shape == (2, 3, 32, 32)
        assert targets.shape == (2, 8)

        # 4. Model instantiation & forward pass
        model = build_model(architecture="densenet121", num_classes=8, pretrained=False)
        logits = model(images)
        assert logits.shape == (2, 8)

        # 5. Loss calculation
        criterion = MaskedBCEWithLogitsLoss()
        loss = criterion(logits, targets)
        loss.backward()
        assert not torch.isnan(loss)

        # 6. Metrics
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7])
        disc_metrics = compute_discrimination_metrics(y_true, y_pred)
        assert "auroc" in disc_metrics

        # 7. XAI Grad-CAM & faithfulness
        target_layer = get_target_layer_for_xai(model, "densenet121")
        explainer = GradCAMExplainer(model, target_layer)
        cam = explainer.generate_cam(images[:1], target_class_idx=5, target_size=(32, 32))
        assert cam.shape == (32, 32)

        faith = compute_deletion_insertion_curves(model, images[:1], cam, target_class_idx=5, num_steps=2)
        assert "deletion_auc" in faith
        explainer.release_hooks()

