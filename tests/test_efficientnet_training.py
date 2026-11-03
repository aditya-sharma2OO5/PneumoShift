"""Unit tests for EfficientNet-B0 model pipeline, differential fine-tuning, and dual-monitor checkpointing."""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.models.factory import PneumoShiftModel, build_model, get_target_layer_for_xai
from pneumoshift.models.heads import MultiLabelClassificationHead
from pneumoshift.training.callbacks import ModelCheckpoint
from pneumoshift.training.losses import MaskedBCEWithLogitsLoss
from pneumoshift.training.trainer import Trainer


class TestEfficientNetArchitecture:
    """Verifies EfficientNet-B0 architectural specifications and layer dimensions."""

    def test_efficientnet_b0_head_dimensions(self):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False, dropout=0.2)
        assert isinstance(model, PneumoShiftModel)
        assert model.architecture == "efficientnet_b0"
        assert isinstance(model.head, MultiLabelClassificationHead)
        assert model.head.in_features == 1280
        assert model.head.num_classes == 8
        assert model.head.dropout.p == 0.2

    def test_efficientnet_b0_feature_extraction(self):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
        x = torch.randn(2, 3, 64, 64)
        logits, feats = model(x, return_features=True)
        assert logits.shape == (2, 8)
        assert feats.shape == (2, 1280)

    def test_efficientnet_b0_target_layer_for_xai(self):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
        target_layer = get_target_layer_for_xai(model, "efficientnet_b0")
        assert target_layer is not None
        # In torchvision EfficientNet, the last features block is Conv2dNormActivation
        assert isinstance(target_layer, nn.Module)


class TestEfficientNetTwoStageFreeze:
    """Verifies parameter freeze, BatchNorm evaluation isolation, and unfreeze mechanics."""

    def test_freeze_backbone_disables_gradients(self):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
        model.freeze_backbone()

        for p in model.backbone.parameters():
            assert not p.requires_grad, "Backbone parameter still requires grad after freeze"

        for p in model.head.parameters():
            assert p.requires_grad, "Head parameter should remain trainable after freeze"

    def test_freeze_backbone_maintains_batchnorm_eval(self):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
        model.freeze_backbone()
        model.train()  # Simulates trainer entering train_epoch

        for m in model.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                assert not m.training, f"BatchNorm layer {m} must remain in eval mode during freeze phase"

    def test_unfreeze_backbone_restores_train_mode(self):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
        model.freeze_backbone()
        model.unfreeze_backbone()

        for p in model.backbone.parameters():
            assert p.requires_grad, "Backbone parameter should require grad after unfreeze"

        for m in model.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                assert m.training, "BatchNorm layer should be in train mode after unfreeze"

    def test_differential_param_groups(self):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
        head_lr = 1e-4
        backbone_lr = 1e-5
        param_groups = [
            {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
            {"params": list(model.head.parameters()), "lr": head_lr, "name": "head"},
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-5)

        assert optimizer.param_groups[0]["name"] == "backbone"
        assert optimizer.param_groups[0]["lr"] == 0.0
        assert optimizer.param_groups[1]["name"] == "head"
        assert optimizer.param_groups[1]["lr"] == head_lr

        # Simulate unfreezing transition
        model.unfreeze_backbone()
        for pg in optimizer.param_groups:
            if pg.get("name") == "backbone":
                pg["lr"] = backbone_lr

        assert optimizer.param_groups[0]["lr"] == backbone_lr


class TestEfficientNetTrainingAndCheckpointing:
    """Verifies trainer execution, dual-monitor checkpointing, and metric logging."""

    class SyntheticCxrDataset(torch.utils.data.Dataset):
        def __init__(self, n_samples: int = 16):
            self.images = torch.randn(n_samples, 3, 64, 64)
            self.targets = torch.randint(0, 2, (n_samples, 8)).float()

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return {"image": self.images[idx], "target": self.targets[idx]}

    def test_trainer_dual_monitor_checkpoints(self, tmp_path: Path):
        model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
        train_ds = self.SyntheticCxrDataset(16)
        val_ds = self.SyntheticCxrDataset(8)

        train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

        ckpt_dir = tmp_path / "checkpoints"
        run_out_dir = tmp_path / "runs"
        config = {
            "train": {
                "checkpoint_dir": str(ckpt_dir),
                "run_output_dir": str(run_out_dir),
                "save_best_metric": "val_macro_auroc",
                "save_headline_metric": "val_pneumonia_auroc",
                "two_stage_finetune": True,
                "freeze_epochs": 1,
                "backbone_lr": 1e-5,
                "learning_rate": 1e-4,
                "mixed_precision": False,
            }
        }

        param_groups = [
            {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
            {"params": list(model.head.parameters()), "lr": 1e-4, "name": "head"},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=1e-4)
        criterion = MaskedBCEWithLogitsLoss()

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            label_cols=SHARED_CLASSES,
        )

        history = trainer.fit(epochs=2)
        assert len(history["train_loss"]) == 2
        assert len(history["val_macro_auroc"]) == 2

        # Verify dual checkpoint files were generated
        assert (ckpt_dir / "checkpoint_latest.pt").is_file()
        assert (ckpt_dir / "checkpoint_best.pt").is_file()
        assert (ckpt_dir / "checkpoint_best_pneumonia.pt").is_file()

