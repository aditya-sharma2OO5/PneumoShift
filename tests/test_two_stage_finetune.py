"""Tests for two-stage fine-tuning: backbone freeze/unfreeze, BatchNorm handling,
gradient flow isolation, and differential learning rate param groups."""

import pytest
import torch
import torch.nn as nn

from pneumoshift.models.factory import PneumoShiftModel, build_model
from pneumoshift.models.heads import MultiLabelClassificationHead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model(num_classes: int = 4) -> PneumoShiftModel:
    """Build a small DenseNet121 wrapper for fast tests (uses pretrained=False)."""
    return build_model(architecture="densenet121", num_classes=num_classes, pretrained=False)


def _count_requires_grad(module: nn.Module) -> tuple[int, int]:
    """Returns (num_trainable_params, num_frozen_params) for *module*."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in module.parameters() if not p.requires_grad)
    return trainable, frozen


# ---------------------------------------------------------------------------
# Test: freeze_backbone
# ---------------------------------------------------------------------------

class TestFreezeBackbone:
    """Verify that freeze_backbone correctly disables grad + sets BN to eval."""

    def test_all_backbone_params_frozen(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        trainable, frozen = _count_requires_grad(model.backbone)
        assert trainable == 0, f"Expected 0 trainable backbone params, got {trainable}"
        assert frozen > 0

    def test_head_params_remain_trainable(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        trainable, _ = _count_requires_grad(model.head)
        assert trainable > 0, "Head params should remain trainable after freeze_backbone"

    def test_batchnorm_in_eval_mode(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        for module in model.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                assert not module.training, (
                    f"BatchNorm layer {module} should be in eval mode after freeze_backbone"
                )


# ---------------------------------------------------------------------------
# Test: unfreeze_backbone
# ---------------------------------------------------------------------------

class TestUnfreezeBackbone:
    """Verify that unfreeze_backbone re-enables grad + sets BN to train."""

    def test_all_backbone_params_unfrozen(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        model.unfreeze_backbone()
        trainable, frozen = _count_requires_grad(model.backbone)
        assert frozen == 0, f"Expected 0 frozen backbone params after unfreeze, got {frozen}"
        assert trainable > 0

    def test_head_params_still_trainable(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        model.unfreeze_backbone()
        trainable, _ = _count_requires_grad(model.head)
        assert trainable > 0

    def test_batchnorm_in_train_mode(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        model.unfreeze_backbone()
        for module in model.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                assert module.training, (
                    f"BatchNorm layer {module} should be in train mode after unfreeze_backbone"
                )


# ---------------------------------------------------------------------------
# Test: Gradient flow isolation
# ---------------------------------------------------------------------------

class TestGradientFlow:
    """Verify that gradients only flow to trainable parameters."""

    def test_no_backbone_gradients_when_frozen(self):
        model = _make_tiny_model(num_classes=2)
        model.freeze_backbone()
        model.train()

        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # Backbone params should have no gradient
        for name, param in model.backbone.named_parameters():
            assert param.grad is None or torch.all(param.grad == 0), (
                f"Backbone param '{name}' should have no gradient when frozen"
            )

        # Head params should have gradient
        for name, param in model.head.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Head param '{name}' should have gradient"
                )

    def test_all_gradients_flow_when_unfrozen(self):
        model = _make_tiny_model(num_classes=2)
        model.freeze_backbone()
        model.unfreeze_backbone()
        model.train()

        x = torch.randn(2, 3, 32, 32)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        # At least some backbone params should have nonzero gradient
        has_grad = any(
            param.grad is not None and param.grad.abs().sum() > 0
            for param in model.backbone.parameters()
            if param.requires_grad
        )
        assert has_grad, "Some backbone params should have gradient after unfreeze"


# ---------------------------------------------------------------------------
# Test: BatchNorm running stats stability
# ---------------------------------------------------------------------------

class TestBatchNormRunningStats:
    """Verify BN running mean/var are NOT updated during frozen phase."""

    def _get_first_bn(self, model: PneumoShiftModel) -> nn.BatchNorm2d | None:
        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                return module
        return None

    def test_bn_stats_stable_when_frozen(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        model.train()

        bn = self._get_first_bn(model)
        if bn is None:
            pytest.skip("No BatchNorm2d found in backbone")

        mean_before = bn.running_mean.clone()
        var_before = bn.running_var.clone()

        # Forward pass should NOT update running stats when BN is in eval mode
        x = torch.randn(4, 3, 32, 32)
        with torch.no_grad():
            model(x)

        assert torch.equal(bn.running_mean, mean_before), (
            "BN running_mean should not change during frozen phase"
        )
        assert torch.equal(bn.running_var, var_before), (
            "BN running_var should not change during frozen phase"
        )

    def test_bn_stats_update_when_unfrozen(self):
        model = _make_tiny_model()
        model.freeze_backbone()
        model.unfreeze_backbone()
        model.train()

        bn = self._get_first_bn(model)
        if bn is None:
            pytest.skip("No BatchNorm2d found in backbone")

        mean_before = bn.running_mean.clone()

        x = torch.randn(4, 3, 32, 32)
        model(x)

        # Running mean should update in train mode (almost certainly different)
        assert not torch.equal(bn.running_mean, mean_before), (
            "BN running_mean should update during unfrozen training"
        )


# ---------------------------------------------------------------------------
# Test: Differential learning rate parameter groups
# ---------------------------------------------------------------------------

class TestDifferentialLRParamGroups:
    """Verify optimizer param group setup for two-stage training."""

    def test_two_param_groups_created(self):
        model = _make_tiny_model()
        lr = 1e-4
        backbone_lr = 1e-5
        param_groups = [
            {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
            {"params": list(model.head.parameters()), "lr": lr, "name": "head"},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-5)

        assert len(optimizer.param_groups) == 2
        assert optimizer.param_groups[0]["name"] == "backbone"
        assert optimizer.param_groups[0]["lr"] == 0.0
        assert optimizer.param_groups[1]["name"] == "head"
        assert optimizer.param_groups[1]["lr"] == lr

    def test_backbone_lr_update_on_unfreeze(self):
        model = _make_tiny_model()
        lr = 1e-4
        backbone_lr = 1e-5
        param_groups = [
            {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
            {"params": list(model.head.parameters()), "lr": lr, "name": "head"},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=1e-5)

        # Simulate unfreeze: update backbone LR
        model.unfreeze_backbone()
        for pg in optimizer.param_groups:
            if pg.get("name") == "backbone":
                pg["lr"] = backbone_lr

        assert optimizer.param_groups[0]["lr"] == backbone_lr
        assert optimizer.param_groups[1]["lr"] == lr  # head unchanged

    def test_all_model_params_tracked(self):
        """Ensure optimizer tracks ALL model params (no orphans)."""
        model = _make_tiny_model()
        param_groups = [
            {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
            {"params": list(model.head.parameters()), "lr": 1e-4, "name": "head"},
        ]
        optimizer = torch.optim.AdamW(param_groups, lr=1e-4)

        opt_param_ids = set()
        for pg in optimizer.param_groups:
            for p in pg["params"]:
                opt_param_ids.add(id(p))

        model_param_ids = {id(p) for p in model.parameters()}
        orphans = model_param_ids - opt_param_ids
        assert len(orphans) == 0, (
            f"{len(orphans)} model parameters are not tracked by the optimizer"
        )


# ---------------------------------------------------------------------------
# Test: Model factory builds correct architectures
# ---------------------------------------------------------------------------

class TestModelFactory:
    """Verify build_model produces valid PneumoShiftModel instances."""

    @pytest.mark.parametrize("arch", ["densenet121", "efficientnet_b0"])
    def test_build_and_forward(self, arch: str):
        model = build_model(architecture=arch, num_classes=8, pretrained=False)
        assert isinstance(model, PneumoShiftModel)
        x = torch.randn(1, 3, 32, 32)
        logits = model(x)
        assert logits.shape == (1, 8)

    @pytest.mark.parametrize("arch", ["densenet121", "efficientnet_b0"])
    def test_freeze_unfreeze_cycle(self, arch: str):
        model = build_model(architecture=arch, num_classes=4, pretrained=False)
        model.freeze_backbone()
        _, frozen = _count_requires_grad(model.backbone)
        assert frozen > 0
        model.unfreeze_backbone()
        trainable, frozen = _count_requires_grad(model.backbone)
        assert frozen == 0 and trainable > 0

    def test_invalid_architecture_raises(self):
        with pytest.raises(ValueError, match="Unsupported architecture"):
            build_model(architecture="resnet50")

