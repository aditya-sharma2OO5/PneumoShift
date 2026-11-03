"""Model instantiation factory supporting DenseNet121 and EfficientNet-B0."""

from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torchvision.models as tv_models
from pneumoshift.models.heads import MultiLabelClassificationHead
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.models.factory")


class PneumoShiftModel(nn.Module):
    """Wrapper encapsulating backbone, pooling, and swappable multi-label head."""

    def __init__(
        self,
        backbone: nn.Module,
        head: MultiLabelClassificationHead,
        architecture: str = "densenet121",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.architecture = architecture
        self._backbone_frozen = False

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if "densenet" in self.architecture:
            features = self.backbone(x)
            out = nn.functional.relu(features, inplace=False)
            pooled = nn.functional.adaptive_avg_pool2d(out, (1, 1)).flatten(1)
        elif "efficientnet" in self.architecture:
            features = self.backbone(x)
            pooled = nn.functional.adaptive_avg_pool2d(features, (1, 1)).flatten(1)
        else:
            pooled = self.backbone(x)
            if pooled.dim() > 2:
                pooled = nn.functional.adaptive_avg_pool2d(pooled, (1, 1)).flatten(1)

        return self.head(pooled, return_features=return_features)

    def train(self, mode: bool = True) -> "PneumoShiftModel":
        """Sets module in training mode, preserving backbone freeze and BatchNorm eval mode if frozen."""
        super().train(mode)
        if getattr(self, "_backbone_frozen", False) and mode:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for module in self.backbone.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                    module.eval()
        return self

    def freeze_backbone(self) -> None:
        """Freezes all backbone parameters for warm-up head training.

        Sets requires_grad=False on every backbone parameter AND switches
        BatchNorm (and similar normalization) layers to eval mode so that
        running statistics are not corrupted by the head-only warm-up data.
        """
        self._backbone_frozen = True
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Freeze BatchNorm running stats by switching to eval mode
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.eval()
        logger.info("Backbone frozen (incl. BN eval) for %s.", self.architecture)

    def unfreeze_backbone(self) -> None:
        """Unfreezes backbone parameters for end-to-end fine-tuning.

        Re-enables gradient computation for all backbone parameters AND
        restores BatchNorm layers to train mode so running statistics update.
        """
        self._backbone_frozen = False
        for param in self.backbone.parameters():
            param.requires_grad = True
        # Restore BatchNorm layers to training mode
        for module in self.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
                module.train()
        logger.info("Backbone unfrozen (incl. BN train) for %s.", self.architecture)


def build_model(
    architecture: str = "densenet121",
    num_classes: int = 8,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> PneumoShiftModel:
    """Instantiates deep learning model according to specification.

    Args:
        architecture: 'densenet121', 'efficientnet_b0', or 'efficientnet_b3'.
        num_classes: Number of target prediction classes. Defaults to 8.
        pretrained: Whether to load ImageNet pre-trained weights.
        dropout: Classification head dropout probability.

    Returns:
        PneumoShiftModel instance.
    """
    arch = architecture.lower()

    if arch == "densenet121":
        weights = tv_models.DenseNet121_Weights.DEFAULT if pretrained else None
        base_model = tv_models.densenet121(weights=weights)
        in_features = base_model.classifier.in_features
        backbone = base_model.features
        head = MultiLabelClassificationHead(in_features=in_features, num_classes=num_classes, dropout_rate=dropout)
        model = PneumoShiftModel(backbone=backbone, head=head, architecture=arch)

    elif arch == "efficientnet_b0":
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        base_model = tv_models.efficientnet_b0(weights=weights)
        in_features = base_model.classifier[1].in_features
        backbone = base_model.features
        head = MultiLabelClassificationHead(in_features=in_features, num_classes=num_classes, dropout_rate=dropout)
        model = PneumoShiftModel(backbone=backbone, head=head, architecture=arch)

    elif arch == "efficientnet_b3":
        weights = tv_models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base_model = tv_models.efficientnet_b3(weights=weights)
        in_features = base_model.classifier[1].in_features
        backbone = base_model.features
        head = MultiLabelClassificationHead(in_features=in_features, num_classes=num_classes, dropout_rate=dropout)
        model = PneumoShiftModel(backbone=backbone, head=head, architecture=arch)

    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    logger.info("Built model %s with num_classes=%d, pretrained=%s", arch, num_classes, pretrained)
    return model


def get_target_layer_for_xai(model: nn.Module, architecture: str) -> nn.Module:
    """Returns the final convolutional layer targeted for Grad-CAM.

    DenseNet121: final norm/dense block layer (features.norm5 or features[-1])
    EfficientNet: final conv feature map (features[-1])
    """
    arch = architecture.lower()
    if isinstance(model, PneumoShiftModel):
        backbone = model.backbone
    else:
        backbone = getattr(model, "features", model)

    if "densenet" in arch:
        if hasattr(backbone, "norm5"):
            return backbone.norm5
        return list(backbone.children())[-1]
    elif "efficientnet" in arch:
        return list(backbone.children())[-1]
    else:
        return list(backbone.children())[-1]

