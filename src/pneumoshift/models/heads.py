"""Classification heads with penultimate feature extraction support."""

from typing import Optional, Tuple
import torch
import torch.nn as nn


class MultiLabelClassificationHead(nn.Module):
    """Linear classification head with dropout and optional feature extraction."""

    def __init__(
        self,
        in_features: int,
        num_classes: int = 8,
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.dropout(x)
        logits = self.linear(features)
        if return_features:
            return logits, x
        return logits

