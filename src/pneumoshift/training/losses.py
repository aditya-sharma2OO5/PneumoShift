"""Loss functions with inverse frequency weighting and uncertain label masking."""

from typing import List, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pneumoshift.data.harmonize_labels import SHARED_CLASSES


def compute_class_weights(df: pd.DataFrame, label_cols: Optional[List[str]] = None) -> torch.Tensor:
    """Computes inverse positive class frequency weights for BCE loss.

    Formula:
        pos_weight_c = (N_total - N_pos_c) / max(N_pos_c, 1)

    Args:
        df: Training split DataFrame.
        label_cols: List of target classes.

    Returns:
        torch.Tensor of length len(label_cols).
    """
    labels = label_cols or SHARED_CLASSES
    weights = []
    total = len(df)

    for col in labels:
        if col in df.columns:
            # Only count confirmed positive cases (1.0)
            pos_count = (df[col] == 1.0).sum()
            neg_count = total - pos_count
            w = float(neg_count / max(pos_count, 1))
            # Clip extreme weights to avoid gradient explosion
            w = min(max(w, 0.1), 100.0)
        else:
            w = 1.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


class MaskedBCEWithLogitsLoss(nn.Module):
    """Multi-label BCEWithLogitsLoss with support for inverse frequency pos_weight

    and loss masking (e.g. for U-Ignore uncertainty ablation).
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.register_buffer("pos_weight", pos_weight if pos_weight is not None else None)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:

            logits: Raw network predictions of shape (B, C).
            targets: Binary target indicators of shape (B, C).
            mask: Binary mask of shape (B, C) where 0 indicates ignored labels.

        Returns:
            Scalar loss tensor.
        """
        # Ensure target values are clamped to [0, 1] in case -1 is passed
        clamped_targets = torch.clamp(targets, min=0.0, max=1.0)

        loss_elements = nn.functional.binary_cross_entropy_with_logits(
            logits,
            clamped_targets,
            pos_weight=self.pos_weight,
            reduction="none",
        )

        if mask is not None:
            loss_elements = loss_elements * mask
            total_elements = mask.sum()
            if total_elements > 0:
                return loss_elements.sum() / total_elements
            return loss_elements.sum()

        return loss_elements.mean()

