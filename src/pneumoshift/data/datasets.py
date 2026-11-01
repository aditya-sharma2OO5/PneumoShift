"""PyTorch Dataset implementations for chest X-ray cohorts."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from pneumoshift.data.harmonize_labels import SHARED_CLASSES


class ChestXrayDataset(Dataset):
    """Standardized PyTorch Dataset for NIH ChestX-ray14, CheXpert, and synthetic data."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Optional[str | Path] = None,
        image_col: str = "image_path",
        label_cols: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            df: DataFrame containing metadata and label columns.
            images_dir: Base directory for image resolution if paths in df are relative.
            image_col: Name of column containing image file path or filename.
            label_cols: List of label column names. Defaults to SHARED_CLASSES.
            transform: Optional torchvision transform pipeline.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir) if images_dir else None
        self.image_col = image_col
        self.label_cols = label_cols or SHARED_CLASSES
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_image_path(self, raw_path: str) -> Path:
        p = Path(raw_path)
        if p.is_file():
            return p
        if self.images_dir and (self.images_dir / p.name).is_file():
            return self.images_dir / p.name
        if self.images_dir and (self.images_dir / p).is_file():
            return self.images_dir / p
        return p

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self._resolve_image_path(str(row[self.image_col]))

        # Read image and convert to 3 channels (grayscale duplicated across RGB)
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load image at {img_path}: {e}")

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Extract target labels
        labels = [float(row[c]) if c in row and pd.notna(row[c]) else 0.0 for c in self.label_cols]
        target_tensor = torch.tensor(labels, dtype=torch.float32)

        # For u_ignore, create mask where label != -1.0
        mask_tensor = (target_tensor != -1.0).float()

        metadata = {
            "patient_id": str(row.get("patient_id", "")),
            "image_path": str(img_path),
            "view_position": str(row.get("view_position", row.get("View Position", "Unknown"))),
            "age": float(row.get("age", row.get("Patient Age", -1))),
            "sex": str(row.get("sex", row.get("Patient Gender", "Unknown"))),
        }

        return {
            "image": image_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "metadata": metadata,
        }

