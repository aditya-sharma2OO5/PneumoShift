"""PyTorch Dataset implementations with preprocessing cache and pneumonia-only ablation support."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.data.datasets")


class ChestXrayDataset(Dataset):
    """Standardized PyTorch Dataset for NIH ChestX-ray14, CheXpert, and synthetic cohorts."""

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: Optional[str | Path] = None,
        image_col: str = "image_path",
        label_cols: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        headline_only: bool = False,
        cache_dir: Optional[str | Path] = None,
    ) -> None:
        """Initializes the dataset.

        Args:
            df: DataFrame containing metadata and label columns.
            images_dir: Base directory for image resolution if paths are relative.
            image_col: Column name containing image path or filename.
            label_cols: List of label column names. Defaults to SHARED_CLASSES.
            transform: Optional torchvision transform pipeline.
            headline_only: If True, operates in pneumonia-only single-label ablation mode.
            cache_dir: Optional directory to cache resized/preprocessed image tensors.
        """
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir) if images_dir else None
        self.image_col = image_col
        self.transform = transform
        self.headline_only = headline_only
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if self.headline_only:
            self.label_cols = [HEADLINE_CLASS]
        else:
            self.label_cols = label_cols or SHARED_CLASSES

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

    def _load_image(self, img_path: Path, idx: int) -> Image.Image:
        """Loads image, reading from preprocessed cache if available."""
        if self.cache_dir:
            cache_file = self.cache_dir / f"cached_{idx}_{img_path.stem}.pt"
            if cache_file.is_file():
                try:
                    return torch.load(cache_file)
                except Exception:
                    pass

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise FileNotFoundError(f"Failed to load chest X-ray image at {img_path}: {e}")

        return image

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        raw_path = str(row.get(self.image_col, row.get("image_id", "")))
        img_path = self._resolve_image_path(raw_path)

        image_obj = self._load_image(img_path, idx)

        if isinstance(image_obj, torch.Tensor):
            image_tensor = image_obj
        elif self.transform:
            image_tensor = self.transform(image_obj)
        else:
            arr = np.array(image_obj, dtype=np.float32) / 255.0
            image_tensor = torch.tensor(arr).permute(2, 0, 1)

        # Save to cache if cache_dir is enabled and not already a cached tensor
        if self.cache_dir and not isinstance(image_obj, torch.Tensor):
            cache_file = self.cache_dir / f"cached_{idx}_{img_path.stem}.pt"
            if not cache_file.is_file():
                try:
                    torch.save(image_tensor, cache_file)
                except Exception:
                    pass

        # Extract target labels
        labels = [float(row[c]) if c in row and pd.notna(row[c]) else 0.0 for c in self.label_cols]
        target_tensor = torch.tensor(labels, dtype=torch.float32)

        # Mask for U-Ignore ablation (1.0 for valid labels, 0.0 for uncertain/ignored -1.0)
        mask_tensor = (target_tensor != -1.0).float()

        metadata = {
            "patient_id": str(row.get("patient_id", "")),
            "image_id": str(row.get("image_id", img_path.name)),
            "image_path": str(img_path),
            "view_position": str(row.get("view_position", "Unknown")),
            "age": float(row.get("age", -1.0)),
            "sex": str(row.get("sex", "Unknown")),
            "support_devices": float(row.get("support_devices", 0.0)),
        }

        return {
            "image": image_tensor,
            "target": target_tensor,
            "mask": mask_tensor,
            "metadata": metadata,
        }
