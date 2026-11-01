"""Tests for ChestXrayDataset loading, ablation modes, and image caching."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.data.transforms import get_transforms


def test_dataset_multilabel_and_ablation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_path = tmp_path / "sample.png"

        # Create dummy image
        img = Image.fromarray(np.full((32, 32), 128, dtype=np.uint8)).convert("RGB")
        img.save(img_path)

        data = {
            "image_id": ["sample.png"],
            "image_path": [str(img_path)],
            "patient_id": ["P001"],
            "view_position": ["PA"],
            "age": [52.0],
            "sex": ["M"],
            "Pneumonia": [1.0],
            "Atelectasis": [-1.0],  # Uncertain
        }
        for c in SHARED_CLASSES:
            if c not in data:
                data[c] = [0.0]

        df = pd.DataFrame(data)

        # Standard Multi-Label Mode
        transform = get_transforms(split="val", image_size=32)
        ds_multi = ChestXrayDataset(df, transform=transform)

        item = ds_multi[0]
        assert item["image"].shape == (3, 32, 32)
        assert item["target"].shape == (8,)
        assert item["mask"].shape == (8,)
        # Atelectasis has label -1.0 so its mask should be 0.0
        atel_idx = SHARED_CLASSES.index("Atelectasis")
        assert item["mask"][atel_idx] == 0.0
        pneu_idx = SHARED_CLASSES.index("Pneumonia")
        assert item["mask"][pneu_idx] == 1.0

        # Headline Pneumonia-Only Ablation Mode
        ds_ablation = ChestXrayDataset(df, transform=transform, headline_only=True)
        item_abl = ds_ablation[0]
        assert item_abl["target"].shape == (1,)
        assert item_abl["target"][0] == 1.0


def test_dataset_disk_caching():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cache_path = tmp_path / "cache"
        img_path = tmp_path / "sample.png"

        img = Image.fromarray(np.full((32, 32), 128, dtype=np.uint8)).convert("RGB")
        img.save(img_path)

        df = pd.DataFrame({
            "image_id": ["sample.png"],
            "image_path": [str(img_path)],
            "patient_id": ["P001"],
            "Pneumonia": [1.0],
        })

        transform = get_transforms(split="val", image_size=32)
        ds = ChestXrayDataset(df, label_cols=["Pneumonia"], transform=transform, cache_dir=cache_path)

        # First load should generate cache file
        item1 = ds[0]
        cached_files = list(cache_path.glob("*.pt"))
        assert len(cached_files) == 1

        # Second load reads directly from cache
        item2 = ds[0]
        assert torch.allclose(item1["image"], item2["image"])

