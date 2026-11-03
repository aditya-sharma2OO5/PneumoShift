"""Unit tests for radiological bounding box loader and coordinate scaling."""

from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from pneumoshift.data.bbox_loader import (
    load_nih_bounding_boxes,
    get_bboxes_for_image,
    create_synthetic_bbox_dataset,
)


class TestSyntheticBBoxDataset:
    """Tests synthetic bounding box dataset generation."""

    def test_create_synthetic_dataset_structure(self, tmp_path: Path):
        csv_path = tmp_path / "test_bboxes.csv"
        out_file = create_synthetic_bbox_dataset(csv_path, num_samples=15, image_size=224, seed=42)
        assert out_file.is_file()

        df = pd.read_csv(out_file)
        assert len(df) == 15
        expected_cols = {"Image Index", "Finding Label", "Bbox [x", "y", "w", "h]"}
        assert expected_cols.issubset(set(df.columns))
        assert "Pneumonia" in df["Finding Label"].values


class TestLoadNIHBoundingBoxes:
    """Tests loading, parsing, scaling, and filtering of bounding boxes."""

    def test_load_with_original_coordinates(self, tmp_path: Path):
        csv_path = tmp_path / "bboxes.csv"
        create_synthetic_bbox_dataset(csv_path, num_samples=20, image_size=1024, seed=123)

        df = load_nih_bounding_boxes(csv_path, original_size=(1024, 1024), target_size=None, finding_filter=None)
        assert not df.empty
        assert set(df.columns) == {"image_name", "finding_label", "x", "y", "w", "h"}
        assert (df["x"] >= 0).all()
        assert (df["y"] >= 0).all()
        assert (df["w"] > 0).all()
        assert (df["h"] > 0).all()

    def test_coordinate_scaling_to_target_size(self, tmp_path: Path):
        csv_path = tmp_path / "scale_test.csv"
        # Create a single box with known coordinates on 1024x1024
        df_manual = pd.DataFrame(
            [
                {
                    "Image Index": "test_001.png",
                    "Finding Label": "Pneumonia",
                    "Bbox [x": 512,
                    "y": 256,
                    "w": 128,
                    "h": 256,
                }
            ]
        )
        df_manual.to_csv(csv_path, index=False)

        # Scale down to 256x256 (factor of 4)
        scaled_df = load_nih_bounding_boxes(
            csv_path, original_size=(1024, 1024), target_size=(256, 256), finding_filter="Pneumonia"
        )
        assert len(scaled_df) == 1
        row = scaled_df.iloc[0]
        assert row["x"] == 128
        assert row["y"] == 64
        assert row["w"] == 32
        assert row["h"] == 64

    def test_finding_filter(self, tmp_path: Path):
        csv_path = tmp_path / "filter_test.csv"
        create_synthetic_bbox_dataset(csv_path, num_samples=25, seed=42)

        pneu_df = load_nih_bounding_boxes(csv_path, finding_filter="Pneumonia")
        assert not pneu_df.empty
        assert (pneu_df["finding_label"].str.lower() == "pneumonia").all()

        all_df = load_nih_bounding_boxes(csv_path, finding_filter=None)
        assert len(all_df) >= len(pneu_df)

    def test_get_bboxes_for_image(self, tmp_path: Path):
        csv_path = tmp_path / "query_test.csv"
        df_manual = pd.DataFrame(
            [
                {"Image Index": "img_A.png", "Finding Label": "Pneumonia", "Bbox [x": 10, "y": 20, "w": 30, "h": 40},
                {"Image Index": "img_A.png", "Finding Label": "Pneumonia", "Bbox [x": 50, "y": 60, "w": 70, "h": 80},
                {"Image Index": "img_B.png", "Finding Label": "Pneumonia", "Bbox [x": 15, "y": 25, "w": 35, "h": 45},
            ]
        )
        df_manual.to_csv(csv_path, index=False)
        loaded = load_nih_bounding_boxes(csv_path, finding_filter=None)

        boxes_a = get_bboxes_for_image(loaded, "img_A.png")
        assert len(boxes_a) == 2
        assert boxes_a[0] == (10, 20, 30, 40)
        assert boxes_a[1] == (50, 60, 70, 80)

        boxes_b = get_bboxes_for_image(loaded, "img_B.png")
        assert len(boxes_b) == 1
        assert boxes_b[0] == (15, 25, 35, 45)

        boxes_c = get_bboxes_for_image(loaded, "non_existent.png")
        assert boxes_c == []

    def test_missing_file_raises_not_found(self):
        with pytest.raises(FileNotFoundError, match="Bounding box file not found"):
            load_nih_bounding_boxes("non_existent_bbox_file_xyz.csv")

    def test_missing_mandatory_columns_raises_value_error(self, tmp_path: Path):
        bad_csv = tmp_path / "bad.csv"
        df_bad = pd.DataFrame([{"RandomCol1": 123, "RandomCol2": 456}])
        df_bad.to_csv(bad_csv, index=False)

        with pytest.raises(ValueError, match="Failed to identify mandatory bounding box columns"):
            load_nih_bounding_boxes(bad_csv)

