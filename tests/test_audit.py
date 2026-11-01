"""Tests for dataset distribution auditing, prevalences, demographics, and report generation."""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from pneumoshift.data.audit import (
    compute_demographic_stats,
    compute_intensity_statistics,
    compute_projection_stats,
    compute_split_prevalences,
    format_markdown_audit_report,
    run_data_audit,
)
from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES


def test_audit_prevalences_and_demographics():
    df = pd.DataFrame({
        "patient_id": ["P1", "P2", "P3", "P4"],
        "age": [30, 45, 60, -1],  # -1 is invalid/missing
        "sex": ["M", "F", "F", "M"],
        "view_position": ["PA", "PA", "AP", "PA"],
        "Pneumonia": [1.0, 0.0, 0.0, 0.0],
        "Atelectasis": [0.0, 1.0, -1.0, 0.0],  # 1 pos, 1 unc
        "No Finding": [0.0, 0.0, 0.0, 1.0],
    })

    prev = compute_split_prevalences(df, [HEADLINE_CLASS, "Atelectasis", "No Finding"])
    assert prev[HEADLINE_CLASS]["positive_count"] == 1
    assert prev[HEADLINE_CLASS]["positive_rate_percent"] == 25.0
    assert prev["Atelectasis"]["uncertain_count"] == 1

    demo = compute_demographic_stats(df)
    assert demo["age"]["mean"] == 45.0
    assert demo["age"]["valid_count"] == 3
    assert demo["sex"]["F"]["count"] == 2

    views = compute_projection_stats(df)
    assert views["PA"]["count"] == 3
    assert views["AP"]["count"] == 1


def test_intensity_statistics_and_report_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_p = tmp_path / "img1.png"

        # Create dummy image with known intensities
        arr = np.linspace(0, 255, 32 * 32, dtype=np.uint8).reshape((32, 32))
        Image.fromarray(arr).convert("RGB").save(img_p)

        df = pd.DataFrame({
            "image_id": ["img1.png"],
            "image_path": [str(img_p)],
            "patient_id": ["P1"],
            "age": [50],
            "sex": ["M"],
            "view_position": ["PA"],
            "Pneumonia": [1.0],
            "No Finding": [0.0],
        })

        stats = compute_intensity_statistics(df, images_dir=tmp_path, sample_size=1)
        assert stats["sampled_images_count"] == 1
        assert 0.0 <= stats["mean_intensity"] <= 1.0

        report = run_data_audit({"train": df}, images_dir=tmp_path, output_dir=tmp_path, generate_grid=True)
        assert "train" in report
        assert (tmp_path / "data_audit_report.json").is_file()
        assert (tmp_path / "data_audit_report.md").is_file()
        assert (tmp_path / "data_audit_sample_grid.png").is_file()

        md_text = format_markdown_audit_report(report)
        assert "PneumoShift Dataset Audit Report" in md_text
        assert "Pneumonia" in md_text

