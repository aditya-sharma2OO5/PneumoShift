"""Unit tests for Phase 7 real dataset onboarding, verification, and extraction CLI."""

from pathlib import Path
import zipfile
import pandas as pd
import pytest

from scripts.setup_real_data import (
    ArchiveSecurityError,
    check_kaggle_credentials,
    create_subsampled_cohort,
    run_real_data_setup,
    safe_extract_archive,
    verify_dataset_installation,
)


def test_check_kaggle_credentials():
    has_creds, msg = check_kaggle_credentials()
    assert isinstance(has_creds, bool)
    assert isinstance(msg, str)
    assert len(msg) > 0


def test_verify_dataset_installation_empty(tmp_path):
    res = verify_dataset_installation(tmp_path)
    assert res["all_ready"] is False
    assert res["nih"]["installed"] is False
    assert res["chexpert"]["installed"] is False
    assert res["nih"]["images_count"] == 0


def test_safe_extract_archive_zip(tmp_path):
    # Create sample zip
    zip_path = tmp_path / "sample.zip"
    extract_target = tmp_path / "extracted"

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("test_file_1.txt", "PneumoShift test content 1")
        zf.writestr("subfolder/test_file_2.txt", "PneumoShift test content 2")

    count = safe_extract_archive(zip_path, extract_target)
    assert count == 2
    assert (extract_target / "test_file_1.txt").is_file()
    assert (extract_target / "subfolder" / "test_file_2.txt").is_file()


def test_safe_extract_archive_path_traversal(tmp_path):
    # Malicious zip with path traversal member
    zip_path = tmp_path / "malicious.zip"
    extract_target = tmp_path / "safe_dir"

    with zipfile.ZipFile(zip_path, "w") as zf:
        # Member with traversal outside destination
        zf.writestr("../../../evil.txt", "malicious payload")

    with pytest.raises(ArchiveSecurityError, match="Malicious archive member detected"):
        safe_extract_archive(zip_path, extract_target)


def test_create_subsampled_cohort(tmp_path):
    input_csv = tmp_path / "full_cohort.csv"
    output_csv = tmp_path / "subsampled.csv"

    # Create dummy dataframe with multiple images per patient
    df = pd.DataFrame({
        "Patient ID": [101, 101, 102, 103, 103, 104, 105, 105, 106, 107],
        "Image Index": [f"img_{i}.png" for i in range(10)],
        "Pneumonia": [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    })
    df.to_csv(input_csv, index=False)

    saved = create_subsampled_cohort(
        input_csv=input_csv,
        output_csv=output_csv,
        target_samples=4,
        patient_id_col="Patient ID",
    )

    assert saved.is_file()
    sub_df = pd.read_csv(saved)
    assert len(sub_df) >= 4
    assert len(sub_df) < len(df)

    # Verify patient integrity: each patient in sub_df has all their images included
    for pid in sub_df["Patient ID"].unique():
        orig_count = len(df[df["Patient ID"] == pid])
        sub_count = len(sub_df[sub_df["Patient ID"] == pid])
        assert orig_count == sub_count, f"Patient {pid} images were split or dropped"


def test_run_real_data_setup_smoke_test(tmp_path):
    res = run_real_data_setup(data_dir=tmp_path, smoke_test=True)

    assert res["overall_success"] is True
    assert res["download_status"] == "SMOKE_TEST_SYNTHETIC"
    assert res["extraction_status"] == "SMOKE_TEST_SYNTHETIC"

    # Verify mock directory structure was populated
    nih_img = tmp_path / "nih_cxr" / "images"
    assert nih_img.is_dir()
    assert len(list(nih_img.glob("*.png"))) == 5

    chexpert_csv = tmp_path / "chexpert" / "CheXpert-v1.0-small" / "train.csv"
    assert chexpert_csv.is_file()


def test_run_real_data_setup_verify_only(tmp_path):
    res = run_real_data_setup(data_dir=tmp_path, verify_only=True)
    assert res["download_status"] == "SKIPPED"
    assert res["extraction_status"] == "SKIPPED"
    assert "verification_before" in res
