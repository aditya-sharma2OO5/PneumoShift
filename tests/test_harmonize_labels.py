"""Tests for label harmonization across NIH ChestX-ray14 and CheXpert."""

import pandas as pd
import pytest
from pneumoshift.data.harmonize_labels import (
    SHARED_CLASSES,
    harmonize_chexpert_labels,
    harmonize_nih_labels,
)


def test_nih_label_harmonization():
    raw_data = {
        "Image Index": ["001.png", "002.png", "003.png"],
        "Finding Labels": ["Atelectasis|Effusion", "Pneumonia", "No Finding"],
        "View Position": ["PA", "AP", "PA"],
    }
    df = pd.DataFrame(raw_data)
    harmonized = harmonize_nih_labels(df)

    assert "Pleural Effusion" in harmonized.columns
    assert harmonized.loc[0, "Pleural Effusion"] == 1
    assert harmonized.loc[0, "Atelectasis"] == 1
    assert harmonized.loc[0, "Pneumonia"] == 0

    assert harmonized.loc[1, "Pneumonia"] == 1
    assert harmonized.loc[1, "Cardiomegaly"] == 0

    assert harmonized.loc[2, "No Finding"] == 1
    assert harmonized.loc[2, "Pneumonia"] == 0


def test_chexpert_uncertainty_policies():
    raw_data = {
        "Path": ["CheXpert-v1.0-small/train/patient0001/study1/view1_frontal.jpg"],
        "Frontal/Lateral": ["Frontal"],
        "Pneumonia": [-1.0],  # Uncertain
        "Atelectasis": [1.0],  # Positive
        "Cardiomegaly": [None],  # Blank
    }
    df = pd.DataFrame(raw_data)

    # Policy: U-Zeros (uncertain -> 0)
    df_u_zeros = harmonize_chexpert_labels(df, uncertainty_policy="u_zeros")
    assert df_u_zeros.loc[0, "Pneumonia"] == 0.0
    assert df_u_zeros.loc[0, "Atelectasis"] == 1.0
    assert df_u_zeros.loc[0, "Cardiomegaly"] == 0.0

    # Policy: U-Ones (uncertain -> 1)
    df_u_ones = harmonize_chexpert_labels(df, uncertainty_policy="u_ones")
    assert df_u_ones.loc[0, "Pneumonia"] == 1.0
    assert df_u_ones.loc[0, "Atelectasis"] == 1.0

    # Policy: U-Ignore (uncertain -> -1)
    df_u_ignore = harmonize_chexpert_labels(df, uncertainty_policy="u_ignore")
    assert df_u_ignore.loc[0, "Pneumonia"] == -1.0
    assert df_u_ignore.loc[0, "Atelectasis"] == 1.0

