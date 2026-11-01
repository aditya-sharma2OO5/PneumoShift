"""Tests for label harmonization across NIH ChestX-ray14 and CheXpert."""

import numpy as np
import pandas as pd
import pytest
from pneumoshift.data.harmonize_labels import (
    SHARED_CLASSES,
    clean_nih_age,
    get_evaluation_targets,
    harmonize_chexpert_labels,
    harmonize_nih_labels,
)


def test_nih_label_harmonization():
    raw_data = {
        "Image Index": ["001.png", "002.png", "003.png"],
        "Patient ID": [101, 102, 103],
        "Patient Age": ["058Y", "024Y", "414Y"],  # 414Y is outlier
        "Patient Gender": ["M", "F", "M"],
        "Finding Labels": ["Atelectasis|Effusion", "Pneumonia", "No Finding"],
        "View Position": ["PA", "AP", "PA"],
    }
    df = pd.DataFrame(raw_data)
    harmonized = harmonize_nih_labels(df)

    # Check shared classes present
    for col in SHARED_CLASSES:
        assert col in harmonized.columns

    # Pleural Effusion mapping from Effusion
    assert harmonized.loc[0, "Pleural Effusion"] == 1.0
    assert harmonized.loc[0, "Atelectasis"] == 1.0
    assert harmonized.loc[0, "Pneumonia"] == 0.0

    # Pneumonia
    assert harmonized.loc[1, "Pneumonia"] == 1.0
    assert harmonized.loc[1, "Cardiomegaly"] == 0.0

    # No Finding
    assert harmonized.loc[2, "No Finding"] == 1.0
    assert harmonized.loc[2, "Pneumonia"] == 0.0

    # Metadata columns
    assert harmonized.loc[0, "patient_id"] == "101"
    assert harmonized.loc[0, "sex"] == "M"
    assert harmonized.loc[0, "view_position"] == "PA"
    assert harmonized.loc[0, "age"] == 58.0
    assert harmonized.loc[2, "age"] == -1.0  # Outlier cleaned


def test_chexpert_uncertainty_policies():
    raw_data = {
        "Path": [
            "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
            "CheXpert-v1.0-small/train/patient00002/study1/view2_lateral.jpg",
        ],
        "Frontal/Lateral": ["Frontal", "Lateral"],
        "Age": [65, 45],
        "Sex": ["Male", "Female"],
        "AP/PA": ["AP", "Lateral"],
        "Pneumonia": [-1.0, 1.0],  # Uncertain on frontal
        "Atelectasis": [1.0, 1.0],  # Positive
        "Cardiomegaly": [None, 0.0],  # Blank
    }
    df = pd.DataFrame(raw_data)

    # Policy: U-Zeros (uncertain -> 0, lateral dropped)
    df_u_zeros = harmonize_chexpert_labels(df, uncertainty_policy="u_zeros", drop_lateral=True)
    assert len(df_u_zeros) == 1  # Lateral dropped
    assert df_u_zeros.iloc[0]["Pneumonia"] == 0.0
    assert df_u_zeros.iloc[0]["Atelectasis"] == 1.0
    assert df_u_zeros.iloc[0]["Cardiomegaly"] == 0.0
    assert df_u_zeros.iloc[0]["patient_id"] == "patient00001"
    assert df_u_zeros.iloc[0]["view_position"] == "AP"

    # Policy: U-Ones (uncertain -> 1)
    df_u_ones = harmonize_chexpert_labels(df, uncertainty_policy="u_ones", drop_lateral=True)
    assert df_u_ones.iloc[0]["Pneumonia"] == 1.0
    assert df_u_ones.iloc[0]["Atelectasis"] == 1.0

    # Policy: U-Ignore (uncertain -> -1 preserved for loss mask)
    df_u_ignore = harmonize_chexpert_labels(df, uncertainty_policy="u_ignore", drop_lateral=True)
    assert df_u_ignore.iloc[0]["Pneumonia"] == -1.0
    assert df_u_ignore.iloc[0]["Atelectasis"] == 1.0


def test_evaluation_target_mapping():
    y = np.array([1.0, 0.0, -1.0, 1.0, -1.0])

    y_neg = get_evaluation_targets(y, mode="uncertain_as_negative")
    assert np.array_equal(y_neg, np.array([1.0, 0.0, 0.0, 1.0, 0.0]))

    y_exc = get_evaluation_targets(y, mode="uncertain_excluded")
    assert np.array_equal(y_exc, np.array([1.0, 0.0, -1.0, 1.0, -1.0]))
