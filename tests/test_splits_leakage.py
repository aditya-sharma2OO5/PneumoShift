"""Critical test: FAILS build if any patient appears in more than one partition."""

import pandas as pd
import pytest
from pneumoshift.data.splits import verify_patient_leakage


def test_zero_leakage_passes():
    train_df = pd.DataFrame({"patient_id": ["P001", "P002", "P003"]})
    val_df = pd.DataFrame({"patient_id": ["P004", "P005"]})
    test_df = pd.DataFrame({"patient_id": ["P006", "P007"]})
    target_cal_df = pd.DataFrame({"patient_id": ["P008"]})

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
        "target_cal": target_cal_df,
    }

    # Must pass without raising AssertionError
    verify_patient_leakage(splits, patient_col="patient_id")


def test_leakage_fails_build():
    # Intentionally introduce patient leakage between train and test
    train_df = pd.DataFrame({"patient_id": ["P001", "P002", "P003"]})
    val_df = pd.DataFrame({"patient_id": ["P004"]})
    test_df = pd.DataFrame({"patient_id": ["P003", "P005"]})  # P003 is leaked!

    splits = {
        "train": train_df,
        "val": val_df,
        "test": test_df,
    }

    with pytest.raises(AssertionError, match="Patient leakage detected"):
        verify_patient_leakage(splits, patient_col="patient_id")

