"""Label harmonization and metadata standardization across NIH ChestX-ray14 and CheXpert."""

import re
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd

SHARED_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
    "No Finding",
]

PATHOLOGY_CLASSES: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
    "Pneumonia",
    "Pneumothorax",
]

HEADLINE_CLASS: str = "Pneumonia"

# CheXpert column mapping to canonical shared class names
CHEXPERT_COLUMN_MAPPING: Dict[str, str] = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Pleural Effusion": "Pleural Effusion",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "No Finding": "No Finding",
}

UncertaintyPolicy = Literal["u_zeros", "u_ones", "u_ignore"]
EvalUncertaintyMode = Literal["uncertain_as_negative", "uncertain_excluded"]


def clean_nih_age(val: Any) -> float:
    """Cleans NIH age string format (e.g. '058Y' -> 58.0) and filters biological outliers."""
    if pd.isna(val):
        return -1.0
    s = str(val).strip().rstrip("Yy")
    try:
        age_f = float(s)
        # NIH has some extreme erroneous outlier entries (e.g. 414 years)
        if 0.0 <= age_f <= 120.0:
            return age_f
        return -1.0
    except ValueError:
        return -1.0


def harmonize_nih_labels(
    df: pd.DataFrame,
    label_column: str = "Finding Labels",
    drop_lateral: bool = True,
) -> pd.DataFrame:
    """Harmonizes NIH ChestX-ray14 metadata into standardized columns and labels.

    Transforms:
        - Drops non-frontal views (PA / AP retained).
        - Maps 'Effusion' -> 'Pleural Effusion'.
        - Maps pipe-separated strings into 0/1 binary indicator columns.
        - Standardizes metadata columns: patient_id, age, sex, view_position, support_devices.
        - Explicitly calculates 'No Finding'.

    Args:
        df: Raw NIH DataFrame (from Data_Entry_2017_v2020.csv).
        label_column: Column name containing pipe-separated labels.
        drop_lateral: Keep only frontal projections (PA and AP).

    Returns:
        Clean DataFrame with SHARED_CLASSES and standardized metadata.
    """
    res = df.copy()

    # Filter frontal views only
    if drop_lateral and "View Position" in res.columns:
        res = res[res["View Position"].isin(["PA", "AP"])].copy()

    # Standardize metadata
    if "Patient ID" in res.columns:
        res["patient_id"] = res["Patient ID"].astype(str)
    elif "patient_id" not in res.columns:
        res["patient_id"] = res.index.astype(str)

    if "Image Index" in res.columns:
        res["image_id"] = res["Image Index"].astype(str)
        if "image_path" not in res.columns:
            res["image_path"] = res["Image Index"].astype(str)

    if "Patient Age" in res.columns:
        res["age"] = res["Patient Age"].apply(clean_nih_age)
    elif "age" not in res.columns:
        res["age"] = -1.0

    if "Patient Gender" in res.columns:
        res["sex"] = res["Patient Gender"].astype(str).str.upper().str.strip()
    elif "sex" not in res.columns:
        res["sex"] = "Unknown"

    if "View Position" in res.columns:
        res["view_position"] = res["View Position"].astype(str).str.upper().str.strip()
    elif "view_position" not in res.columns:
        res["view_position"] = "Unknown"

    if "Follow-up #" in res.columns:
        res["follow_up_num"] = pd.to_numeric(res["Follow-up #"], errors="coerce").fillna(0).astype(int)

    # Support devices is not annotated as a distinct category in NIH
    res["support_devices"] = 0.0

    # Harmonize the 8 shared labels
    for col in SHARED_CLASSES:
        if col == "Pleural Effusion":
            res[col] = res[label_column].apply(
                lambda s: 1.0 if "Effusion" in str(s).split("|") else 0.0
            )
        elif col == "No Finding":
            res[col] = res[label_column].apply(
                lambda s: 1.0 if "No Finding" in str(s).split("|") else 0.0
            )
        else:
            res[col] = res[label_column].apply(
                lambda s: 1.0 if col in str(s).split("|") else 0.0
            )

    # Ensure No Finding consistency: if any disease is positive, No Finding must be 0
    any_disease = res[PATHOLOGY_CLASSES].sum(axis=1) > 0
    res.loc[any_disease, "No Finding"] = 0.0
    res.loc[~any_disease & (res[label_column].str.contains("No Finding", na=False)), "No Finding"] = 1.0

    return res


def extract_chexpert_patient_id(path_str: str) -> str:
    """Extracts standardized patient ID (e.g. 'patient00001') from CheXpert path string."""
    match = re.search(r"patient\d+", str(path_str))
    if match:
        return match.group(0)
    # Fallback to directory split
    parts = str(path_str).replace("\\", "/").split("/")
    for part in parts:
        if part.startswith("patient"):
            return part
    return "unknown_patient"


def harmonize_chexpert_labels(
    df: pd.DataFrame,
    uncertainty_policy: UncertaintyPolicy = "u_zeros",
    drop_lateral: bool = True,
    view_column: str = "Frontal/Lateral",
) -> pd.DataFrame:
    """Harmonizes CheXpert labels according to specified uncertainty policy.

    CheXpert label values in raw CSV:
        1.0: positive
        0.0: negative
        -1.0: uncertain
        NaN / blank: unmentioned (mapped to 0.0)

    Uncertainty Policies:
        - 'u_zeros': uncertain (-1.0) -> 0.0 (primary protocol, Tayebi Arasteh et al., 2023)
        - 'u_ones': uncertain (-1.0) -> 1.0 (ablation, Irvin et al., 2019)
        - 'u_ignore': uncertain (-1.0) -> -1.0 (masked in loss, Irvin et al., 2019)

    Args:
        df: CheXpert raw metadata DataFrame.
        uncertainty_policy: One of 'u_zeros', 'u_ones', 'u_ignore'.
        drop_lateral: Whether to filter out lateral radiographic views.
        view_column: Column name indicating Frontal vs Lateral view.

    Returns:
        DataFrame with standardized metadata and SHARED_CLASSES columns.
    """
    res = df.copy()

    # Drop lateral views if requested
    if drop_lateral and view_column in res.columns:
        res = res[res[view_column] == "Frontal"].copy()

    # Standardize metadata
    if "Path" in res.columns:
        res["image_path"] = res["Path"].astype(str)
        res["image_id"] = res["Path"].apply(lambda p: str(p).split("/")[-1])
        res["patient_id"] = res["Path"].apply(extract_chexpert_patient_id)
    elif "patient_id" not in res.columns:
        res["patient_id"] = res.index.astype(str)

    if "Age" in res.columns:
        res["age"] = pd.to_numeric(res["Age"], errors="coerce").fillna(-1.0)
    elif "age" not in res.columns:
        res["age"] = -1.0

    if "Sex" in res.columns:
        res["sex"] = res["Sex"].astype(str).str.upper().str.strip()
    elif "sex" not in res.columns:
        res["sex"] = "Unknown"

    if "AP/PA" in res.columns:
        res["view_position"] = res["AP/PA"].fillna("Unknown").astype(str).str.upper().str.strip()
    elif "view_position" not in res.columns:
        res["view_position"] = "Unknown"

    if "Support Devices" in res.columns:
        res["support_devices"] = res["Support Devices"].fillna(0.0).apply(lambda v: 1.0 if v == 1.0 else 0.0)
    else:
        res["support_devices"] = 0.0

    # Harmonize the shared observation classes
    for canonical_name in SHARED_CLASSES:
        if canonical_name not in res.columns:
            res[canonical_name] = 0.0
            continue

        raw = res[canonical_name].fillna(0.0).astype(float)

        if uncertainty_policy == "u_zeros":
            # Map -1.0 -> 0.0, 1.0 -> 1.0, 0.0 -> 0.0
            res[canonical_name] = raw.apply(lambda v: 1.0 if v == 1.0 else 0.0)
        elif uncertainty_policy == "u_ones":
            # Map -1.0 -> 1.0, 1.0 -> 1.0, 0.0 -> 0.0
            res[canonical_name] = raw.apply(lambda v: 1.0 if v in (1.0, -1.0) else 0.0)
        elif uncertainty_policy == "u_ignore":
            # Preserve -1.0 for loss masking
            res[canonical_name] = raw.apply(
                lambda v: 1.0 if v == 1.0 else (-1.0 if v == -1.0 else 0.0)
            )
        else:
            raise ValueError(f"Unknown uncertainty policy: {uncertainty_policy}")

    # No Finding handling
    if "No Finding" in res.columns and uncertainty_policy != "u_ignore":
        any_disease = (res[PATHOLOGY_CLASSES] == 1.0).any(axis=1)
        res.loc[any_disease, "No Finding"] = 0.0

    return res


def get_evaluation_targets(
    y_true: np.ndarray,
    mode: EvalUncertaintyMode = "uncertain_as_negative",
) -> np.ndarray:
    """Prepares target array for evaluation according to specified uncertainty mode.

    Args:
        y_true: Target array containing 1.0, 0.0, or -1.0.
        mode:
            - 'uncertain_as_negative': treats -1.0 as 0.0 (binary negative).
            - 'uncertain_excluded': leaves -1.0 as is (to be masked out by evaluation metric).

    Returns:
        Processed targets np.ndarray.
    """
    yt = np.array(y_true, dtype=float).copy()
    if mode == "uncertain_as_negative":
        yt[yt == -1.0] = 0.0
    return yt
