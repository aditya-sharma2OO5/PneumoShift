"""Label harmonization across NIH ChestX-ray14 and Stanford CheXpert datasets."""

from typing import Dict, List, Literal, Optional
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

# Mapping from canonical shared name to CheXpert column name
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


def harmonize_nih_labels(df: pd.DataFrame, label_column: str = "Finding Labels") -> pd.DataFrame:
    """Harmonizes NIH ChestX-ray14 pipe-delimited label strings into binary indicator columns.

    Args:
        df: NIH metadata DataFrame.
        label_column: Column name containing pipe-separated findings (e.g., 'Cardiomegaly|Effusion').

    Returns:
        DataFrame with standardized binary 0/1 columns for all SHARED_CLASSES.
    """
    result_df = df.copy()

    # Drop non-frontal if View Position exists and requested
    if "View Position" in result_df.columns:
        result_df = result_df[result_df["View Position"].isin(["PA", "AP"])].copy()

    for col in SHARED_CLASSES:
        if col == "Pleural Effusion":
            # In NIH, it is listed as 'Effusion'
            result_df[col] = result_df[label_column].apply(
                lambda s: 1 if "Effusion" in str(s).split("|") else 0
            )
        else:
            result_df[col] = result_df[label_column].apply(
                lambda s: 1 if col in str(s).split("|") else 0
            )

    return result_df


def harmonize_chexpert_labels(
    df: pd.DataFrame,
    uncertainty_policy: UncertaintyPolicy = "u_zeros",
    drop_lateral: bool = True,
    view_column: str = "Frontal/Lateral",
) -> pd.DataFrame:
    """Harmonizes CheXpert labels according to specified uncertainty policy.

    CheXpert raw labels:
        1.0: positive
        0.0: negative
        -1.0: uncertain
        NaN/blank: not mentioned (mapped to 0.0)

    Uncertainty policies:
        - 'u_zeros': uncertain -> 0.0 (primary protocol, Tayebi Arasteh et al., 2023)
        - 'u_ones': uncertain -> 1.0 (ablation, Irvin et al., 2019)
        - 'u_ignore': uncertain -> -1.0 (masked in loss, Irvin et al., 2019)

    Args:
        df: CheXpert raw metadata DataFrame.
        uncertainty_policy: One of 'u_zeros', 'u_ones', 'u_ignore'.
        drop_lateral: Whether to filter out lateral radiographic views.
        view_column: Column name indicating Frontal vs Lateral view.

    Returns:
        DataFrame with standardized columns for all SHARED_CLASSES.
    """
    result_df = df.copy()

    if drop_lateral and view_column in result_df.columns:
        result_df = result_df[result_df[view_column] == "Frontal"].copy()

    for canonical_name, source_col in CHEXPERT_COLUMN_MAPPING.items():
        if source_col not in result_df.columns:
            result_df[canonical_name] = 0.0
            continue

        raw_series = result_df[source_col].fillna(0.0).astype(float)

        if uncertainty_policy == "u_zeros":
            # Map -1.0 -> 0.0, keep 1.0, 0.0
            result_df[canonical_name] = raw_series.apply(lambda v: 1.0 if v == 1.0 else 0.0)
        elif uncertainty_policy == "u_ones":
            # Map -1.0 -> 1.0, keep 1.0, 0.0
            result_df[canonical_name] = raw_series.apply(lambda v: 1.0 if v in (1.0, -1.0) else 0.0)
        elif uncertainty_policy == "u_ignore":
            # Keep -1.0 as -1.0 for loss masking
            result_df[canonical_name] = raw_series.apply(
                lambda v: 1.0 if v == 1.0 else (-1.0 if v == -1.0 else 0.0)
            )
        else:
            raise ValueError(f"Unknown uncertainty policy: {uncertainty_policy}")

    return result_df

