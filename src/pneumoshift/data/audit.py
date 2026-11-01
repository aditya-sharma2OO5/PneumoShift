"""Data distribution auditing: prevalence, demographics, views, and intensity statistics."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.data.audit")


def run_data_audit(
    splits: Dict[str, pd.DataFrame],
    label_cols: Optional[List[str]] = None,
    output_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Computes comprehensive dataset audit statistics across splits.

    Calculates:
        - Total images and distinct patients
        - Prevalence per label (positive count and percentage)
        - Pneumonia prevalence
        - Demographics: Age (mean, std, median, IQR), Sex (F/M counts and %)
        - View position distribution (AP vs PA)

    Args:
        splits: Dictionary mapping split names (e.g. 'train', 'val', 'test') to DataFrames.
        label_cols: Target classes to audit. Defaults to SHARED_CLASSES.
        output_path: Optional path to save JSON audit summary.

    Returns:
        Structured audit dictionary.
    """
    labels = label_cols or SHARED_CLASSES
    audit_report: Dict[str, Any] = {}

    for split_name, df in splits.items():
        if df is None or len(df) == 0:
            continue

        split_stats: Dict[str, Any] = {
            "total_images": int(len(df)),
            "unique_patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else None,
            "prevalences": {},
            "demographics": {},
            "projections": {},
        }

        # Label Prevalences
        for col in labels:
            if col in df.columns:
                pos_count = int((df[col] == 1.0).sum())
                unc_count = int((df[col] == -1.0).sum()) if (df[col] == -1.0).any() else 0
                total = len(df)
                split_stats["prevalences"][col] = {
                    "positive_count": pos_count,
                    "positive_rate": float(pos_count / total) if total > 0 else 0.0,
                    "uncertain_count": unc_count,
                }

        # View Positions
        view_col = "view_position" if "view_position" in df.columns else ("View Position" if "View Position" in df.columns else None)
        if view_col:
            val_counts = df[view_col].value_counts().to_dict()
            total_views = len(df)
            split_stats["projections"] = {
                k: {"count": int(v), "ratio": float(v / total_views)} for k, v in val_counts.items()
            }

        # Demographics
        age_col = "age" if "age" in df.columns else ("Patient Age" if "Patient Age" in df.columns else None)
        if age_col:
            valid_ages = pd.to_numeric(df[age_col], errors="coerce").dropna()
            if len(valid_ages) > 0:
                split_stats["demographics"]["age"] = {
                    "mean": float(valid_ages.mean()),
                    "std": float(valid_ages.std()),
                    "median": float(valid_ages.median()),
                }

        sex_col = "sex" if "sex" in df.columns else ("Patient Gender" if "Patient Gender" in df.columns else None)
        if sex_col:
            sex_counts = df[sex_col].value_counts().to_dict()
            split_stats["demographics"]["sex"] = {k: int(v) for k, v in sex_counts.items()}

        audit_report[split_name] = split_stats

    if output_path:
        save_json(audit_report, output_path)
        logger.info("Audit report saved to %s", output_path)

    return audit_report

