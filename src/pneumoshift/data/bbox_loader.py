"""Ground-truth bounding box loader and coordinate scaler for radiological datasets (NIH ChestX-ray14).

Parses official NIH BBox_List_2017.csv annotations, normalizes column formats,
scales coordinates to target model spatial resolutions, and provides synthetic
generators for smoke-test pipeline execution.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.data.bbox_loader")

# Candidate column names across NIH official releases and pre-processed variants
IMAGE_COL_CANDIDATES = ["Image Index", "image_index", "image_name", "image_id", "filename"]
LABEL_COL_CANDIDATES = ["Finding Label", "finding_label", "label", "finding"]
X_COL_CANDIDATES = ["Bbox [x", "x", "bbox_x", "x_min", "left"]
Y_COL_CANDIDATES = ["y", "bbox_y", "y_min", "top"]
W_COL_CANDIDATES = ["w", "bbox_w", "width"]
H_COL_CANDIDATES = ["h]", "h", "bbox_h", "height"]


def _match_column(df_columns: List[str], candidates: List[str]) -> Optional[str]:
    """Finds first matching column name from a list of candidates (case-insensitive)."""
    col_map = {c.lower().strip(): c for c in df_columns}
    for cand in candidates:
        cand_lower = cand.lower().strip()
        if cand_lower in col_map:
            return col_map[cand_lower]
    return None


def load_nih_bounding_boxes(
    csv_path: Union[str, Path],
    original_size: Tuple[int, int] = (1024, 1024),
    target_size: Optional[Tuple[int, int]] = None,
    finding_filter: Optional[str] = "Pneumonia",
) -> pd.DataFrame:
    """Loads and scales NIH ChestX-ray14 ground-truth bounding boxes.

    Args:
        csv_path: Path to BBox_List_2017.csv.
        original_size: Original image spatial resolution (H, W), default (1024, 1024).
        target_size: Optional target spatial resolution (H, W) to scale coordinates.
        finding_filter: Optional finding to filter (e.g. 'Pneumonia'). If None, loads all.

    Returns:
        pd.DataFrame with standardized columns: ['image_name', 'finding_label', 'x', 'y', 'w', 'h'].

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If mandatory coordinate or image columns cannot be identified.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Bounding box file not found at: {path}")

    df = pd.read_csv(path)
    cols = list(df.columns)

    img_col = _match_column(cols, IMAGE_COL_CANDIDATES)
    label_col = _match_column(cols, LABEL_COL_CANDIDATES)
    x_col = _match_column(cols, X_COL_CANDIDATES)
    y_col = _match_column(cols, Y_COL_CANDIDATES)
    w_col = _match_column(cols, W_COL_CANDIDATES)
    h_col = _match_column(cols, H_COL_CANDIDATES)

    missing = []
    if not img_col:
        missing.append("image_name")
    if not label_col:
        missing.append("finding_label")
    if not x_col or not y_col or not w_col or not h_col:
        missing.append("coordinates (x, y, w, h)")

    if missing:
        raise ValueError(
            f"Failed to identify mandatory bounding box columns in {path}. Missing: {missing}. Available: {cols}"
        )

    clean_df = pd.DataFrame(
        {
            "image_name": df[img_col].astype(str),
            "finding_label": df[label_col].astype(str).str.strip(),
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "w": pd.to_numeric(df[w_col], errors="coerce"),
            "h": pd.to_numeric(df[h_col], errors="coerce"),
        }
    )

    # Drop invalid rows with NaN coordinates or non-positive sizes
    clean_df = clean_df.dropna(subset=["x", "y", "w", "h"])
    clean_df = clean_df[(clean_df["w"] > 0) & (clean_df["h"] > 0)].copy()

    # Apply finding filter if requested
    if finding_filter is not None:
        clean_df = clean_df[clean_df["finding_label"].str.lower() == finding_filter.lower()].copy()

    orig_h, orig_w = original_size
    if target_size is not None:
        tgt_h, tgt_w = target_size
        scale_x = tgt_w / float(orig_w)
        scale_y = tgt_h / float(orig_h)

        clean_df["x"] = (clean_df["x"] * scale_x).round().astype(int)
        clean_df["y"] = (clean_df["y"] * scale_y).round().astype(int)
        clean_df["w"] = (clean_df["w"] * scale_x).round().astype(int)
        clean_df["h"] = (clean_df["h"] * scale_y).round().astype(int)

        # Coordinate boundary clamping
        clean_df["x"] = clean_df["x"].clip(0, tgt_w - 1)
        clean_df["y"] = clean_df["y"].clip(0, tgt_h - 1)
        clean_df["w"] = clean_df["w"].clip(1, tgt_w)
        clean_df["h"] = clean_df["h"].clip(1, tgt_h)
    else:
        clean_df["x"] = clean_df["x"].round().astype(int)
        clean_df["y"] = clean_df["y"].round().astype(int)
        clean_df["w"] = clean_df["w"].round().astype(int)
        clean_df["h"] = clean_df["h"].round().astype(int)

    logger.info(
        "Loaded %d bounding boxes from %s (finding: %s, scaled: %s)",
        len(clean_df),
        path.name,
        finding_filter or "all",
        target_size or "original",
    )
    return clean_df.reset_index(drop=True)


def get_bboxes_for_image(
    df: pd.DataFrame,
    image_name: str,
) -> List[Tuple[int, int, int, int]]:
    """Retrieves all bounding box tuples (x, y, w, h) for a specific image filename.

    Args:
        df: DataFrame loaded via load_nih_bounding_boxes.
        image_name: Target image filename to query.

    Returns:
        List of (x, y, w, h) coordinate tuples.
    """
    matches = df[df["image_name"] == str(image_name)]
    if matches.empty:
        return []
    return [
        (int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"]))
        for _, row in matches.iterrows()
    ]


def create_synthetic_bbox_dataset(
    output_path: Union[str, Path],
    num_samples: int = 20,
    image_size: int = 224,
    seed: int = 42,
) -> Path:
    """Generates synthetic bounding box CSV dataset for smoke testing and CI verification.

    Args:
        output_path: File path to save generated CSV.
        num_samples: Number of synthetic bounding box annotations to generate.
        image_size: Coordinate space resolution (assumed square).
        seed: Random seed for reproducibility.

    Returns:
        Path to generated CSV file.
    """
    rng = np.random.default_rng(seed)
    findings = ["Pneumonia", "Atelectasis", "Cardiomegaly", "Infiltrate", "Effusion"]

    records = []
    for i in range(num_samples):
        img_id = f"syn_patient_{i:04d}_001.png"
        finding = findings[i % len(findings)]

        # Generate realistic box coordinates centered in lung fields
        box_w = int(rng.integers(int(image_size * 0.15), int(image_size * 0.35)))
        box_h = int(rng.integers(int(image_size * 0.15), int(image_size * 0.35)))
        box_x = int(rng.integers(int(image_size * 0.1), max(int(image_size * 0.9 - box_w), int(image_size * 0.1 + 1))))
        box_y = int(rng.integers(int(image_size * 0.2), max(int(image_size * 0.8 - box_h), int(image_size * 0.2 + 1))))

        records.append(
            {
                "Image Index": img_id,
                "Finding Label": finding,
                "Bbox [x": box_x,
                "y": box_y,
                "w": box_w,
                "h]": box_h,
            }
        )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False)
    logger.info("Saved %d synthetic bounding box records to %s", len(df), out_file)
    return out_file

