"""Patient-level stratified splitting and leakage verification."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.utils.io import compute_file_sha256, save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.data.splits")


def verify_patient_leakage(
    splits: Dict[str, pd.DataFrame],
    patient_col: str = "patient_id",
) -> None:
    """Verifies that patient IDs are strictly disjoint across all provided splits.

    Raises:
        AssertionError: If any patient ID is present in more than one split partition.
    """
    seen_patients: Dict[str, str] = {}
    leakages: List[str] = []

    for split_name, df in splits.items():
        if df is None or patient_col not in df.columns:
            continue
        patients: Set[str] = set(df[patient_col].dropna().astype(str).unique())
        for pid in patients:
            if pid in seen_patients:
                leakages.append(
                    f"Patient ID '{pid}' found in both '{seen_patients[pid]}' and '{split_name}'"
                )
            else:
                seen_patients[pid] = split_name

    if leakages:
        msg = f"FATAL: Patient leakage detected across splits ({len(leakages)} violations):\n" + "\n".join(
            leakages[:10]
        )
        logger.error(msg)
        raise AssertionError(msg)

    logger.info("Zero patient leakage verified across all splits (%d unique patients).", len(seen_patients))


def create_nih_splits(
    metadata_csv: str | Path,
    train_val_list: str | Path,
    test_list: str | Path,
    output_dir: str | Path,
    val_patient_ratio: float = 0.15,
    target_cal_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generates patient-level disjoint splits for NIH ChestX-ray14."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    df = pd.read_csv(metadata_csv)
    if "Patient ID" in df.columns and "patient_id" not in df.columns:
        df["patient_id"] = df["Patient ID"].astype(str)

    with open(train_val_list, "r") as f:
        train_val_files = set(line.strip() for line in f if line.strip())
    with open(test_list, "r") as f:
        test_files = set(line.strip() for line in f if line.strip())

    train_val_df = df[df["Image Index"].isin(train_val_files)].copy()
    test_df = df[df["Image Index"].isin(test_files)].copy()

    # Partition train_val into train and val by patient
    train_val_patients = np.array(list(train_val_df["patient_id"].unique()))
    rng.shuffle(train_val_patients)

    n_val = int(len(train_val_patients) * val_patient_ratio)
    val_patients = set(train_val_patients[:n_val])
    train_patients = set(train_val_patients[n_val:])

    train_df = train_val_df[train_val_df["patient_id"].isin(train_patients)].copy()
    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)].copy()

    # Carve target calibration holdout from test set by patient
    test_patients = np.array(list(test_df["patient_id"].unique()))
    rng.shuffle(test_patients)
    n_cal = int(len(test_patients) * target_cal_ratio)
    cal_patients = set(test_patients[:n_cal])
    eval_test_patients = set(test_patients[n_cal:])

    cal_df = test_df[test_df["patient_id"].isin(cal_patients)].copy()
    eval_test_df = test_df[test_df["patient_id"].isin(eval_test_patients)].copy()

    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": eval_test_df,
        "target_cal": cal_df,
    }

    verify_patient_leakage(split_dfs, patient_col="patient_id")

    split_paths = {}
    manifest = {}
    for name, s_df in split_dfs.items():
        p = output_path / f"nih_{name}.csv"
        s_df.to_csv(p, index=False)
        split_paths[name] = p
        manifest[name] = {
            "path": str(p),
            "num_images": len(s_df),
            "num_patients": s_df["patient_id"].nunique(),
            "sha256": compute_file_sha256(p),
        }

    save_json(manifest, output_path / "nih_manifest.json")
    return split_paths


def create_chexpert_splits(
    train_csv: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    target_cal_ratio: float = 0.10,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generates patient-level disjoint splits for Stanford CheXpert."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    df = pd.read_csv(train_csv)

    # Extract patient ID from Path (e.g., 'CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg')
    if "patient_id" not in df.columns:
        df["patient_id"] = df["Path"].apply(lambda p: p.split("/")[2] if "patient" in p else "unknown")

    patients = np.array(list(df["patient_id"].unique()))
    rng.shuffle(patients)

    n_total = len(patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients_all = patients[n_train + n_val :]

    # Carve target calibration split from test partition by patient
    n_cal = int(len(test_patients_all) * target_cal_ratio)
    cal_patients = set(test_patients_all[:n_cal])
    eval_test_patients = set(test_patients_all[n_cal:])

    split_dfs = {
        "train": df[df["patient_id"].isin(train_patients)].copy(),
        "val": df[df["patient_id"].isin(val_patients)].copy(),
        "test": df[df["patient_id"].isin(eval_test_patients)].copy(),
        "target_cal": df[df["patient_id"].isin(cal_patients)].copy(),
    }

    verify_patient_leakage(split_dfs, patient_col="patient_id")

    split_paths = {}
    manifest = {}
    for name, s_df in split_dfs.items():
        p = output_path / f"chexpert_{name}.csv"
        s_df.to_csv(p, index=False)
        split_paths[name] = p
        manifest[name] = {
            "path": str(p),
            "num_images": len(s_df),
            "num_patients": s_df["patient_id"].nunique(),
            "sha256": compute_file_sha256(p),
        }

    save_json(manifest, output_path / "chexpert_manifest.json")
    return split_paths


def create_synthetic_dataset(
    output_dir: str | Path,
    num_samples: int = 30,
    image_size: int = 64,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generates synthetic images and metadata for end-to-end smoke testing."""
    out_dir = Path(output_dir)
    img_dir = out_dir / "images"
    splits_dir = out_dir / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    records = []

    # Assign 5 samples per patient (e.g., 6 patients)
    num_patients = max(3, num_samples // 5)
    for i in range(num_samples):
        pid = f"synthetic_pat_{i % num_patients:03d}"
        filename = f"synth_{i:04d}.png"
        img_path = img_dir / filename

        # Generate synthetic grayscale noise image with synthetic oval lung fields
        arr = rng.randint(40, 90, (image_size, image_size), dtype=np.uint8)
        # Mock anatomy
        y, x = np.ogrid[:image_size, :image_size]
        mask_left = ((x - image_size * 0.35) ** 2 + (y - image_size * 0.5) ** 2) < (image_size * 0.25) ** 2
        mask_right = ((x - image_size * 0.65) ** 2 + (y - image_size * 0.5) ** 2) < (image_size * 0.25) ** 2
        arr[mask_left | mask_right] = rng.randint(110, 180, size=np.count_nonzero(mask_left | mask_right))

        img = Image.fromarray(arr).convert("RGB")
        img.save(img_path)

        rec = {
            "image_id": filename,
            "patient_id": pid,
            "image_path": str(img_path.resolve()),
            "view_position": "PA" if i % 2 == 0 else "AP",
            "age": rng.randint(20, 80),
            "sex": "M" if i % 2 == 0 else "F",
        }
        for cls_name in SHARED_CLASSES:
            rec[cls_name] = int(rng.rand() > 0.7)
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metadata.csv", index=False)

    # Patient-partitioned splits
    patients = np.array(list(df["patient_id"].unique()))
    rng.shuffle(patients)

    n_train = max(1, int(len(patients) * 0.6))
    n_val = max(1, int(len(patients) * 0.2))

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train : n_train + n_val])
    test_p = set(patients[n_train + n_val :])
    if not test_p:
        test_p = val_p

    split_dfs = {
        "train": df[df["patient_id"].isin(train_p)].copy(),
        "val": df[df["patient_id"].isin(val_p)].copy(),
        "test": df[df["patient_id"].isin(test_p)].copy(),
    }

    paths = {}
    for name, s_df in split_dfs.items():
        p = splits_dir / f"synthetic_{name}.csv"
        s_df.to_csv(p, index=False)
        paths[name] = p

    return paths

