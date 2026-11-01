"""Patient-level stratified splitting, zero-leakage verification, and manifest generation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from PIL import Image

from pneumoshift.data.harmonize_labels import (
    HEADLINE_CLASS,
    PATHOLOGY_CLASSES,
    SHARED_CLASSES,
    UncertaintyPolicy,
    harmonize_chexpert_labels,
    harmonize_nih_labels,
)
from pneumoshift.utils.io import compute_file_sha256, save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.data.splits")


def verify_patient_leakage(
    splits: Dict[str, pd.DataFrame],
    patient_col: str = "patient_id",
) -> None:
    """Verifies that patient IDs are strictly disjoint across all provided splits.

    Args:
        splits: Mapping of split name to DataFrame.
        patient_col: Column name containing unique patient identifier.

    Raises:
        AssertionError: If any patient ID is present in more than one partition.
    """
    seen_patients: Dict[str, str] = {}
    leakages: List[str] = []
    patient_counts: Dict[str, int] = {}

    for split_name, df in splits.items():
        if df is None or len(df) == 0:
            continue
        if patient_col not in df.columns:
            raise KeyError(f"Column '{patient_col}' not found in split '{split_name}'. Available: {list(df.columns)}")

        unique_pids: Set[str] = set(df[patient_col].dropna().astype(str).unique())
        patient_counts[split_name] = len(unique_pids)

        for pid in unique_pids:
            if pid in seen_patients:
                leakages.append(
                    f"Patient ID '{pid}' is leaked between '{seen_patients[pid]}' and '{split_name}'"
                )
            else:
                seen_patients[pid] = split_name

    if leakages:
        msg = (
            f"FATAL: Patient leakage detected across splits ({len(leakages)} violations)!\n"
            + "\n".join(leakages[:10])
        )
        logger.error(msg)
        raise AssertionError(msg)

    counts_str = ", ".join(f"{k}: {v} patients" for k, v in patient_counts.items())
    logger.info("Zero patient leakage verified across %d total unique patients (%s).", len(seen_patients), counts_str)


def create_nih_splits(
    metadata_csv: str | Path,
    train_val_list: str | Path,
    test_list: str | Path,
    output_dir: str | Path,
    val_patient_ratio: float = 0.15,
    target_cal_ratio: float = 0.10,
    seed: int = 42,
    images_dir: Optional[str | Path] = None,
) -> Dict[str, Path]:
    """Generates patient-level disjoint splits for NIH ChestX-ray14.

    Args:
        metadata_csv: Path to Data_Entry_2017_v2020.csv.
        train_val_list: Path to official train_val_list.txt.
        test_list: Path to official test_list.txt.
        output_dir: Directory to save generated split CSVs and manifest.
        val_patient_ratio: Fraction of train_val patients reserved for validation (default 0.15).
        target_cal_ratio: Fraction of test patients reserved for target calibration oracle (default 0.10).
        seed: Random seed for deterministic patient partitioning.
        images_dir: Optional base directory containing image files.

    Returns:
        Dict mapping split names to saved CSV file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    raw_df = pd.read_csv(metadata_csv)
    # Harmonize labels and metadata columns
    df = harmonize_nih_labels(raw_df, drop_lateral=True)

    if images_dir:
        img_base = Path(images_dir)
        df["image_path"] = df["image_id"].apply(lambda fname: str(img_base / fname))

    with open(train_val_list, "r", encoding="utf-8") as f:
        train_val_files = set(line.strip() for line in f if line.strip())
    with open(test_list, "r", encoding="utf-8") as f:
        test_files = set(line.strip() for line in f if line.strip())

    train_val_df = df[df["image_id"].isin(train_val_files)].copy()
    test_full_df = df[df["image_id"].isin(test_files)].copy()

    # Partition train_val into train and val by patient ID
    train_val_patients = np.array(sorted(train_val_df["patient_id"].unique()))
    rng.shuffle(train_val_patients)

    n_val = int(len(train_val_patients) * val_patient_ratio)
    val_patients = set(train_val_patients[:n_val])
    train_patients = set(train_val_patients[n_val:])

    train_df = train_val_df[train_val_df["patient_id"].isin(train_patients)].copy()
    val_df = train_val_df[train_val_df["patient_id"].isin(val_patients)].copy()

    # Partition test_full into test (eval) and target_cal by patient ID
    test_patients = np.array(sorted(test_full_df["patient_id"].unique()))
    rng.shuffle(test_patients)

    n_cal = int(len(test_patients) * target_cal_ratio)
    cal_patients = set(test_patients[:n_cal])
    eval_test_patients = set(test_patients[n_cal:])

    cal_df = test_full_df[test_full_df["patient_id"].isin(cal_patients)].copy()
    eval_test_df = test_full_df[test_full_df["patient_id"].isin(eval_test_patients)].copy()

    split_dfs = {
        "train": train_df,
        "val": val_df,
        "test": eval_test_df,
        "target_cal": cal_df,
    }

    # Verify zero patient overlap
    verify_patient_leakage(split_dfs, patient_col="patient_id")

    split_paths = {}
    manifest = {
        "dataset": "NIH ChestX-ray14",
        "seed": seed,
        "splits": {},
    }

    for name, s_df in split_dfs.items():
        p = output_path / f"nih_{name}.csv"
        s_df.to_csv(p, index=False)
        split_paths[name] = p
        manifest["splits"][name] = {
            "path": str(p),
            "num_images": len(s_df),
            "num_patients": s_df["patient_id"].nunique(),
            "pneumonia_prevalence": float((s_df[HEADLINE_CLASS] == 1.0).mean()) if HEADLINE_CLASS in s_df else 0.0,
            "sha256": compute_file_sha256(p),
        }

    save_json(manifest, output_path / "nih_manifest.json")
    logger.info("NIH splits created and hashed at %s", output_path)
    return split_paths


def create_chexpert_splits(
    train_csv: str | Path,
    output_dir: str | Path,
    official_valid_csv: Optional[str | Path] = None,
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    target_cal_ratio: float = 0.10,
    uncertainty_policy: UncertaintyPolicy = "u_zeros",
    seed: int = 42,
) -> Dict[str, Path]:
    """Generates patient-level disjoint splits for Stanford CheXpert.

    Args:
        train_csv: Path to CheXpert train.csv.
        output_dir: Directory to store generated split CSVs and manifest.
        official_valid_csv: Optional path to CheXpert official 200-study valid.csv.
        train_ratio: Ratio for training cohort (default 0.80).
        val_ratio: Ratio for validation cohort (default 0.10).
        test_ratio: Ratio for testing cohort (default 0.10).
        target_cal_ratio: Ratio of test patients reserved for target calibration (default 0.10).
        uncertainty_policy: 'u_zeros', 'u_ones', or 'u_ignore'.
        seed: Random seed for deterministic patient partitioning.

    Returns:
        Dict mapping split names to saved CSV file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    raw_df = pd.read_csv(train_csv)
    # Harmonize labels, drop lateral views, extract patient IDs
    df = harmonize_chexpert_labels(raw_df, uncertainty_policy=uncertainty_policy, drop_lateral=True)

    patients = np.array(sorted(df["patient_id"].unique()))
    rng.shuffle(patients)

    n_total = len(patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train : n_train + n_val])
    test_patients_all = patients[n_train + n_val :]

    n_cal = int(len(test_patients_all) * target_cal_ratio)
    cal_patients = set(test_patients_all[:n_cal])
    eval_test_patients = set(test_patients_all[n_cal:])

    split_dfs = {
        "train": df[df["patient_id"].isin(train_patients)].copy(),
        "val": df[df["patient_id"].isin(val_patients)].copy(),
        "test": df[df["patient_id"].isin(eval_test_patients)].copy(),
        "target_cal": df[df["patient_id"].isin(cal_patients)].copy(),
    }

    # Verify zero patient overlap
    verify_patient_leakage(split_dfs, patient_col="patient_id")

    split_paths = {}
    manifest = {
        "dataset": "Stanford CheXpert",
        "uncertainty_policy": uncertainty_policy,
        "seed": seed,
        "splits": {},
    }

    for name, s_df in split_dfs.items():
        p = output_path / f"chexpert_{name}.csv"
        s_df.to_csv(p, index=False)
        split_paths[name] = p
        manifest["splits"][name] = {
            "path": str(p),
            "num_images": len(s_df),
            "num_patients": s_df["patient_id"].nunique(),
            "pneumonia_prevalence": float((s_df[HEADLINE_CLASS] == 1.0).mean()) if HEADLINE_CLASS in s_df else 0.0,
            "sha256": compute_file_sha256(p),
        }

    # Harmonize official validation set if provided
    if official_valid_csv and Path(official_valid_csv).is_file():
        raw_val_df = pd.read_csv(official_valid_csv)
        val_df = harmonize_chexpert_labels(raw_val_df, uncertainty_policy=uncertainty_policy, drop_lateral=True)
        val_p = output_path / "chexpert_official_val.csv"
        val_df.to_csv(val_p, index=False)
        split_paths["official_val"] = val_p
        manifest["splits"]["official_val"] = {
            "path": str(val_p),
            "num_images": len(val_df),
            "num_patients": val_df["patient_id"].nunique(),
            "sha256": compute_file_sha256(val_p),
            "note": "Official 200-study radiologist-annotated validation benchmark",
        }

    save_json(manifest, output_path / "chexpert_manifest.json")
    logger.info("CheXpert splits created and hashed at %s", output_path)
    return split_paths


def create_synthetic_dataset(
    output_dir: str | Path,
    num_samples: int = 40,
    image_size: int = 64,
    seed: int = 42,
) -> Dict[str, Path]:
    """Generates synthetic chest radiographs and patient-disjoint metadata for dry-run testing.

    Args:
        output_dir: Directory where synthetic images and split CSVs will be created.
        num_samples: Total number of synthetic images to generate.
        image_size: Image dimension (square).
        seed: Random seed.

    Returns:
        Dict mapping split names to saved split CSV paths.
    """
    out_dir = Path(output_dir)
    img_dir = out_dir / "images"
    splits_dir = out_dir / "splits"
    img_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(seed)
    records = []

    # Assign images across at least 8 unique patients
    num_patients = max(8, num_samples // 4)
    for i in range(num_samples):
        pid = f"synth_patient_{i % num_patients:03d}"
        filename = f"synth_{i:04d}.png"
        img_path = img_dir / filename

        # Generate realistic synthetic chest X-ray simulation
        arr = rng.randint(20, 50, (image_size, image_size), dtype=np.uint8)
        y, x = np.ogrid[:image_size, :image_size]

        # Thoracic cavity outer contour
        chest_mask = ((x - image_size * 0.5) ** 2 / (image_size * 0.42) ** 2 + (y - image_size * 0.55) ** 2 / (image_size * 0.42) ** 2) <= 1.0
        arr[chest_mask] = rng.randint(70, 110, size=int(np.count_nonzero(chest_mask)))

        # Bilateral lung fields (lower radiopacity)
        left_lung = ((x - image_size * 0.33) ** 2 / (image_size * 0.16) ** 2 + (y - image_size * 0.52) ** 2 / (image_size * 0.30) ** 2) <= 1.0
        right_lung = ((x - image_size * 0.67) ** 2 / (image_size * 0.16) ** 2 + (y - image_size * 0.52) ** 2 / (image_size * 0.30) ** 2) <= 1.0
        arr[left_lung | right_lung] = rng.randint(30, 60, size=int(np.count_nonzero(left_lung | right_lung)))

        # Mediastinum / spine / heart shadow (higher radiopacity)
        spine = (np.abs(x - image_size * 0.5) < (image_size * 0.08)) & (y > image_size * 0.15) & (y < image_size * 0.90)
        heart = ((x - image_size * 0.45) ** 2 / (image_size * 0.15) ** 2 + (y - image_size * 0.60) ** 2 / (image_size * 0.18) ** 2) <= 1.0
        arr[spine | heart] = rng.randint(130, 200, size=int(np.count_nonzero(spine | heart)))

        img = Image.fromarray(arr).convert("RGB")
        img.save(img_path)

        rec = {
            "image_id": filename,
            "patient_id": pid,
            "image_path": str(img_path.resolve()),
            "view_position": "PA" if (i % 2 == 0) else "AP",
            "age": float(rng.randint(22, 79)),
            "sex": "M" if (i % 2 == 0) else "F",
            "support_devices": 1.0 if (rng.rand() > 0.8) else 0.0,
        }

        # Assign findings (ensure pneumonia is ~5% to 10% for testing)
        has_disease = False
        for cls_name in PATHOLOGY_CLASSES:
            if cls_name == HEADLINE_CLASS:
                val = 1.0 if (rng.rand() < 0.15) else 0.0
            else:
                val = 1.0 if (rng.rand() < 0.20) else 0.0
            rec[cls_name] = val
            if val == 1.0:
                has_disease = True

        rec["No Finding"] = 0.0 if has_disease else 1.0
        records.append(rec)

    df = pd.DataFrame(records)
    df.to_csv(out_dir / "metadata.csv", index=False)

    # Patient-partitioned splits
    patients = np.array(sorted(df["patient_id"].unique()))
    rng.shuffle(patients)

    n_patients = len(patients)
    n_train = max(2, int(n_patients * 0.50))
    n_val = max(1, int(n_patients * 0.20))
    n_test = max(1, int(n_patients * 0.20))

    train_p = set(patients[:n_train])
    val_p = set(patients[n_train : n_train + n_val])
    test_p = set(patients[n_train + n_val : n_train + n_val + n_test])
    cal_p = set(patients[n_train + n_val + n_test :])
    if not cal_p:
        cal_p = test_p

    split_dfs = {
        "train": df[df["patient_id"].isin(train_p)].copy(),
        "val": df[df["patient_id"].isin(val_p)].copy(),
        "test": df[df["patient_id"].isin(test_p)].copy(),
        "target_cal": df[df["patient_id"].isin(cal_p)].copy(),
        "official_val": df[df["patient_id"].isin(val_p)].copy(),
    }

    # Verify zero patient leakage between train, val, and test
    verify_patient_leakage({"train": split_dfs["train"], "val": split_dfs["val"], "test": split_dfs["test"]})

    paths = {}
    manifest = {"dataset": "Synthetic Smoke Test", "seed": seed, "splits": {}}
    for name, s_df in split_dfs.items():
        p = splits_dir / f"synthetic_{name}.csv"
        s_df.to_csv(p, index=False)
        paths[name] = p
        manifest["splits"][name] = {
            "path": str(p),
            "num_images": len(s_df),
            "num_patients": s_df["patient_id"].nunique(),
            "sha256": compute_file_sha256(p),
        }

    save_json(manifest, splits_dir / "synthetic_manifest.json")
    return paths
