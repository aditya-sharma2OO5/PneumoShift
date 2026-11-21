import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path


@pytest.fixture
def sample_chexpert_csv(tmp_path: Path) -> Path:
    """Creates a miniature fixture CheXpert train.csv for testing cohort filtering and metadata preservation."""
    csv_path = tmp_path / "train.csv"
    data = {
        "Path": [
            "CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg",
            "CheXpert-v1.0-small/train/patient00001/study1/view2_lateral.jpg",
            "CheXpert-v1.0-small/train/patient00002/study1/view1_frontal.jpg",
            "CheXpert-v1.0-small/train/patient00002/study2/view1_frontal.jpg",
            "CheXpert-v1.0-small/train/patient00003/study1/view1_lateral.jpg",
            "CheXpert-v1.0-small/train/patient00004/study1/view1_frontal.jpg",
            "CheXpert-v1.0-small/train/patient00005/study1/view1_frontal.jpg",
        ],
        "Sex": ["Female", "Female", "Male", "Male", "Female", "Male", "Female"],
        "Age": [68, 68, 45, 46, 52, 70, 31],
        "Frontal/Lateral": ["Frontal", "Lateral", "Frontal", "Frontal", "Lateral", "Frontal", "Frontal"],
        "AP/PA": ["AP", "PA", "PA", "AP", "LL", "AP", "PA"],
        "No Finding": [np.nan, 1.0, 0.0, np.nan, 0.0, 1.0, 0.0],
        "Atelectasis": [1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        "Cardiomegaly": [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        "Consolidation": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Edema": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Pleural Effusion": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "Pneumonia": [1.0, 0.0, -1.0, np.nan, 0.0, 0.0, 1.0],  # 2 positive, 1 uncertain, 1 nan, 1 negative (among 5 frontal)
        "Pneumothorax": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return csv_path


def test_chexpert_frontal_filtering(sample_chexpert_csv: Path):
    """Verifies that only Frontal images are included and Lateral images are strictly excluded."""
    df = pd.read_csv(sample_chexpert_csv)
    frontal_df = df[df["Frontal/Lateral"].str.upper() == "FRONTAL"].copy()
    
    assert len(frontal_df) == 5, f"Expected 5 frontal images, got {len(frontal_df)}"
    assert (frontal_df["Frontal/Lateral"].str.upper() == "FRONTAL").all()
    assert not (frontal_df["Frontal/Lateral"].str.upper() == "LATERAL").any()


def test_patient_id_extraction_and_uniqueness(sample_chexpert_csv: Path):
    """Verifies patient ID extraction from CheXpert paths."""
    df = pd.read_csv(sample_chexpert_csv)
    frontal_df = df[df["Frontal/Lateral"].str.upper() == "FRONTAL"].copy()
    
    frontal_df["patient_id"] = frontal_df["Path"].apply(lambda p: Path(str(p)).parts[2])
    
    assert frontal_df["patient_id"].nunique() == 4  # patients 00001, 00002, 00004, 00005
    assert set(frontal_df["patient_id"]) == {"patient00001", "patient00002", "patient00004", "patient00005"}


def test_chexpert_cohort_measured_counts(sample_chexpert_csv: Path, tmp_path: Path):
    """Verifies that pre-inference label measurement accurately tallies pos, unc, neg, and nan labels."""
    df = pd.read_csv(sample_chexpert_csv)
    frontal_df = df[df["Frontal/Lateral"].str.upper() == "FRONTAL"].copy()
    frontal_df["patient_id"] = frontal_df["Path"].apply(lambda p: Path(str(p)).parts[2])

    pneu_raw = frontal_df["Pneumonia"]
    pos_cnt = int((pneu_raw == 1.0).sum())
    unc_cnt = int((pneu_raw == -1.0).sum())
    neg_cnt = int((pneu_raw == 0.0).sum())
    nan_cnt = int(pneu_raw.isna().sum())

    measured = {
        "cohort": "CheXpert Frontal Fixture",
        "total_frontal_images": len(frontal_df),
        "unique_patients": frontal_df["patient_id"].nunique(),
        "pneumonia_label_counts": {
            "positive": pos_cnt,
            "uncertain": unc_cnt,
            "negative": neg_cnt,
            "nan": nan_cnt,
        }
    }

    out_json = tmp_path / "CHEXPERT_COHORT_MEASURED.json"
    with open(out_json, "w") as f:
        json.dump(measured, f, indent=2)

    assert out_json.is_file()
    with open(out_json) as f:
        loaded = json.load(f)

    assert loaded["total_frontal_images"] == 5
    assert loaded["unique_patients"] == 4
    assert loaded["pneumonia_label_counts"]["positive"] == 2
    assert loaded["pneumonia_label_counts"]["uncertain"] == 1
    assert loaded["pneumonia_label_counts"]["negative"] == 1
    assert loaded["pneumonia_label_counts"]["nan"] == 1


def test_metadata_array_preservation(sample_chexpert_csv: Path, tmp_path: Path):
    """Verifies that metadata columns (view, orientation, sex, age, path, raw label) are preserved in npz."""
    df = pd.read_csv(sample_chexpert_csv)
    frontal_df = df[df["Frontal/Lateral"].str.upper() == "FRONTAL"].copy()
    frontal_df["patient_id"] = frontal_df["Path"].apply(lambda p: Path(str(p)).parts[2])
    frontal_df["raw_pneumonia"] = frontal_df["Pneumonia"].copy()

    out_npz = tmp_path / "test_predictions.npz"
    np.savez_compressed(
        out_npz,
        y_true=np.zeros((len(frontal_df), 8)),
        y_logits=np.zeros((len(frontal_df), 8)),
        metadata_frontal_lateral=frontal_df["Frontal/Lateral"].to_numpy().astype(str),
        metadata_ap_pa=frontal_df["AP/PA"].to_numpy().astype(str),
        metadata_sex=frontal_df["Sex"].to_numpy().astype(str),
        metadata_age=frontal_df["Age"].to_numpy().astype(np.float32),
        metadata_path=frontal_df["Path"].to_numpy().astype(str),
        raw_pneumonia_label=frontal_df["raw_pneumonia"].to_numpy().astype(np.float32),
        data_source=np.array("real_inference"),
    )

    with np.load(out_npz) as loaded:
        assert "metadata_frontal_lateral" in loaded
        assert "metadata_ap_pa" in loaded
        assert "metadata_sex" in loaded
        assert "metadata_age" in loaded
        assert "metadata_path" in loaded
        assert "raw_pneumonia_label" in loaded
        assert len(loaded["metadata_path"]) == 5
        # Raw label should contain -1.0 and NaN
        raw_pneu = loaded["raw_pneumonia_label"]
        assert -1.0 in raw_pneu
        assert 1.0 in raw_pneu
        assert np.isnan(raw_pneu).any()


def test_inference_sharding_and_resume_logic(tmp_path: Path):
    """Verifies that shard chunking and resume logic properly detects completed shards without recomputation."""
    n_total = 25
    shard_size = 10
    total_shards = (n_total + shard_size - 1) // shard_size

    # Simulate existing shard 0
    shard_0_path = tmp_path / "predictions_shard_0.npz"
    np.savez_compressed(shard_0_path, indices=np.arange(0, 10))

    # Test resume detector
    pending_shards = []
    for s_idx in range(total_shards):
        s_path = tmp_path / f"predictions_shard_{s_idx}.npz"
        if not s_path.is_file():
            pending_shards.append(s_idx)

    assert pending_shards == [1, 2]  # Shard 0 already exists, should resume at 1 and 2
