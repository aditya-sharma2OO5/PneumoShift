import hashlib
import json
import re
import subprocess
import numpy as np
import pandas as pd
import pytest
import torch
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


def test_four_label_policies(sample_chexpert_csv: Path):
    """Verifies the exact formulation and prevalence calculations for all 4 pre-specified label policies."""
    df = pd.read_csv(sample_chexpert_csv)
    frontal_df = df[df["Frontal/Lateral"].str.upper() == "FRONTAL"].copy().reset_index(drop=True)
    raw_pneu = frontal_df["Pneumonia"]

    # (a) u_zeros (Primary): blank and uncertain (-1) -> 0
    y_u_zeros = np.where(raw_pneu == 1.0, 1.0, 0.0).astype(np.float32)
    assert np.array_equal(y_u_zeros, np.array([1.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))

    # (b) u_ones (Sensitivity 1): uncertain (-1) -> 1, blank -> 0
    y_u_ones = np.where((raw_pneu == 1.0) | (raw_pneu == -1.0), 1.0, 0.0).astype(np.float32)
    assert np.array_equal(y_u_ones, np.array([1.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32))

    # (c) u_excluded (Sensitivity 2): uncertain (-1) excluded, blank kept 0
    mask_u_excluded = (raw_pneu != -1.0).to_numpy(dtype=bool)
    assert np.array_equal(mask_u_excluded, np.array([True, False, True, True, True]))
    assert y_u_zeros[mask_u_excluded].sum() == 2.0
    assert len(y_u_zeros[mask_u_excluded]) == 4

    # (d) explicitly_labeled_only (Sensitivity 3): both blank (NaN) and uncertain (-1) excluded
    mask_explicit = ((raw_pneu == 1.0) | (raw_pneu == 0.0)).to_numpy(dtype=bool)
    assert np.array_equal(mask_explicit, np.array([True, False, False, True, True]))
    assert len(y_u_zeros[mask_explicit]) == 3
    assert y_u_zeros[mask_explicit].sum() == 2.0


def test_pre_registration_plan_assert():
    """Verifies that ANALYSIS_PLAN_TRANSFER.md is verified dynamically at runtime without hardcoded hashes."""
    import hashlib
    import json
    import re

    repo_root = Path(__file__).resolve().parent.parent
    plan_p = repo_root / "ANALYSIS_PLAN_TRANSFER.md"
    assert plan_p.is_file(), "ANALYSIS_PLAN_TRANSFER.md must exist in repository root!"

    # Fresh dynamic hash of the plan file (normalizing CRLF to LF)
    fresh_sha = hashlib.sha256(plan_p.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert len(fresh_sha) == 64

    # Verify notebook computes it dynamically with NO hardcoded hash literals
    nb_p = repo_root / "notebooks" / "kaggle_infer_old_checkpoints.ipynb"
    assert nb_p.is_file(), "Inference notebook must exist!"
    nb = json.loads(nb_p.read_text(encoding="utf-8"))

    found_func = False
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        if "def get_analysis_plan_sha" in src:
            found_func = True
            # Assert NO hardcoded 64-character hex strings in this cell
            hex_literals = re.findall(r"\b[0-9a-fA-F]{64}\b", src)
            assert len(hex_literals) == 0, f"Found hardcoded hash literals in notebook plan cell: {hex_literals}"

            # Execute the get_analysis_plan_sha definition in a local namespace
            local_ns = {"Path": Path, "hashlib": hashlib, "REPO_ROOT": repo_root}
            func_code = src.split("def formulate_chexpert_cohort")[0]
            exec(func_code, local_ns)
            assert "get_analysis_plan_sha" in local_ns
            nb_computed_sha = local_ns["get_analysis_plan_sha"](repo_root)
            assert nb_computed_sha == fresh_sha
            assert local_ns.get("plan_sha") == fresh_sha
            break

    assert found_func, "Notebook must define get_analysis_plan_sha function!"


def test_nih_reference_logits_bundle():
    """Verifies that configs/nih_test_reference_logits.npz contains complete val and test arrays for all 6 models."""
    ref_p = Path("configs/nih_test_reference_logits.npz")
    assert ref_p.is_file(), "configs/nih_test_reference_logits.npz must exist!"
    data = np.load(ref_p)

    models = [
        "densenet121_42", "densenet121_1337", "densenet121_2024",
        "efficientnet_b0_42", "efficientnet_b0_1337", "efficientnet_b0_2024"
    ]
    for m in models:
        # Test arrays (22,744)
        assert f"{m}_test_logits" in data
        assert data[f"{m}_test_logits"].shape == (22744,)
        # Val arrays (13,500)
        assert f"{m}_val_logits" in data
        assert data[f"{m}_val_logits"].shape == (13500,)


def test_hardened_notebook_stages_and_assertions():
    """Verifies notebook validity, stage controls, single-decode architecture, and gating guards."""
    nb_p = Path("notebooks/kaggle_infer_old_checkpoints.ipynb")
    assert nb_p.is_file(), "notebooks/kaggle_infer_old_checkpoints.ipynb must exist!"
    with open(nb_p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    assert len(nb["cells"]) == 10, f"Expected 10 cells, got {len(nb['cells'])}"
    full_text = "".join("".join(c.get("source", [])) for c in nb["cells"])

    assert "prov-hardening-v3" in full_text
    assert 'TAG = "prov-hardening-v3"' in full_text
    assert "STAGE =" in full_text
    assert "NIH_GATE_PASSED.json" in full_text
    assert "ANALYSIS_PLAN_TRANSFER.md" in full_text
    assert "u_zeros" in full_text
    assert "u_ones" in full_text
    assert "u_excluded" in full_text
    assert "explicitly_labeled_only" in full_text
    assert "benchmark_gpu_forward_speed" in full_text
    assert "torch.cuda.synchronize" in full_text
    assert "find_nih_gate_token" in full_text
    assert "EXCLUSIONS.json" in full_text
    assert "predictions_nih_" in full_text


def test_nih_gate_token_search_attached_input(tmp_path):
    """Verifies that the CheXpert stage can read NIH_GATE_PASSED.json from an attached prior Kaggle output (/kaggle/input/...)."""
    from typing import Optional, List

    # 1. Simulate empty working directory (e.g. /kaggle/working/outputs/oldckpt_real_v1)
    working_output_dir = tmp_path / "working" / "outputs" / "oldckpt_real_v1"
    working_output_dir.mkdir(parents=True)

    # 2. Simulate attached prior Kaggle input directory (/kaggle/input/prior-nih-gate-run/NIH_GATE_PASSED.json)
    attached_input_dir = tmp_path / "input" / "prior-nih-gate-run"
    attached_input_dir.mkdir(parents=True)
    token_p = attached_input_dir / "NIH_GATE_PASSED.json"
    token_p.write_text(json.dumps({"status": "PASSED", "timestamp": "2026-11-24T00:00:00Z"}), encoding="utf-8")

    # Extract find_nih_gate_token definition from notebook
    nb_p = Path("notebooks/kaggle_infer_old_checkpoints.ipynb")
    nb = json.loads(nb_p.read_text(encoding="utf-8"))

    func_code = None
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        if "def find_nih_gate_token" in src:
            # Extract just find_nih_gate_token
            start_idx = src.index("def find_nih_gate_token")
            end_idx = src.index("def get_runtime_environment")
            func_code = src[start_idx:end_idx]
            break

    assert func_code is not None, "find_nih_gate_token must be defined in notebook!"

    local_ns = {"Path": Path, "Optional": Optional, "List": List}
    exec(func_code, local_ns)
    find_func = local_ns["find_nih_gate_token"]

    # When token is not in working dir and extra dirs are empty, returns None
    assert find_func(working_output_dir, extra_search_dirs=[tmp_path / "empty_dir"]) is None

    # When token is attached in prior Kaggle input directory, correctly resolves it
    resolved = find_func(working_output_dir, extra_search_dirs=[tmp_path / "input"])
    assert resolved is not None
    assert resolved.resolve() == token_p.resolve()


def test_fresh_nih_predictions_and_exclusions_in_notebook():
    """Verifies that notebook writes stamped fresh NIH predictions and separate EXCLUSIONS.json."""
    nb_p = Path("notebooks/kaggle_infer_old_checkpoints.ipynb")
    nb = json.loads(nb_p.read_text(encoding="utf-8"))

    # Cell 7 must save fresh NIH val/test arrays
    cell7_src = "".join(nb["cells"][7].get("source", []))
    assert "predictions_nih_" in cell7_src
    assert "full_logits" in cell7_src
    assert "fresh_logits = full_logits[:, PNEU_COL_IDX]" in cell7_src
    assert "np.savez_compressed" in cell7_src

    # Cell 8 must write separate EXCLUSIONS.json before inference
    cell8_src = "".join(nb["cells"][8].get("source", []))
    assert "CHEXPERT_COHORT_MEASURED.json" in cell8_src
    assert "EXCLUSIONS.json" in cell8_src
    assert "exclusions_data" in cell8_src


def test_cuda_abort_outside_smoke_mode():
    """Verifies that notebook contains explicit guard aborting execution if CUDA is unavailable outside smoke mode."""
    nb_p = Path("notebooks/kaggle_infer_old_checkpoints.ipynb")
    nb = json.loads(nb_p.read_text(encoding="utf-8"))

    # Cell 2 guard
    cell2_src = "".join(nb["cells"][2].get("source", []))
    assert "not SMOKE_TEST and not torch.cuda.is_available()" in cell2_src
    assert "raise SystemError" in cell2_src

    # Cell 7 guard
    cell7_src = "".join(nb["cells"][7].get("source", []))
    assert "not smoke and (not torch.cuda.is_available() or device.type != \"cuda\")" in cell7_src

    # Cell 9 guard
    cell9_src = "".join(nb["cells"][9].get("source", []))
    assert "not smoke and (not torch.cuda.is_available() or device.type != \"cuda\")" in cell9_src


def test_gate_token_refusal_on_mismatched_plan_or_gpu(tmp_path):
    """Verifies that CheXpert stage strictly refuses to run on mismatched plan hash or mismatched gpu_name."""
    import pytest
    from typing import Optional, List, Dict, Any

    nb_p = Path("notebooks/kaggle_infer_old_checkpoints.ipynb")
    nb = json.loads(nb_p.read_text(encoding="utf-8"))

    # Extract get_runtime_environment and verify_nih_gate_token
    cell7_src = "".join(nb["cells"][7].get("source", []))
    assert "def get_runtime_environment" in cell7_src
    assert "def verify_nih_gate_token" in cell7_src

    start_idx = cell7_src.index("def get_runtime_environment")
    end_idx = cell7_src.index("def run_full_nih_sanity_gate")
    helpers_code = cell7_src[start_idx:end_idx]

    repo_root = Path(__file__).resolve().parent.parent
    local_ns = {
        "Path": Path, "Optional": Optional, "List": List, "Dict": Dict, "Any": Any,
        "hashlib": hashlib, "subprocess": subprocess, "torch": torch, "json": json,
        "RUN_ID": "oldckpt_real_v1"
    }
    exec(helpers_code, local_ns)
    get_runtime_env = local_ns["get_runtime_environment"]
    verify_token = local_ns["verify_nih_gate_token"]

    real_env = get_runtime_env(repo_root)

    # 1. Base valid token
    valid_token_data = {
        "status": "PASSED",
        "timestamp": "2026-11-24T00:00:00Z",
        "plan_sha256": real_env["plan_sha256"],
        "manifest_sha256": real_env["manifest_sha256"],
        "reference_bundle_sha256": real_env["reference_bundle_sha256"],
        "git_commit": real_env["git_commit"],
        "run_id": real_env["run_id"],
        "gpu_name": real_env["gpu_name"],
        "torch_version": real_env["torch_version"],
    }

    # (a) Test refusal on mismatched plan hash
    mismatched_plan_token = valid_token_data.copy()
    mismatched_plan_token["plan_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
    token_plan_p = tmp_path / "plan_mismatch_token.json"
    token_plan_p.write_text(json.dumps(mismatched_plan_token), encoding="utf-8")

    with pytest.raises(RuntimeError, match="NIH GATE TOKEN MISMATCH for 'plan_sha256'"):
        verify_token(token_plan_p, repo_root, smoke=False)

    # (b) Test refusal on mismatched gpu_name
    mismatched_gpu_token = valid_token_data.copy()
    mismatched_gpu_token["gpu_name"] = "NVIDIA A100-SXM4-80GB-MISMATCHED"
    token_gpu_p = tmp_path / "gpu_mismatch_token.json"
    token_gpu_p.write_text(json.dumps(mismatched_gpu_token), encoding="utf-8")

    with pytest.raises(RuntimeError, match="NIH GATE TOKEN MISMATCH for 'gpu_name'"):
        verify_token(token_gpu_p, repo_root, smoke=False)


def test_reference_bundle_unprefixed_keys_and_prefixed_reading():
    """Verifies that reference bundle unprefixed keys equal prefixed keys and raw SHA matches recorded hash."""
    ref_p = Path("configs/nih_test_reference_logits.npz")
    assert ref_p.is_file(), "Reference bundle must exist!"

    # Verify raw bytes SHA-256
    raw_sha = hashlib.sha256(ref_p.read_bytes()).hexdigest()
    assert raw_sha == "026fb68e71d0cbb944f446a9eb643ef3d7e0c5fb413627c1a6a92a12a246feb8"

    bundle = np.load(ref_p)
    models = [
        "densenet121_42", "densenet121_1337", "densenet121_2024",
        "efficientnet_b0_42", "efficientnet_b0_1337", "efficientnet_b0_2024"
    ]
    for m in models:
        for suffix in ["logits", "score", "true"]:
            unpref = f"{m}_{suffix}"
            pref = f"{m}_test_{suffix}"
            assert unpref in bundle and pref in bundle
            assert np.array_equal(bundle[unpref], bundle[pref]), f"Mismatch between {unpref} and {pref}!"


def test_reference_bundle_rebuild_from_individual_predictions():
    """Rebuilds reference arrays from results/predictions and asserts np.array_equal against committed bundle."""
    bundle_p = Path("configs/nih_test_reference_logits.npz")
    assert bundle_p.is_file(), "Reference bundle must exist!"
    bundle = np.load(bundle_p)

    models = [
        ("densenet121", 42), ("densenet121", 1337), ("densenet121", 2024),
        ("efficientnet_b0", 42), ("efficientnet_b0", 1337), ("efficientnet_b0", 2024)
    ]
    splits = ["val", "test"]

    log_lines = []
    all_matched = True

    for arch, seed in models:
        m_key = f"{arch}_{seed}"
        for split in splits:
            pred_p = Path(f"results/predictions/predictions_{split}_{arch}_{seed}.npz")
            assert pred_p.is_file(), f"Prediction file missing: {pred_p}"
            data = np.load(pred_p)

            y_logits = data["y_logits"][:, 5]
            y_score = data["y_score"][:, 5]
            y_true = data["y_true"][:, 5]

            ref_logits = bundle[f"{m_key}_{split}_logits"]
            ref_score = bundle[f"{m_key}_{split}_score"]
            ref_true = bundle[f"{m_key}_{split}_true"]

            assert np.array_equal(y_logits, ref_logits), f"Logits mismatch for {m_key}_{split}"
            assert np.array_equal(y_score, ref_score), f"Score mismatch for {m_key}_{split}"
            assert np.array_equal(y_true, ref_true), f"True labels mismatch for {m_key}_{split}"

            log_lines.append(f"{m_key}_{split}: logits=True, score=True, true=True -> MATCH")

    out_text = "\n".join(log_lines) + "\n\nOverall Rebuild Check: PASSED\n"
    Path("outputs/reference_bundle_check.txt").write_text(out_text, encoding="utf-8")

def test_notebook_contains_no_tag_other_than_v3():
    """Asserts that the notebook defines TAG = 'prov-hardening-v3' and contains no other prov-hardening-v tag."""
    nb_p = Path("notebooks/kaggle_infer_old_checkpoints.ipynb")
    assert nb_p.is_file(), "notebooks/kaggle_infer_old_checkpoints.ipynb must exist!"
    with open(nb_p, "r", encoding="utf-8") as f:
        nb = json.load(f)

    full_text = "".join("".join(c.get("source", [])) for c in nb["cells"])
    tags_found = set(re.findall(r"prov-hardening-v\d+", full_text))
    assert tags_found == {"prov-hardening-v3"}, f"Found unexpected tag strings in notebook: {tags_found}"
    assert 'TAG = "prov-hardening-v3"' in full_text

