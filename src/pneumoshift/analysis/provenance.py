"""Provenance loading and strict pre-analysis validation for transfer evaluation."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

ARCHITECTURES: Tuple[str, ...] = ("densenet121", "efficientnet_b0")
SEEDS: Tuple[int, ...] = (42, 1337, 2024)
PNEU_COL_IDX: int = 5
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
EXPECTED_PLAN_RUN_ID: str = "oldckpt_real_v1"
# Label-blind corrupted static-noise rows documented in DEVIATIONS.md (PRE-E-001)
CHEXPERT_SENSITIVITY_EXCLUDE_ROWS: Tuple[int, ...] = (68292, 76123)


def sha256_file(path: Path, lf_normalize: bool = False) -> str:
    """Computes hex SHA-256 of a file, optionally normalizing CRLF to LF."""
    raw = Path(path).read_bytes()
    if lf_normalize:
        raw = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(raw).hexdigest()


def _scalar_str(arr: Any) -> str:
    """Extracts stripped UTF-8 string from numpy 0D array or scalar."""
    if isinstance(arr, np.ndarray):
        arr = arr.item() if arr.ndim == 0 else str(arr)
    if isinstance(arr, bytes):
        return arr.decode("utf-8", "replace").strip()
    return str(arr).strip()


def _scalar_int(arr: Any) -> int:
    """Extracts Python int from numpy scalar/array."""
    if isinstance(arr, np.ndarray):
        return int(arr.item() if arr.ndim == 0 else arr.flat[0])
    return int(arr)


def load_manifest(manifest_path: Path) -> Tuple[str, Dict[str, Tuple[str, int]]]:
    """Loads checkpoint_manifest.json and returns (LF-SHA-256, sha256 -> (arch, seed))."""
    mp = Path(manifest_path)
    if not mp.is_file():
        raise FileNotFoundError(f"Checkpoint manifest missing: {mp}")
    manifest_sha = sha256_file(mp, lf_normalize=True)
    data = json.loads(mp.read_text(encoding="utf-8"))
    sha_map: Dict[str, Tuple[str, int]] = {}
    for entry in data.get("checkpoints", {}).values():
        sha = str(entry["sha256"]).strip().lower()
        arch = str(entry["architecture"]).strip().lower()
        seed = int(entry["seed"])
        sha_map[sha] = (arch, seed)
    if len(sha_map) != 6:
        raise ValueError(f"Expected 6 checkpoint hashes in manifest {mp}, got {len(sha_map)}")
    return manifest_sha, sha_map


def _validate_npz_common(
    npz: Any,
    filepath: Path,
    arch: str,
    seed: int,
    sha_map: Dict[str, Tuple[str, int]],
    expected_run_id: Optional[str] = None,
    expected_plan_sha: Optional[str] = None,
    expected_gpu: Optional[str] = None,
    expected_torch: Optional[str] = None,
) -> Dict[str, str]:
    """Validates common provenance stamps inside a loaded .npz file."""
    files = set(npz.files)
    if "data_source" not in files:
        raise ValueError(f"{filepath.name}: missing required 'data_source' stamp.")
    ds = _scalar_str(npz["data_source"])
    if ds != "real_inference":
        raise ValueError(f"{filepath.name}: invalid data_source={ds!r} (must be 'real_inference').")

    if "checkpoint_sha256" not in files:
        raise ValueError(f"{filepath.name}: missing required 'checkpoint_sha256' stamp.")
    ckpt_sha = _scalar_str(npz["checkpoint_sha256"]).lower()
    if ckpt_sha not in sha_map:
        raise ValueError(f"{filepath.name}: checkpoint_sha256={ckpt_sha!r} not authorized in manifest.")
    mapped = sha_map[ckpt_sha]
    if mapped != (arch, seed):
        raise ValueError(
            f"{filepath.name}: checkpoint SHA maps to {mapped}, expected {(arch, seed)}."
        )

    if "run_id" not in files:
        raise ValueError(f"{filepath.name}: missing required 'run_id' stamp.")
    run_id = _scalar_str(npz["run_id"])
    if expected_run_id is not None and run_id != expected_run_id:
        raise ValueError(f"{filepath.name}: run_id={run_id!r}, expected {expected_run_id!r}.")

    if "label_cols" in files:
        cols = [_scalar_str(c) for c in npz["label_cols"].tolist()]
        pidx = _scalar_int(npz["pneumonia_col_idx"]) if "pneumonia_col_idx" in files else PNEU_COL_IDX
        if pidx < 0 or pidx >= len(cols) or cols[pidx] != "Pneumonia":
            raise ValueError(f"{filepath.name}: label_cols[{pidx}] != 'Pneumonia' (got {cols}).")

    plan_sha = _scalar_str(npz["analysis_plan_sha256"]).lower() if "analysis_plan_sha256" in files else ""
    if expected_plan_sha is not None:
        if not plan_sha:
            raise ValueError(f"{filepath.name}: missing 'analysis_plan_sha256' stamp.")
        if plan_sha != expected_plan_sha:
            raise ValueError(
                f"{filepath.name}: analysis_plan_sha256={plan_sha!r} != {expected_plan_sha!r}."
            )

    gpu = _scalar_str(npz["gpu_name"]) if "gpu_name" in files else ""
    torch_v = _scalar_str(npz["torch_version"]) if "torch_version" in files else ""
    if expected_gpu is not None:
        if not gpu or gpu != expected_gpu:
            raise ValueError(f"{filepath.name}: gpu_name={gpu!r} != gate gpu_name={expected_gpu!r}.")
    if expected_torch is not None:
        if not torch_v or torch_v != expected_torch:
            raise ValueError(
                f"{filepath.name}: torch_version={torch_v!r} != gate torch_version={expected_torch!r}."
            )

    return {
        "checkpoint_sha256": ckpt_sha,
        "run_id": run_id,
        "data_source": ds,
        "analysis_plan_sha256": plan_sha,
        "gpu_name": gpu,
        "torch_version": torch_v,
    }


def _extract_pneumonia_logits(npz: Any, filepath: Path) -> np.ndarray:
    """Extracts 1D float64 raw logits for Pneumonia column index 5 (never uses y_score)."""
    key = "y_logits" if "y_logits" in npz.files else ("logits" if "logits" in npz.files else None)
    if key is None:
        raise ValueError(f"{filepath.name}: missing logit array ('y_logits' or 'logits').")
    raw = np.asarray(npz[key], dtype=np.float64)
    pidx = _scalar_int(npz["pneumonia_col_idx"]) if "pneumonia_col_idx" in npz.files else PNEU_COL_IDX
    if raw.ndim == 2:
        if raw.shape[1] <= pidx:
            raise ValueError(f"{filepath.name}: logits width {raw.shape[1]} <= pidx {pidx}.")
        z = raw[:, pidx].copy()
    elif raw.ndim == 1:
        z = raw.copy()
    else:
        raise ValueError(f"{filepath.name}: unexpected logits ndim={raw.ndim}.")
    if not np.isfinite(z).all():
        n_bad = int((~np.isfinite(z)).sum())
        raise ValueError(f"{filepath.name}: contains {n_bad} non-finite Pneumonia logit(s).")
    return z


def load_real_run_bundle(
    preds_dir: Path,
    plan_path: Path,
    manifest_path: Path,
    ref_bundle_path: Path,
    expected_run_id: str = EXPECTED_PLAN_RUN_ID,
    expected_git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    """Loads and cryptographically/provenance-validates the complete oldckpt_real_v1 run.

    Verifies:
      - NIH_GATE_PASSED.json status == 'PASSED', matching plan LF-SHA, manifest LF-SHA,
        reference bundle raw SHA-256, run_id, non-empty git_commit, gpu_name, torch_version.
      - EXCLUSIONS.json presence and >0.1% failed-image threshold status.
      - All 12 fresh NIH .npz (val, test x 6) and all 6 CheXpert .npz files have
        data_source == 'real_inference', matching plan SHA, matching run_id,
        valid checkpoint SHA mapped to (arch, seed), and identical gpu_name / torch_version.
      - Cross-model row alignment across all 6 checkpoints per split.
    """
    pdir = Path(preds_dir)
    if not pdir.is_dir():
        raise FileNotFoundError(f"Predictions directory does not exist: {pdir}")

    plan_sha = sha256_file(plan_path, lf_normalize=True)
    manifest_sha, sha_map = load_manifest(manifest_path)
    if not Path(ref_bundle_path).is_file():
        raise FileNotFoundError(f"Reference bundle missing: {ref_bundle_path}")
    ref_sha = sha256_file(ref_bundle_path, lf_normalize=False)

    gate_path = pdir / "NIH_GATE_PASSED.json"
    if not gate_path.is_file():
        raise ValueError(f"Refusing to run: {gate_path} is missing.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))

    if gate.get("status") != "PASSED":
        raise ValueError(f"Refusing to run: NIH gate status={gate.get('status')!r} != 'PASSED'.")
    if gate.get("smoke") is not False:
        raise ValueError(f"Refusing to run: NIH gate smoke={gate.get('smoke')!r} != False.")
    if gate.get("run_id") != expected_run_id:
        raise ValueError(f"Gate run_id={gate.get('run_id')!r} != {expected_run_id!r}.")
    if str(gate.get("plan_sha256", "")).lower() != plan_sha:
        raise ValueError("Gate plan_sha256 mismatch against ANALYSIS_PLAN_TRANSFER.md.")
    if str(gate.get("manifest_sha256", "")).lower() != manifest_sha:
        raise ValueError("Gate manifest_sha256 mismatch against configs/checkpoint_manifest.json.")
    if str(gate.get("reference_bundle_sha256", "")).lower() != ref_sha:
        raise ValueError("Gate reference_bundle_sha256 mismatch against configs/nih_test_reference_logits.npz.")

    gate_gpu = str(gate.get("gpu_name", "")).strip()
    gate_torch = str(gate.get("torch_version", "")).strip()
    gate_commit = str(gate.get("git_commit", "")).strip().lower()
    if not gate_gpu or not gate_torch:
        raise ValueError("Gate token missing gpu_name or torch_version.")
    if not gate_commit or gate_commit == "unknown":
        raise ValueError(f"Gate token has invalid git_commit={gate_commit!r}.")
    if expected_git_commit is not None and gate_commit != expected_git_commit.lower():
        raise ValueError(f"Gate git_commit={gate_commit!r} != expected {expected_git_commit!r}.")

    excl_path = pdir / "EXCLUSIONS.json"
    if not excl_path.is_file():
        raise ValueError(f"Refusing to run: {excl_path} is missing.")
    excl = json.loads(excl_path.read_text(encoding="utf-8"))
    n_excluded = int(excl.get("n_excluded", -1))
    if n_excluded < 0:
        raise ValueError(f"Invalid n_excluded in {excl_path}: {n_excluded}")

    models_data: Dict[str, Dict[str, Any]] = {}
    nih_meta: Dict[str, Dict[str, Any]] = {}
    chex_meta: Optional[Dict[str, Any]] = None

    for arch in ARCHITECTURES:
        for seed in SEEDS:
            mkey = f"{arch}_{seed}"
            models_data[mkey] = {"arch": arch, "seed": seed}

            # Load NIH val and test
            for split in ("val", "test"):
                p_nih = pdir / f"predictions_nih_{split}_{arch}_{seed}.npz"
                if not p_nih.is_file():
                    raise FileNotFoundError(f"Missing NIH {split} predictions: {p_nih}")
                with np.load(p_nih, allow_pickle=False) as zf:
                    _validate_npz_common(
                        zf,
                        p_nih,
                        arch=arch,
                        seed=seed,
                        sha_map=sha_map,
                        expected_run_id=expected_run_id,
                        expected_plan_sha=plan_sha,
                        expected_gpu=gate_gpu,
                        expected_torch=gate_torch,
                    )
                    logits = _extract_pneumonia_logits(zf, p_nih)
                    y_key = "y_true" if "y_true" in zf.files else "ground_truth"
                    y_true = np.asarray(zf[y_key], dtype=np.float64)
                    if y_true.ndim == 2:
                        y_true = y_true[:, PNEU_COL_IDX]
                    pids = np.array([_scalar_str(x) for x in zf["patient_ids"]], dtype=object)

                if split not in nih_meta:
                    nih_meta[split] = {"y_true": y_true, "patient_ids": pids}
                else:
                    if not np.array_equal(nih_meta[split]["y_true"], y_true):
                        raise ValueError(f"NIH {split} y_true row mismatch in {mkey}.")
                    if not np.array_equal(nih_meta[split]["patient_ids"], pids):
                        raise ValueError(f"NIH {split} patient_ids row mismatch in {mkey}.")

                models_data[mkey][f"nih_{split}_logits"] = logits

            # Load CheXpert frontal
            p_cx = pdir / f"predictions_chexpert_frontal_{arch}_{seed}.npz"
            if not p_cx.is_file():
                raise FileNotFoundError(f"Missing CheXpert predictions: {p_cx}")
            with np.load(p_cx, allow_pickle=False) as zf:
                _validate_npz_common(
                    zf,
                    p_cx,
                    arch=arch,
                    seed=seed,
                    sha_map=sha_map,
                    expected_run_id=expected_run_id,
                    expected_plan_sha=plan_sha,
                    expected_gpu=gate_gpu,
                    expected_torch=gate_torch,
                )
                cx_logits = _extract_pneumonia_logits(zf, p_cx)
                raw_pneu_key = (
                    "raw_pneumonia" if "raw_pneumonia" in zf.files else "raw_pneumonia_label"
                )
                raw_pneu = np.asarray(zf[raw_pneu_key], dtype=np.float64)
                pids = np.array([_scalar_str(x) for x in zf["patient_ids"]], dtype=object)
                row_hash = _scalar_str(zf["row_order_hash"]) if "row_order_hash" in zf.files else ""
                m_paths = (
                    np.array([_scalar_str(x) for x in zf["metadata_path"]], dtype=object)
                    if "metadata_path" in zf.files
                    else None
                )
                ap_pa = (
                    np.array([_scalar_str(x) for x in zf["metadata_ap_pa"]], dtype=object)
                    if "metadata_ap_pa" in zf.files
                    else None
                )
                sex = (
                    np.array([_scalar_str(x) for x in zf["metadata_sex"]], dtype=object)
                    if "metadata_sex" in zf.files
                    else None
                )
                age = (
                    np.asarray(zf["metadata_age"], dtype=np.float64)
                    if "metadata_age" in zf.files
                    else None
                )

                # Verify saved label policy masks match raw_pneumonia if present
                if "y_true_u_zeros" in zf.files:
                    expected_uz = np.where(raw_pneu == 1.0, 1.0, 0.0)
                    if not np.array_equal(expected_uz, np.asarray(zf["y_true_u_zeros"], dtype=np.float64)):
                        raise ValueError(f"{p_cx.name}: y_true_u_zeros inconsistent with raw_pneumonia.")
                if "y_true_u_ones" in zf.files:
                    expected_uo = np.where((raw_pneu == 1.0) | (raw_pneu == -1.0), 1.0, 0.0)
                    if not np.array_equal(expected_uo, np.asarray(zf["y_true_u_ones"], dtype=np.float64)):
                        raise ValueError(f"{p_cx.name}: y_true_u_ones inconsistent with raw_pneumonia.")

            if chex_meta is None:
                chex_meta = {
                    "raw_pneumonia": raw_pneu,
                    "patient_ids": pids,
                    "row_order_hash": row_hash,
                    "metadata_path": m_paths,
                    "view_position": ap_pa,
                    "sex": sex,
                    "age": age,
                }
            else:
                if row_hash != chex_meta["row_order_hash"]:
                    raise ValueError(f"CheXpert row_order_hash mismatch in {mkey}.")
                if not np.array_equal(pids, chex_meta["patient_ids"]):
                    raise ValueError(f"CheXpert patient_ids mismatch in {mkey}.")
                if not np.array_equal(raw_pneu, chex_meta["raw_pneumonia"], equal_nan=True):
                    raise ValueError(f"CheXpert raw_pneumonia mismatch in {mkey}.")

            models_data[mkey]["target_logits"] = cx_logits

    assert chex_meta is not None
    n_cx_eval = len(chex_meta["raw_pneumonia"])
    n_cx_attempted = n_cx_eval + n_excluded
    fail_rate = float(n_excluded / max(n_cx_attempted, 1))
    excluded_patient_ids: List[str] = []
    if fail_rate > 0.001:
        for item in excl.get("exclusions", []):
            if isinstance(item, dict) and "patient_id" in item:
                excluded_patient_ids.append(str(item["patient_id"]))

    return {
        "dry_run": False,
        "mode": "PRODUCTION",
        "run_id": expected_run_id,
        "plan_sha256": plan_sha,
        "manifest_sha256": manifest_sha,
        "reference_bundle_sha256": ref_sha,
        "git_commit": gate_commit,
        "gpu_name": gate_gpu,
        "torch_version": gate_torch,
        "exclusions": {
            "n_excluded": n_excluded,
            "n_attempted": n_cx_attempted,
            "fail_rate": fail_rate,
            "triggers_patient_sensitivity": fail_rate > 0.001,
            "excluded_patient_ids": sorted(set(excluded_patient_ids)),
        },
        "source_val": nih_meta["val"],
        "source_test": nih_meta["test"],
        "target": chex_meta,
        "models": models_data,
        "target_name": "CheXpert_frontal",
        "sensitivity_exclude_rows": list(CHEXPERT_SENSITIVITY_EXCLUDE_ROWS),
    }


def load_nih_dry_run_bundle(
    preds_dir: Path,
    nih_splits_dir: Path,
    plan_path: Path,
    manifest_path: Path,
) -> Dict[str, Any]:
    """Loads stored NIH val and test predictions for an NIH-only DRY_RUN pseudo-contrast.

    Uses NIH_test as Cohort A and NIH_val as a stand-in for Cohort B ('pseudo-contrast')
    without touching or producing any CheXpert results. Still verifies real_inference and
    authorized checkpoint SHA-256 lineage against configs/checkpoint_manifest.json.
    """
    pdir = Path(preds_dir)
    sdir = Path(nih_splits_dir)
    plan_sha = sha256_file(plan_path, lf_normalize=True)
    manifest_sha, sha_map = load_manifest(manifest_path)

    val_df = pd.read_csv(sdir / "nih_val.csv")
    test_df = pd.read_csv(sdir / "nih_test.csv")

    models_data: Dict[str, Dict[str, Any]] = {}
    splits_meta: Dict[str, Dict[str, Any]] = {}
    run_ids = set()
    git_commits = set()

    for arch in ARCHITECTURES:
        for seed in SEEDS:
            mkey = f"{arch}_{seed}"
            models_data[mkey] = {"arch": arch, "seed": seed}

            for split, ref_df in (("val", val_df), ("test", test_df)):
                p_nih = pdir / f"predictions_{split}_{arch}_{seed}.npz"
                if not p_nih.is_file():
                    raise FileNotFoundError(f"Missing stored NIH {split} prediction: {p_nih}")
                with np.load(p_nih, allow_pickle=False) as zf:
                    meta = _validate_npz_common(
                        zf,
                        p_nih,
                        arch=arch,
                        seed=seed,
                        sha_map=sha_map,
                    )
                    run_ids.add(meta["run_id"])
                    if "git_commit" in zf.files:
                        git_commits.add(_scalar_str(zf["git_commit"]).lower())

                    logits = _extract_pneumonia_logits(zf, p_nih)
                    y_arr = np.asarray(zf["y_true"], dtype=np.float64)
                    y_true = y_arr[:, PNEU_COL_IDX] if y_arr.ndim == 2 else y_arr
                    pids = np.array([_scalar_str(x) for x in zf["patient_ids"]], dtype=object)

                if not np.array_equal(pids, ref_df["patient_id"].astype(str).to_numpy()):
                    raise ValueError(f"DRY_RUN row alignment error vs nih_{split}.csv in {p_nih.name}")
                if not np.array_equal(y_true, ref_df["Pneumonia"].to_numpy(dtype=np.float64)):
                    raise ValueError(f"DRY_RUN Pneumonia label mismatch vs nih_{split}.csv in {p_nih.name}")

                if split not in splits_meta:
                    splits_meta[split] = {
                        "y_true": y_true,
                        "raw_pneumonia": y_true.copy(),
                        "patient_ids": pids,
                        "view_position": ref_df["view_position"].astype(str).to_numpy(dtype=object),
                        "sex": ref_df["sex"].astype(str).to_numpy(dtype=object),
                        "age": pd.to_numeric(ref_df["age"], errors="coerce").to_numpy(dtype=np.float64),
                    }
                models_data[mkey][f"nih_{split}_logits"] = logits
                if split == "val":
                    # Stand-in target cohort = NIH validation (pseudo-contrast ONLY)
                    models_data[mkey]["target_logits"] = logits.copy()

    if len(run_ids) != 1:
        raise ValueError(f"Inconsistent run_id across stored NIH files: {run_ids}")

    return {
        "dry_run": True,
        "mode": "DRY_RUN",
        "run_id": f"DRY_RUN_{next(iter(run_ids))}",
        "plan_sha256": plan_sha,
        "manifest_sha256": manifest_sha,
        "reference_bundle_sha256": "DRY_RUN_NOT_USED",
        "git_commit": next(iter(git_commits)) if len(git_commits) == 1 else "mixed",
        "gpu_name": "DRY_RUN_STORED_NIH",
        "torch_version": "DRY_RUN_STORED_NIH",
        "exclusions": {
            "n_excluded": 0,
            "n_attempted": len(val_df),
            "fail_rate": 0.0,
            "triggers_patient_sensitivity": False,
            "excluded_patient_ids": [],
        },
        "source_val": splits_meta["val"],
        "source_test": splits_meta["test"],
        "target": splits_meta["val"],
        "models": models_data,
        "target_name": "DRY_RUN_NIH_VAL_STANDIN",
        "sensitivity_exclude_rows": [],
    }
