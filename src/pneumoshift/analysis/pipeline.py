"""End-to-end orchestrator implementing ANALYSIS_PLAN_TRANSFER.md v2.2.3 & DEVIATIONS.md PRE-E-001."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import pandas as pd

from pneumoshift.analysis.bootstrap import (
    ClusterBootstrapPlan,
    benjamini_hochberg,
    compute_paired_architecture_bootstrap,
)
from pneumoshift.analysis.metrics import (
    apply_label_policy,
    compute_auprc,
    compute_calibration_suite,
    compute_operating_metrics,
    delong_two_sample,
    extract_subgroup_masks,
    fast_auroc,
    fit_platt_scaling,
    fit_temperature_scaling,
    select_thresholds,
    target_10pct_platt,
)
from pneumoshift.analysis.provenance import (
    ARCHITECTURES,
    SEEDS,
    sha256_file,
)


def format_dry_run_note(is_dry_run: bool) -> str:
    """Returns explicit provenance tag string marking DRY_RUN vs PRODUCTION output."""
    return "DRY_RUN (NIH-only stand-in pseudo-contrast; NOT CheXpert results)" if is_dry_run else "PRODUCTION"


def _compute_cohort_counts(y_true: np.ndarray, patient_ids: np.ndarray) -> Dict[str, Any]:
    """Computes image-level and patient-level counts and prevalences."""
    y = np.asarray(y_true, dtype=np.float64)
    pids = np.asarray(patient_ids, dtype=object)
    n_img = int(len(y))
    n_pos_img = int((y == 1.0).sum())
    n_pat = int(len(np.unique(pids))) if n_img > 0 else 0
    n_pos_pat = int(len(np.unique(pids[y == 1.0]))) if n_pos_img > 0 else 0
    return {
        "n_images": n_img,
        "n_pos_images": n_pos_img,
        "image_prevalence": float(n_pos_img / n_img) if n_img > 0 else float("nan"),
        "n_patients": n_pat,
        "n_pos_patients": n_pos_pat,
        "patient_prevalence": float(n_pos_pat / n_pat) if n_pat > 0 else float("nan"),
    }


def _stamp_df(df: pd.DataFrame, stamps: Dict[str, Any]) -> pd.DataFrame:
    """Prepends immutable provenance stamps to every output CSV table."""
    out = df.copy()
    for col, val in reversed(list(stamps.items())):
        out.insert(0, col, val)
    return out


def _clean_json(obj: Any) -> Any:
    """Recursively converts numpy scalars/arrays and nan/inf for strict JSON serialization."""
    if isinstance(obj, dict):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if not np.isfinite(v) else v
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_transfer_analysis(
    bundle: Dict[str, Any],
    out_dir: Path,
    script_path: Path,
    n_bootstrap: int = 2000,
    bootstrap_seed: int = 42,
    n_target_platt_splits: int = 20,
    target_platt_seed: int = 100,
) -> Dict[str, Any]:
    """Executes all pre-specified primary, secondary, recalibration, and subgroup analyses."""
    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    is_dry_run = bool(bundle.get("dry_run", False))
    mode_tag = "DRY_RUN" if is_dry_run else "PRODUCTION"
    script_sha = sha256_file(script_path, lf_normalize=True)
    analysis_pkg_shas = {
        p.name: sha256_file(p, lf_normalize=True)
        for p in sorted(Path(__file__).parent.glob("*.py"))
    }
    plan_sha = str(bundle["plan_sha256"])

    common_stamps = {
        "mode": mode_tag,
        "dry_run": is_dry_run,
        "run_id": str(bundle["run_id"]),
        "analysis_plan_sha256": plan_sha,
        "analysis_script_sha256": script_sha,
    }

    val_meta = bundle["source_val"]
    test_meta = bundle["source_test"]
    tgt_meta = bundle["target"]
    models_data = bundle["models"]

    y_val = np.asarray(val_meta["y_true"], dtype=np.float64)
    y_test = np.asarray(test_meta["y_true"], dtype=np.float64)
    pids_test = np.asarray(test_meta["patient_ids"], dtype=object)

    raw_tgt = np.asarray(tgt_meta["raw_pneumonia"], dtype=np.float64)
    pids_tgt = np.asarray(tgt_meta["patient_ids"], dtype=object)

    # Primary policy (u_zeros) on target
    u_zeros_tgt = apply_label_policy(raw_tgt, "u_zeros", pids_tgt)
    y_tgt_primary = u_zeros_tgt["y_true"]

    cohort_summary = {
        "source_val": _compute_cohort_counts(y_val, np.asarray(val_meta["patient_ids"], dtype=object)),
        "source_test": _compute_cohort_counts(y_test, pids_test),
        "target_primary_u_zeros": {
            k: u_zeros_tgt[k]
            for k in ("n_images", "n_pos_images", "image_prevalence", "n_patients", "n_pos_patients", "patient_prevalence")
        },
    }

    # 1. Per-seed thresholds & source-fitted recalibration on fresh NIH validation (§7.1-7.2, §9.2-9.3)
    per_seed_thresholds: Dict[str, Dict[str, float]] = {}
    source_scalers: Dict[str, Dict[str, float]] = {}
    thresh_rows: List[Dict[str, Any]] = []

    for arch in ARCHITECTURES:
        for seed in SEEDS:
            mkey = f"{arch}_{seed}"
            z_val = models_data[mkey]["nih_val_logits"]
            th = select_thresholds(y_val, z_val)
            temp_src = fit_temperature_scaling(z_val, y_val)
            platt_a_src, platt_b_src = fit_platt_scaling(z_val, y_val)

            per_seed_thresholds[mkey] = th
            source_scalers[mkey] = {
                "temp_T": float(temp_src),
                "platt_a": float(platt_a_src),
                "platt_b": float(platt_b_src),
            }

            row = {"architecture": arch, "seed": seed, **th, **source_scalers[mkey]}
            for tname, tau in th.items():
                op = compute_operating_metrics(y_val, z_val, tau)
                row[f"val_{tname}_sens"] = op["sensitivity"]
                row[f"val_{tname}_spec"] = op["specificity"]
            thresh_rows.append(row)

    # 2. Primary Paired Patient-Clustered Bootstrap (B=2,000, seed=42, §4 & §5)
    boot_plan = ClusterBootstrapPlan(
        patient_ids_a=pids_test,
        patient_ids_b=pids_tgt,
        n_resamples=n_bootstrap,
        seed=bootstrap_seed,
    )

    arch_boot: Dict[str, Dict[str, Any]] = {}
    for arch in ARCHITECTURES:
        seed_inputs = {
            seed: {
                "logits_a": models_data[f"{arch}_{seed}"]["nih_test_logits"],
                "logits_b": models_data[f"{arch}_{seed}"]["target_logits"],
                "thresholds": per_seed_thresholds[f"{arch}_{seed}"],
            }
            for seed in SEEDS
        }
        arch_boot[arch] = compute_paired_architecture_bootstrap(
            plan=boot_plan,
            y_a=y_test,
            y_b=y_tgt_primary,
            seed_inputs=seed_inputs,
        )

    # Extract exact 6-test confirmatory BH family (§4.1 & §4.2):
    # 3 primary endpoints x 2 architectures
    primary_specs = [
        ("Endpoint_1_Delta_AUROC", "auroc", "Delta AUROC (NIH_test - Target)"),
        ("Endpoint_2_Delta_Slope", "slope", "Delta Calibration Slope (NIH_test - Target)"),
        ("Endpoint_3_Delta_Sens_tau90", "sens_tau_sens90", "Delta Sensitivity at tau_sens90 (NIH_test - Target)"),
    ]
    raw_p_family: List[float] = []
    primary_entries: List[Dict[str, Any]] = []
    for arch in ARCHITECTURES:
        joint = arch_boot[arch]["joint"]
        for ep_id, metric_key, label in primary_specs:
            ca = joint["cohort_a"][metric_key]
            cb = joint["cohort_b"][metric_key]
            cd = joint["contrast"][metric_key]
            raw_p_family.append(cd["p_value"])
            primary_entries.append({
                "endpoint_id": ep_id,
                "endpoint_label": label,
                "architecture": arch,
                "metric_key": metric_key,
                "nih_test_mean": ca["estimate"],
                "nih_test_ci_lower": ca["ci_lower"],
                "nih_test_ci_upper": ca["ci_upper"],
                "nih_test_between_seed_sd": ca["between_seed_sd_ddof1"],
                "target_mean": cb["estimate"],
                "target_ci_lower": cb["ci_lower"],
                "target_ci_upper": cb["ci_upper"],
                "target_between_seed_sd": cb["between_seed_sd_ddof1"],
                "delta_mean": cd["estimate"],
                "delta_ci_lower": cd["ci_lower"],
                "delta_ci_upper": cd["ci_upper"],
                "delta_between_seed_sd": cd["between_seed_sd_ddof1"],
                "p_value_two_sided": cd["p_value"],
            })

    bh_q, bh_sig = benjamini_hochberg(raw_p_family, alpha=0.05)
    for entry, qv, sig in zip(primary_entries, bh_q, bh_sig):
        entry["bh_q_value"] = float(qv)
        entry["bh_significant_0_05"] = bool(sig)

    # Per-seed contrast table
    per_seed_rows: List[Dict[str, Any]] = []
    contrast_metrics = (
        "auroc",
        "slope",
        "intercept",
        "sens_tau_sens90",
        "spec_tau_sens90",
        "ppv_tau_sens90",
        "npv_tau_sens90",
        "sens_tau_youden",
        "spec_tau_youden",
        "ppv_tau_youden",
        "npv_tau_youden",
        "sens_tau_sens95",
        "spec_tau_sens95",
        "ppv_tau_sens95",
        "npv_tau_sens95",
    )
    for arch in ARCHITECTURES:
        for seed in SEEDS:
            ps = arch_boot[arch]["per_seed"][seed]
            for mk in contrast_metrics:
                ca = ps["cohort_a"][mk]
                cb = ps["cohort_b"][mk]
                cd = ps["contrast"][mk]
                per_seed_rows.append({
                    "architecture": arch,
                    "seed": seed,
                    "metric": mk,
                    "nih_test": ca["estimate"],
                    "nih_test_ci_lo": ca["ci_lower"],
                    "nih_test_ci_hi": ca["ci_upper"],
                    "target": cb["estimate"],
                    "target_ci_lo": cb["ci_lower"],
                    "target_ci_hi": cb["ci_upper"],
                    "delta": cd["estimate"],
                    "delta_ci_lo": cd["ci_lower"],
                    "delta_ci_hi": cd["ci_upper"],
                    "p_value": cd["p_value"],
                })

    # Joint secondary contrasts table
    secondary_joint_rows: List[Dict[str, Any]] = []
    for arch in ARCHITECTURES:
        joint = arch_boot[arch]["joint"]
        for mk in contrast_metrics:
            ca = joint["cohort_a"][mk]
            cb = joint["cohort_b"][mk]
            cd = joint["contrast"][mk]
            secondary_joint_rows.append({
                "architecture": arch,
                "metric": mk,
                "nih_test_mean": ca["estimate"],
                "nih_test_ci_lo": ca["ci_lower"],
                "nih_test_ci_hi": ca["ci_upper"],
                "nih_test_seed_sd": ca["between_seed_sd_ddof1"],
                "target_mean": cb["estimate"],
                "target_ci_lo": cb["ci_lower"],
                "target_ci_hi": cb["ci_upper"],
                "target_seed_sd": cb["between_seed_sd_ddof1"],
                "delta_mean": cd["estimate"],
                "delta_ci_lo": cd["ci_lower"],
                "delta_ci_hi": cd["ci_upper"],
                "delta_seed_sd": cd["between_seed_sd_ddof1"],
                "p_value_unadjusted": cd["p_value"],
            })

    # 3. DeLong sensitivity + AUPRC + full Calibration & Recalibration (§5.6, §7, §8)
    delong_rows: List[Dict[str, Any]] = []
    recal_rows: List[Dict[str, Any]] = []
    sec_cal_rows: List[Dict[str, Any]] = []

    for arch in ARCHITECTURES:
        for seed in SEEDS:
            mkey = f"{arch}_{seed}"
            z_test = models_data[mkey]["nih_test_logits"]
            z_tgt = models_data[mkey]["target_logits"]
            sc = source_scalers[mkey]

            dl = delong_two_sample(z_test, y_test, z_tgt, y_tgt_primary)
            delong_rows.append({"architecture": arch, "seed": seed, **dl})

            auprc_test = compute_auprc(z_test, y_test)
            auprc_tgt = compute_auprc(z_tgt, y_tgt_primary)

            # Calibration suites across regimes (§7.1-7.2, §8.2)
            regimes = [
                ("nih_test_uncalibrated", z_test, y_test),
                ("target_uncalibrated", z_tgt, y_tgt_primary),
                ("nih_test_source_temp", z_test / sc["temp_T"], y_test),
                ("target_source_temp", z_tgt / sc["temp_T"], y_tgt_primary),
                ("nih_test_source_platt", sc["platt_a"] * z_test + sc["platt_b"], y_test),
                ("target_source_platt", sc["platt_a"] * z_tgt + sc["platt_b"], y_tgt_primary),
            ]
            for reg_name, reg_z, reg_y in regimes:
                cal = compute_calibration_suite(reg_z, reg_y, n_bins=15)
                row = {
                    "architecture": arch,
                    "seed": seed,
                    "regime": reg_name,
                    "auroc": fast_auroc(reg_z, reg_y),
                    "auprc": compute_auprc(reg_z, reg_y),
                    **cal,
                }
                sec_cal_rows.append(row)

            # Target-10% patient-level Platt recalibration (20 splits, §7.3)
            t10 = target_10pct_platt(
                z_tgt,
                y_tgt_primary,
                pids_tgt,
                n_splits=n_target_platt_splits,
                base_seed=target_platt_seed,
                n_bins=15,
            )
            recal_rows.append({
                "architecture": arch,
                "seed": seed,
                "auprc_nih_test": auprc_test,
                "auprc_target": auprc_tgt,
                "source_temp_T": sc["temp_T"],
                "source_platt_a": sc["platt_a"],
                "source_platt_b": sc["platt_b"],
                **{k: v for k, v in t10.items() if k != "splits"},
            })

    # 4. All Four Label Policies (§3.1-3.4)
    # In DRY_RUN on NIH val (0/1 binary only), all 4 policies still run seamlessly on raw_tgt
    policy_names = ("u_zeros", "u_ones", "u_excluded", "explicitly_labeled_only")
    policy_rows: List[Dict[str, Any]] = []
    for pol in policy_names:
        pol_info = apply_label_policy(raw_tgt, pol, pids_tgt)
        mask = pol_info["mask"]
        y_pol = pol_info["y_true"]
        if len(y_pol) == 0:
            continue
        for arch in ARCHITECTURES:
            for seed in SEEDS:
                mkey = f"{arch}_{seed}"
                z_pol = models_data[mkey]["target_logits"][mask]
                th = per_seed_thresholds[mkey]
                cal = compute_calibration_suite(z_pol, y_pol, n_bins=15)
                op90 = compute_operating_metrics(y_pol, z_pol, th["tau_sens90"])
                opy = compute_operating_metrics(y_pol, z_pol, th["tau_youden"])
                policy_rows.append({
                    "policy": pol,
                    "architecture": arch,
                    "seed": seed,
                    "n_images": pol_info["n_images"],
                    "n_pos_images": pol_info["n_pos_images"],
                    "image_prevalence": pol_info["image_prevalence"],
                    "n_patients": pol_info["n_patients"],
                    "n_pos_patients": pol_info["n_pos_patients"],
                    "patient_prevalence": pol_info["patient_prevalence"],
                    "auroc": fast_auroc(z_pol, y_pol),
                    "auprc": compute_auprc(z_pol, y_pol),
                    "sens_tau_sens90": op90["sensitivity"],
                    "spec_tau_sens90": op90["specificity"],
                    "ppv_tau_sens90": op90["ppv"],
                    "npv_tau_sens90": op90["npv"],
                    "sens_tau_youden": opy["sensitivity"],
                    "spec_tau_youden": opy["specificity"],
                    **cal,
                })

    # 5. Exploratory Subgroups (§11) under u_zeros
    subgroups = extract_subgroup_masks(
        view_position=tgt_meta.get("view_position"),
        sex=tgt_meta.get("sex"),
        age=tgt_meta.get("age"),
        n_rows=len(y_tgt_primary),
    )
    subgroup_rows: List[Dict[str, Any]] = []
    for axis, level_map in subgroups.items():
        for level_name, smask in level_map.items():
            cnt = int(smask.sum())
            if cnt == 0:
                continue
            y_sub = y_tgt_primary[smask]
            pids_sub = pids_tgt[smask]
            counts = _compute_cohort_counts(y_sub, pids_sub)
            for arch in ARCHITECTURES:
                for seed in SEEDS:
                    mkey = f"{arch}_{seed}"
                    z_sub = models_data[mkey]["target_logits"][smask]
                    th = per_seed_thresholds[mkey]
                    cal = compute_calibration_suite(z_sub, y_sub, n_bins=15)
                    op90 = compute_operating_metrics(y_sub, z_sub, th["tau_sens90"])
                    subgroup_rows.append({
                        "status": "EXPLORATORY",
                        "subgroup_axis": axis,
                        "subgroup_level": level_name,
                        "architecture": arch,
                        "seed": seed,
                        **counts,
                        "auroc": fast_auroc(z_sub, y_sub),
                        "auprc": compute_auprc(z_sub, y_sub),
                        "sens_tau_sens90": op90["sensitivity"],
                        "spec_tau_sens90": op90["specificity"],
                        **cal,
                    })

    # 6. Pre-specified Outlier Sensitivity (DEVIATIONS.md PRE-E-001) & §2.4 Exclusion Guard
    outlier_sensitivity_rows: List[Dict[str, Any]] = []
    excl_rows = sorted(
        r for r in bundle.get("sensitivity_exclude_rows", [])
        if 0 <= int(r) < len(y_tgt_primary)
    )
    if excl_rows:
        keep_mask = np.ones(len(y_tgt_primary), dtype=bool)
        keep_mask[excl_rows] = False
        y_clean = y_tgt_primary[keep_mask]
        pids_clean = pids_tgt[keep_mask]

        clean_boot_plan = ClusterBootstrapPlan(
            patient_ids_a=pids_test,
            patient_ids_b=pids_clean,
            n_resamples=n_bootstrap,
            seed=bootstrap_seed,
        )
        eff_seed_inputs = {
            seed: {
                "logits_a": models_data[f"efficientnet_b0_{seed}"]["nih_test_logits"],
                "logits_b": models_data[f"efficientnet_b0_{seed}"]["target_logits"][keep_mask],
                "thresholds": per_seed_thresholds[f"efficientnet_b0_{seed}"],
            }
            for seed in SEEDS
        }
        eff_clean_boot = compute_paired_architecture_bootstrap(
            plan=clean_boot_plan,
            y_a=y_test,
            y_b=y_clean,
            seed_inputs=eff_seed_inputs,
        )

        # Side-by-side comparison on efficientnet_b0_2024 (and 3-seed efficientnet_b0)
        z_2024_all = models_data["efficientnet_b0_2024"]["target_logits"]
        z_2024_clean = z_2024_all[keep_mask]
        sc_2024 = source_scalers["efficientnet_b0_2024"]

        for cohort_tag, z_vec, y_vec in (
            (f"all_rows_N={len(y_tgt_primary)}", z_2024_all, y_tgt_primary),
            (f"excl_corrupted_rows_{excl_rows}_N={len(y_clean)}", z_2024_clean, y_clean),
        ):
            cal_uncal = compute_calibration_suite(z_vec, y_vec, n_bins=15)
            cal_temp = compute_calibration_suite(z_vec / sc_2024["temp_T"], y_vec, n_bins=15)
            cal_platt = compute_calibration_suite(sc_2024["platt_a"] * z_vec + sc_2024["platt_b"], y_vec, n_bins=15)
            outlier_sensitivity_rows.append({
                "sensitivity_id": "PRE-E-001",
                "model": "efficientnet_b0_2024",
                "cohort": cohort_tag,
                "n_images": int(len(y_vec)),
                "n_pos_images": int((y_vec == 1.0).sum()),
                "auroc": fast_auroc(z_vec, y_vec),
                "auprc": compute_auprc(z_vec, y_vec),
                "uncal_slope": cal_uncal["slope"],
                "uncal_intercept": cal_uncal["intercept"],
                "uncal_nll": cal_uncal["nll"],
                "uncal_brier": cal_uncal["brier"],
                "uncal_ece_uniform": cal_uncal["ece_uniform"],
                "uncal_ece_adaptive": cal_uncal["ece_adaptive"],
                "source_temp_nll": cal_temp["nll"],
                "source_temp_ece_uniform": cal_temp["ece_uniform"],
                "source_platt_nll": cal_platt["nll"],
                "source_platt_ece_uniform": cal_platt["ece_uniform"],
                "eff_b0_joint_delta_auroc": (
                    arch_boot["efficientnet_b0"]["joint"]["contrast"]["auroc"]["estimate"]
                    if "all_rows" in cohort_tag
                    else eff_clean_boot["joint"]["contrast"]["auroc"]["estimate"]
                ),
                "eff_b0_joint_delta_slope": (
                    arch_boot["efficientnet_b0"]["joint"]["contrast"]["slope"]["estimate"]
                    if "all_rows" in cohort_tag
                    else eff_clean_boot["joint"]["contrast"]["slope"]["estimate"]
                ),
            })

    # Optional §2.4 >0.1% image-load-failure patient exclusion check
    patient_failure_sensitivity: Optional[Dict[str, Any]] = None
    excl_meta = bundle.get("exclusions", {})
    if excl_meta.get("triggers_patient_sensitivity") and excl_meta.get("excluded_patient_ids"):
        bad_pats = set(excl_meta["excluded_patient_ids"])
        keep_pat_mask = np.array([str(p) not in bad_pats for p in pids_tgt], dtype=bool)
        pat_boot_plan = ClusterBootstrapPlan(
            patient_ids_a=pids_test,
            patient_ids_b=pids_tgt[keep_pat_mask],
            n_resamples=n_bootstrap,
            seed=bootstrap_seed,
        )
        pat_sens_boot = {}
        for arch in ARCHITECTURES:
            pat_sens_boot[arch] = compute_paired_architecture_bootstrap(
                plan=pat_boot_plan,
                y_a=y_test,
                y_b=y_tgt_primary[keep_pat_mask],
                seed_inputs={
                    s: {
                        "logits_a": models_data[f"{arch}_{s}"]["nih_test_logits"],
                        "logits_b": models_data[f"{arch}_{s}"]["target_logits"][keep_pat_mask],
                        "thresholds": per_seed_thresholds[f"{arch}_{s}"],
                    }
                    for s in SEEDS
                },
            )
        patient_failure_sensitivity = {
            "rule": "Section 2.4 (>0.1% image-load failures -> remove affected patients)",
            "n_excluded_images": excl_meta["n_excluded"],
            "n_patients_removed": len(bad_pats),
            "n_remaining_images": int(keep_pat_mask.sum()),
            "results_by_arch": pat_sens_boot,
        }

    # Assemble stamped output tables + JSON
    tables = {
        "primary_endpoints.csv": pd.DataFrame(primary_entries),
        "per_seed_endpoints.csv": pd.DataFrame(per_seed_rows),
        "secondary_joint_contrasts.csv": pd.DataFrame(secondary_joint_rows),
        "secondary_calibration.csv": pd.DataFrame(sec_cal_rows),
        "recalibration_summary.csv": pd.DataFrame(recal_rows),
        "label_policies.csv": pd.DataFrame(policy_rows),
        "exploratory_subgroups.csv": pd.DataFrame(subgroup_rows),
        "delong_sensitivity.csv": pd.DataFrame(delong_rows),
        "thresholds_per_seed.csv": pd.DataFrame(thresh_rows),
    }
    if outlier_sensitivity_rows:
        tables["outlier_sensitivity_pre_e_001.csv"] = pd.DataFrame(outlier_sensitivity_rows)

    written_files: Dict[str, str] = {}
    for fname, df in tables.items():
        fpath = out_p / fname
        _stamp_df(df, common_stamps).to_csv(fpath, index=False)
        written_files[fname] = str(fpath)

    results_payload = {
        "provenance": {
            "mode": mode_tag,
            "dry_run": is_dry_run,
            "note": format_dry_run_note(is_dry_run),
            "run_id": bundle["run_id"],
            "target_name": bundle["target_name"],
            "analysis_plan_sha256": plan_sha,
            "manifest_sha256": bundle["manifest_sha256"],
            "reference_bundle_sha256": bundle["reference_bundle_sha256"],
            "git_commit": bundle["git_commit"],
            "gpu_name": bundle["gpu_name"],
            "torch_version": bundle["torch_version"],
            "analysis_script_path": str(script_path),
            "analysis_script_sha256": script_sha,
            "analysis_module_sha256s": analysis_pkg_shas,
            "bootstrap_iterations": n_bootstrap,
            "bootstrap_seed": bootstrap_seed,
            "target_10pct_platt_splits": n_target_platt_splits,
            "calibration_bins": 15,
            "exclusions": excl_meta,
        },
        "cohort_summary": cohort_summary,
        "thresholds_per_seed": per_seed_thresholds,
        "source_scalers": source_scalers,
        "primary_endpoints_6test_bh_family": primary_entries,
        "bootstrap_by_architecture": arch_boot,
        "secondary_calibration": sec_cal_rows,
        "recalibration_target_10pct_platt": recal_rows,
        "label_policies": policy_rows,
        "exploratory_subgroups": subgroup_rows,
        "delong_sensitivity": delong_rows,
        "outlier_sensitivity_pre_e_001": outlier_sensitivity_rows,
        "patient_failure_sensitivity": patient_failure_sensitivity,
        "output_csv_files": written_files,
    }

    results_json_path = out_p / ("DRY_RUN_results.json" if is_dry_run else "results.json")
    results_json_path.write_text(json.dumps(_clean_json(results_payload), indent=2), encoding="utf-8")
    results_payload["results_json_path"] = str(results_json_path)
    return results_payload
