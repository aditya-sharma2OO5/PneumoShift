"""Cross-dataset evaluation harness, configuration validation, and path resolution."""

from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES, get_evaluation_targets
from pneumoshift.data.splits import create_synthetic_dataset
from pneumoshift.evaluation.bootstrap import (
    benjamini_hochberg_fdr,
    paired_bootstrap_auroc_diff,
    two_sample_bootstrap_diff,
)
from pneumoshift.evaluation.metrics import compute_discrimination_metrics
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.cross_domain")

VALID_SHIFT_DIRECTIONS: List[str] = [
    "nih_to_chexpert",
    "chexpert_to_nih",
    "synthetic_to_synthetic",
]

VALID_UNCERTAINTY_MODES: List[str] = [
    "uncertain_as_negative",
    "uncertain_excluded",
]

VALID_THRESHOLD_METRICS: List[str] = [
    "youden",
    "f1",
    "sens90",
]


def validate_cross_domain_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates schema, parameters, and constraints for cross-dataset evaluation.

    Args:
        cfg: Raw configuration dictionary loaded from YAML.

    Returns:
        Validated and normalized configuration dictionary.

    Raises:
        ValueError: If mandatory sections or invalid parameter values are encountered.
    """
    # 1. Mandatory top-level sections
    for sec in ["source_model", "target_data", "eval_protocol", "outputs"]:
        if sec not in cfg:
            raise ValueError(f"Missing mandatory configuration section: '{sec}'")

    shift_dir = cfg.get("shift_direction", "")
    if shift_dir not in VALID_SHIFT_DIRECTIONS:
        raise ValueError(
            f"Invalid shift_direction: '{shift_dir}'. Expected one of {VALID_SHIFT_DIRECTIONS}"
        )

    # 2. Source model validation
    src_model = cfg["source_model"]
    if "architecture" not in src_model:
        raise ValueError("source_model missing mandatory 'architecture'")
    if "checkpoint_path" not in src_model:
        raise ValueError("source_model missing mandatory 'checkpoint_path'")

    # 3. Target data validation
    tgt_data = cfg["target_data"]
    if "dataset_name" not in tgt_data:
        raise ValueError("target_data missing mandatory 'dataset_name'")
    if "splits_dir" not in tgt_data:
        raise ValueError("target_data missing mandatory 'splits_dir'")

    uncertainty_modes = tgt_data.get("eval_uncertainty_modes", ["uncertain_as_negative"])
    for mode in uncertainty_modes:
        if mode not in VALID_UNCERTAINTY_MODES:
            raise ValueError(
                f"Invalid uncertainty mode '{mode}'. Expected one of {VALID_UNCERTAINTY_MODES}"
            )

    # 4. Evaluation protocol validation
    protocol = cfg["eval_protocol"]
    cal_ratio = float(protocol.get("target_calibration_ratio", 0.10))
    if not (0.0 < cal_ratio < 1.0):
        raise ValueError(
            f"target_calibration_ratio must be in (0, 1), got {cal_ratio}"
        )

    conf_level = float(protocol.get("confidence_level", 0.95))
    if not (0.5 < conf_level < 1.0):
        raise ValueError(f"confidence_level must be in (0.5, 1.0), got {conf_level}")

    boot_samples = int(protocol.get("bootstrap_samples", 1000))
    if boot_samples < 10:
        raise ValueError(f"bootstrap_samples must be >= 10, got {boot_samples}")

    thresh_metrics = protocol.get("threshold_metrics", ["youden", "f1", "sens90"])
    for tm in thresh_metrics:
        if tm not in VALID_THRESHOLD_METRICS:
            raise ValueError(
                f"Invalid threshold metric '{tm}'. Expected one of {VALID_THRESHOLD_METRICS}"
            )

    # 5. Output locations
    out_dir = cfg["outputs"].get("output_dir")
    if not out_dir:
        raise ValueError("outputs section missing mandatory 'output_dir'")

    logger.debug("Cross-domain configuration validated successfully for shift: %s", shift_dir)
    return cfg


def resolve_cross_domain_paths(
    cfg: Dict[str, Any],
    smoke_test: bool = False,
) -> Dict[str, Path]:
    """Resolves all filesystem paths for source model, target splits, and output directories.

    In dry-run or smoke-test mode, automatically falls back to synthetic datasets and mock weights.

    Args:
        cfg: Validated cross-domain configuration dictionary.
        smoke_test: Whether to force synthetic smoke test paths.

    Returns:
        Dict mapping logical keys to resolved Paths.
    """
    src_model = cfg["source_model"]
    tgt_data = cfg["target_data"]
    tgt_ds = tgt_data["dataset_name"]
    src_ds = src_model.get("source_dataset", "nih")

    ckpt_path = Path(src_model["checkpoint_path"])
    src_val_csv = Path(src_model.get("source_val_csv", f"data/splits/{src_ds}/{src_ds}_val.csv"))
    src_metrics_path = Path(src_model.get("source_test_metrics", f"results/runs/{src_ds}_densenet121/test_metrics_densenet121_42.json"))

    splits_dir = Path(tgt_data["splits_dir"])
    test_split_name = tgt_data.get("test_split", "test")
    cal_split_name = tgt_data.get("target_cal_split", "target_cal")
    off_val_split_name = tgt_data.get("official_val_split", "official_val")

    target_test_csv = splits_dir / f"{tgt_ds}_{test_split_name}.csv"
    target_cal_csv = splits_dir / f"{tgt_ds}_{cal_split_name}.csv"
    target_off_val_csv = splits_dir / f"{tgt_ds}_{off_val_split_name}.csv"
    images_dir = Path(tgt_data.get("images_dir", ""))
    src_images_dir = Path(
        src_model.get("source_images_dir")
        or src_model.get("images_dir")
        or ("/tmp/nih_images" if Path("/tmp/nih_images").is_dir() else f"data/{src_ds}/images")
    )

    out_dir = Path(cfg["outputs"]["output_dir"])
    figures_dir = Path(cfg["outputs"].get("figures_dir", out_dir / "figures"))

    # Fallback to synthetic cohort if smoke_test requested or real files missing
    use_synthetic = smoke_test or (not target_test_csv.is_file() and not ckpt_path.is_file())
    if use_synthetic:
        logger.info("Using synthetic smoke test environment for cross-domain evaluation.")
        synth_dir = Path("data/synthetic")
        if not (synth_dir / "splits" / "synthetic_test.csv").is_file():
            create_synthetic_dataset(output_dir=synth_dir, num_samples=40, image_size=64)

        src_val_csv = synth_dir / "splits" / "synthetic_val.csv"
        target_test_csv = synth_dir / "splits" / "synthetic_test.csv"
        target_cal_csv = synth_dir / "splits" / "synthetic_target_cal.csv"
        target_off_val_csv = synth_dir / "splits" / "synthetic_official_val.csv"
        images_dir = synth_dir / "images"
        src_images_dir = synth_dir / "images"

    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    resolved = {
        "checkpoint_path": ckpt_path,
        "source_val_csv": src_val_csv,
        "source_test_metrics": src_metrics_path,
        "target_test_csv": target_test_csv,
        "target_cal_csv": target_cal_csv,
        "target_off_val_csv": target_off_val_csv,
        "images_dir": images_dir,
        "source_images_dir": src_images_dir,
        "output_dir": out_dir,
        "figures_dir": figures_dir,
        "is_synthetic": use_synthetic,
    }
    return resolved


def extract_cross_domain_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str = "Inference",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward inference across loader to extract logits, calibrated probs, targets, and features.

    Returns:
        (logits, probabilities, targets, penultimate_features)
    """
    model.eval()
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_features: List[np.ndarray] = []

    total_batches = len(loader)
    total_samples = len(loader.dataset) if hasattr(loader, "dataset") else total_batches * (loader.batch_size or 1)
    batch_size = loader.batch_size or 1

    log_interval = max(1, 25 if total_batches >= 80 else 10)
    start_time = time.time()
    logger.info("[%s] Beginning inference across %d samples (%d batches, batch_size=%d)...", desc, total_samples, total_batches, batch_size)

    with torch.no_grad():
        for step, batch in enumerate(loader):
            imgs = batch["image"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)
            logits, feats = model(imgs, return_features=True)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_features.append(feats.cpu().numpy())

            if (step + 1) % log_interval == 0 or (step + 1) == total_batches:
                elapsed = time.time() - start_time
                done_batches = step + 1
                pct = 100.0 * done_batches / total_batches
                done_samples = min(done_batches * batch_size, total_samples)
                rate = done_samples / max(0.001, elapsed)
                remaining_sec = (total_samples - done_samples) / max(0.1, rate)
                logger.info(
                    "[%s] Batch [%3d/%3d] (%5.1f%%) | %d/%d images | %.1f img/s | ETA: %.0fs",
                    desc,
                    done_batches,
                    total_batches,
                    pct,
                    done_samples,
                    total_samples,
                    rate,
                    remaining_sec,
                )

    logits = np.vstack(all_logits)
    targets = np.vstack(all_targets)
    features = np.vstack(all_features)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    total_time = time.time() - start_time
    logger.info("[%s] Completed inference in %.1fs (%.1f img/s).", desc, total_time, total_samples / max(0.001, total_time))
    return logits, probs, targets, features


def _extract_metric_value(metrics_dict: Dict[str, Any], finding: str, metric_name: str) -> Optional[float]:
    """Helper to retrieve metric value from varying formats (flat vs nested per_class)."""
    col_key = finding.lower().replace(" ", "_")
    candidates = [
        f"{metric_name}_{col_key}",
        f"val_{col_key}_{metric_name}",
        f"test_{col_key}_{metric_name}",
        f"val_{metric_name}",
        f"test_{metric_name}",
    ]
    for c in candidates:
        if c in metrics_dict and metrics_dict[c] is not None:
            try:
                val = float(metrics_dict[c])
                if not np.isnan(val):
                    return val
            except (ValueError, TypeError):
                pass

    if "per_class" in metrics_dict and finding in metrics_dict["per_class"]:
        f_dict = metrics_dict["per_class"][finding]
        if metric_name in f_dict and f_dict[metric_name] is not None:
            try:
                val = float(f_dict[metric_name])
                if not np.isnan(val):
                    return val
            except (ValueError, TypeError):
                pass

    return None


def compute_generalization_gap_report(
    in_domain_metrics: Dict[str, Any],
    cross_domain_metrics: Dict[str, Any],
    shift_direction: str = "nih_to_chexpert",
    findings: Optional[List[str]] = None,
    source_patients: Optional[np.ndarray] = None,
    source_targets: Optional[np.ndarray] = None,
    source_preds: Optional[np.ndarray] = None,
    target_patients: Optional[np.ndarray] = None,
    target_targets: Optional[np.ndarray] = None,
    target_preds: Optional[np.ndarray] = None,
    target_in_domain_preds: Optional[np.ndarray] = None,
    fdr_alpha: float = 0.05,
    n_resamples: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes discrimination Generalization Gaps and executes statistical hypothesis tests.

    Formula:
        Generalization Gap: Δ = in_domain_metric - cross_domain_metric
        A positive Δ represents performance degradation under domain shift.

    Args:
        in_domain_metrics: Evaluation results dict from in-domain test set.
        cross_domain_metrics: Evaluation results dict from cross-domain target test set.
        shift_direction: 'nih_to_chexpert' or 'chexpert_to_nih'.
        findings: List of target classes (default: SHARED_CLASSES).
        source_patients: Optional array of patient IDs for source cohort.
        source_targets: Optional ground truth targets for source cohort.
        source_preds: Optional model predictions on source cohort.
        target_patients: Optional array of patient IDs for target cohort.
        target_targets: Optional ground truth targets for target cohort.
        target_preds: Optional model predictions on target cohort.
        target_in_domain_preds: Optional in-domain baseline predictions on identical target cases.
        fdr_alpha: Benjamini-Hochberg FDR significance threshold.
        n_resamples: Number of bootstrap iterations.
        seed: Random seed.

    Returns:
        Structured dict with per-class and macro gaps, statistical significance, and markdown table.
    """
    classes = findings or SHARED_CLASSES
    gaps: Dict[str, Dict[str, Any]] = {}
    raw_p_values: List[float] = []
    evaluated_findings: List[str] = []

    macro_in_auc = in_domain_metrics.get("macro_auroc", in_domain_metrics.get("val_macro_auroc"))
    macro_cross_auc = cross_domain_metrics.get("macro_auroc", cross_domain_metrics.get("val_macro_auroc"))
    macro_in_prc = in_domain_metrics.get("macro_auprc", in_domain_metrics.get("val_macro_auprc"))
    macro_cross_prc = cross_domain_metrics.get("macro_auprc", cross_domain_metrics.get("val_macro_auprc"))

    table_rows = [
        f"### Generalization Gap Analysis ({shift_direction.upper()})",
        "",
        "| Finding | In-Domain AUROC | Cross-Domain AUROC | Gap (Δ AUROC) | In-Domain AUPRC | Cross-Domain AUPRC | Gap (Δ AUPRC) | p-value | FDR Sig (q<0.05) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for idx, f in enumerate(classes):
        in_auc = _extract_metric_value(in_domain_metrics, f, "auroc")
        cross_auc = _extract_metric_value(cross_domain_metrics, f, "auroc")
        in_prc = _extract_metric_value(in_domain_metrics, f, "auprc")
        cross_prc = _extract_metric_value(cross_domain_metrics, f, "auprc")
        in_sens90 = _extract_metric_value(in_domain_metrics, f, "sens_at_90_spec")
        cross_sens90 = _extract_metric_value(cross_domain_metrics, f, "sens_at_90_spec")

        gap_auc = (in_auc - cross_auc) if (in_auc is not None and cross_auc is not None) else None
        gap_prc = (in_prc - cross_prc) if (in_prc is not None and cross_prc is not None) else None
        gap_sens90 = (in_sens90 - cross_sens90) if (in_sens90 is not None and cross_sens90 is not None) else None

        # Statistical testing on AUROC difference
        p_val = 1.0
        ci_diff = (0.0, 0.0)
        has_test = False

        # Case A: Paired testing on identical target cases (cross-domain model vs in-domain target baseline)
        if (
            target_patients is not None
            and target_targets is not None
            and target_preds is not None
            and target_in_domain_preds is not None
        ):
            y_t = target_targets[:, idx] if target_targets.ndim > 1 else target_targets
            y_cross = target_preds[:, idx] if target_preds.ndim > 1 else target_preds
            y_in = target_in_domain_preds[:, idx] if target_in_domain_preds.ndim > 1 else target_in_domain_preds

            valid = (y_t == 0.0) | (y_t == 1.0)
            if np.sum(valid) > 10 and len(np.unique(y_t[valid])) == 2:
                test_res = paired_bootstrap_auroc_diff(
                    patient_ids=target_patients[valid],
                    y_true=y_t[valid],
                    y_pred_a=y_in[valid],
                    y_pred_b=y_cross[valid],
                    n_resamples=n_resamples,
                    seed=seed,
                )
                p_val = test_res["p_value"]
                ci_diff = (test_res["ci_lower"], test_res["ci_upper"])
                has_test = True

        # Case B: Two-sample testing across disjoint source and target patient populations
        elif (
            source_patients is not None
            and source_targets is not None
            and source_preds is not None
            and target_patients is not None
            and target_targets is not None
            and target_preds is not None
        ):
            y_s_t = source_targets[:, idx] if source_targets.ndim > 1 else source_targets
            y_s_p = source_preds[:, idx] if source_preds.ndim > 1 else source_preds
            y_t_t = target_targets[:, idx] if target_targets.ndim > 1 else target_targets
            y_t_p = target_preds[:, idx] if target_preds.ndim > 1 else target_preds

            v_s = (y_s_t == 0.0) | (y_s_t == 1.0)
            v_t = (y_t_t == 0.0) | (y_t_t == 1.0)

            if np.sum(v_s) > 10 and np.sum(v_t) > 10:
                test_res = two_sample_bootstrap_diff(
                    patient_ids_a=source_patients[v_s],
                    y_true_a=y_s_t[v_s],
                    y_pred_a=y_s_p[v_s],
                    patient_ids_b=target_patients[v_t],
                    y_true_b=y_t_t[v_t],
                    y_pred_b=y_t_p[v_t],
                    n_resamples=n_resamples,
                    seed=seed,
                )
                p_val = test_res["p_value"]
                ci_diff = (test_res["ci_lower"], test_res["ci_upper"])
                has_test = True

        raw_p_values.append(p_val)
        evaluated_findings.append(f)

        gaps[f] = {
            "in_domain_auroc": in_auc,
            "cross_domain_auroc": cross_auc,
            "gap_auroc": gap_auc,
            "in_domain_auprc": in_prc,
            "cross_domain_auprc": cross_prc,
            "gap_auprc": gap_prc,
            "gap_sens_at_90_spec": gap_sens90,
            "p_value": p_val if has_test else None,
            "ci_diff_95": ci_diff if has_test else None,
        }

    # Apply FDR correction
    adj_p_values, sig_flags = benjamini_hochberg_fdr(raw_p_values, q=fdr_alpha)
    for i, f in enumerate(evaluated_findings):
        gaps[f]["fdr_adjusted_p_value"] = float(adj_p_values[i])
        gaps[f]["fdr_significant"] = bool(sig_flags[i])

        g_info = gaps[f]
        in_auc_str = f"{g_info['in_domain_auroc']:.4f}" if g_info['in_domain_auroc'] is not None else "—"
        cross_auc_str = f"{g_info['cross_domain_auroc']:.4f}" if g_info['cross_domain_auroc'] is not None else "—"
        gap_auc_str = f"{g_info['gap_auroc']:+.4f}" if g_info['gap_auroc'] is not None else "—"
        in_prc_str = f"{g_info['in_domain_auprc']:.4f}" if g_info['in_domain_auprc'] is not None else "—"
        cross_prc_str = f"{g_info['cross_domain_auprc']:.4f}" if g_info['cross_domain_auprc'] is not None else "—"
        gap_prc_str = f"{g_info['gap_auprc']:+.4f}" if g_info['gap_auprc'] is not None else "—"
        p_str = f"{g_info['p_value']:.4f}" if g_info['p_value'] is not None else "—"
        sig_str = "**Yes**" if g_info['fdr_significant'] else "No"

        table_rows.append(
            f"| **{f}** | {in_auc_str} | {cross_auc_str} | {gap_auc_str} | {in_prc_str} | {cross_prc_str} | {gap_prc_str} | {p_str} | {sig_str} |"
        )

    # Macro averages
    macro_gap_auc = (macro_in_auc - macro_cross_auc) if (macro_in_auc and macro_cross_auc) else None
    macro_gap_prc = (macro_in_prc - macro_cross_prc) if (macro_in_prc and macro_cross_prc) else None

    table_rows.append(
        f"| **Macro Average** | {macro_in_auc:.4f} | {macro_cross_auc:.4f} | {macro_gap_auc:+.4f} | {macro_in_prc:.4f} | {macro_cross_prc:.4f} | {macro_gap_prc:+.4f} | — | — |"
        if (macro_in_auc and macro_cross_auc and macro_in_prc and macro_cross_prc) else ""
    )

    markdown_report = "\n".join([r for r in table_rows if r])

    pneu_gap = gaps.get(HEADLINE_CLASS, {})
    in_auc = pneu_gap.get("in_domain_auroc")
    cd_auc = pneu_gap.get("cross_domain_auroc")
    delta_val = (cd_auc - in_auc) if (cd_auc is not None and in_auc is not None) else None
    headline_dict = {
        "in_domain_auroc": in_auc,
        "cross_domain_auroc": cd_auc,
        "gap_auroc": pneu_gap.get("gap_auroc"),
        "delta_auroc": delta_val,
        "delta_auroc_ci_95": pneu_gap.get("ci_diff_95", [delta_val, delta_val] if delta_val is not None else None),
        "p_value": pneu_gap.get("p_value"),
        "fdr_significant": pneu_gap.get("fdr_significant", False),
    }

    return {
        "shift_direction": shift_direction,
        "per_class_gaps": gaps,
        "macro_gap_auroc": macro_gap_auc,
        "macro_gap_auprc": macro_gap_prc,
        "headline_pneumonia": headline_dict,
        "headline_pneumonia_gap": headline_dict,
        "markdown_table": markdown_report,
    }

