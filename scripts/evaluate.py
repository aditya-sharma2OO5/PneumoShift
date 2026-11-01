"""In-domain and cross-dataset evaluation, calibration, and literature benchmark comparison."""

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES, get_evaluation_targets
from pneumoshift.data.splits import create_synthetic_dataset
from pneumoshift.data.transforms import get_transforms
from pneumoshift.evaluation.benchmarks import compare_against_benchmarks
from pneumoshift.evaluation.bootstrap import patient_bootstrap_ci
from pneumoshift.evaluation.calibration import compute_ece
from pneumoshift.evaluation.metrics import compute_discrimination_metrics
from pneumoshift.models.factory import build_model
from pneumoshift.utils.io import load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.evaluate")


def resolve_split_path(base_dir: Path, ds_name: str, split_name: str) -> Path:
    """Finds split CSV either directly in base_dir or in a dataset subfolder."""
    direct = base_dir / f"{ds_name}_{split_name}.csv"
    if direct.is_file():
        return direct
    sub = base_dir / ds_name / f"{ds_name}_{split_name}.csv"
    if sub.is_file():
        return sub
    return direct


def extract_features_and_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward inference across loader and extracts logits, probs, targets, features."""
    model.eval()
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_features: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            targets = batch["target"].to(device)
            logits, feats = model(imgs, return_features=True)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_features.append(feats.cpu().numpy())

    logits = np.vstack(all_logits)
    targets = np.vstack(all_targets)
    features = np.vstack(all_features)
    # Numerically stable sigmoid
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    return logits, probs, targets, features


def evaluate_split(
    model: torch.nn.Module,
    loader: DataLoader,
    df: pd.DataFrame,
    label_cols: List[str],
    device: torch.device,
    compute_bootstrap: bool = True,
    n_resamples: int = 1000,
    seed: int = 42,
    uncertainty_mode: str = "uncertain_as_negative",
) -> Dict[str, Any]:
    """Computes comprehensive evaluation metrics including bootstrap CIs."""
    _, probs, targets, _ = extract_features_and_predictions(model, loader, device)

    patient_ids = df["patient_id"].to_numpy() if "patient_id" in df.columns else np.arange(len(df))

    results: Dict[str, Any] = {
        "num_samples": len(df),
        "num_patients": int(len(np.unique(patient_ids))),
        "uncertainty_mode": uncertainty_mode,
        "per_class": {},
    }

    aurocs: List[float] = []
    auprcs: List[float] = []
    summary_metrics: Dict[str, float] = {}

    for idx, col in enumerate(label_cols):
        raw_y_true = targets[:, idx]
        y_true = get_evaluation_targets(raw_y_true, mode=uncertainty_mode)
        y_pred = probs[:, idx]

        disc = compute_discrimination_metrics(y_true, y_pred)
        ece_val, mce_val, _ = compute_ece(y_true, y_pred, n_bins=15)

        class_res: Dict[str, Any] = {
            "auroc": disc["auroc"],
            "auprc": disc["auprc"],
            "sens_at_90_spec": disc.get("sens_at_90_spec", float("nan")),
            "spec_at_90_sens": disc.get("spec_at_90_sens", float("nan")),
            "prevalence": disc.get("prevalence", 0.0),
            "ece": ece_val,
            "mce": mce_val,
        }

        # Clustered patient bootstrap confidence intervals
        if compute_bootstrap and not np.isnan(disc["auroc"]):
            valid_mask = (y_true == 0.0) | (y_true == 1.0)
            if np.sum(valid_mask) > 10 and len(np.unique(y_true[valid_mask])) == 2:
                _, auroc_lo, auroc_hi = patient_bootstrap_ci(
                    patient_ids[valid_mask],
                    y_true[valid_mask],
                    y_pred[valid_mask],
                    metric_fn=lambda yt, yp: compute_discrimination_metrics(yt, yp)["auroc"],
                    n_resamples=n_resamples,
                    seed=seed,
                )
                _, auprc_lo, auprc_hi = patient_bootstrap_ci(
                    patient_ids[valid_mask],
                    y_true[valid_mask],
                    y_pred[valid_mask],
                    metric_fn=lambda yt, yp: compute_discrimination_metrics(yt, yp)["auprc"],
                    n_resamples=n_resamples,
                    seed=seed,
                )
                class_res["auroc_ci_95"] = [auroc_lo, auroc_hi]
                class_res["auprc_ci_95"] = [auprc_lo, auprc_hi]

        results["per_class"][col] = class_res

        if not np.isnan(disc["auroc"]):
            aurocs.append(disc["auroc"])
        if not np.isnan(disc["auprc"]):
            auprcs.append(disc["auprc"])

        col_key = col.lower().replace(" ", "_")
        summary_metrics[f"auroc_{col_key}"] = disc["auroc"]
        summary_metrics[f"auprc_{col_key}"] = disc["auprc"]

    macro_auroc = float(np.mean(aurocs)) if aurocs else 0.0
    macro_auprc = float(np.mean(auprcs)) if auprcs else 0.0
    results["macro_auroc"] = macro_auroc
    results["macro_auprc"] = macro_auprc
    summary_metrics["macro_auroc"] = macro_auroc
    summary_metrics["macro_auprc"] = macro_auprc

    if "Pneumonia" in results["per_class"]:
        pneu_metrics = results["per_class"]["Pneumonia"]
        results["headline_pneumonia"] = pneu_metrics
        summary_metrics["pneumonia_auroc"] = pneu_metrics["auroc"]
        summary_metrics["pneumonia_auprc"] = pneu_metrics["auprc"]

    results["summary_flat"] = summary_metrics
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PneumoShift model on in-domain or cross-dataset cohorts.")
    parser.add_argument("--config", default="configs/train/nih_densenet121.yaml", help="Path to experiment config YAML")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint (.pt). Defaults to best checkpoint.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "target_cal", "official_val"], help="Dataset split to evaluate")
    parser.add_argument("--smoke_test", action="store_true", help="Run in synthetic dry-run smoke test mode")
    parser.add_argument("--output_dir", default=None, help="Directory to save evaluation tables and reports")
    parser.add_argument("--benchmark", default=None, help="Literature benchmark override (e.g. pan_2019_nih, irvin_2019_chexpert, arasteh_2023_chexpert)")
    parser.add_argument("--uncertainty_mode", default=None, choices=["uncertain_as_negative", "uncertain_excluded"], help="Uncertainty evaluation policy (default from config or uncertain_as_negative)")
    parser.add_argument("--no_bootstrap", action="store_true", help="Disable computation of bootstrap confidence intervals")
    parser.add_argument("--n_resamples", type=int, default=1000, help="Number of bootstrap resamples (default 1000)")
    args = parser.parse_args()

    cfg = load_yaml_config("configs/smoke_test.yaml" if args.smoke_test else args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = cfg.get("model", {}).get("architecture", "densenet121")
    img_size = cfg.get("data", {}).get("image_size", 224)
    headline_only = cfg.get("data", {}).get("headline_only", False)
    label_cols = [HEADLINE_CLASS] if headline_only else SHARED_CLASSES
    num_classes = 1 if headline_only else len(label_cols)
    ds_name = cfg.get("data", {}).get("dataset_name", "nih")

    # Output directory
    out_dir = Path(args.output_dir or cfg.get("eval", {}).get("output_dir", f"results/tables/{ds_name}_{arch}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Locate dataset split CSV
    if args.smoke_test or ds_name == "synthetic":
        synth_dir = Path("data/synthetic")
        if not (synth_dir / "splits" / f"synthetic_{args.split}.csv").is_file():
            create_synthetic_dataset(output_dir=synth_dir, num_samples=40, image_size=img_size)
        split_csv = synth_dir / "splits" / f"synthetic_{args.split}.csv"
        img_dir = synth_dir / "images"
    else:
        splits_dir = Path(cfg.get("data", {}).get("splits_dir", "data/splits"))
        split_csv = resolve_split_path(splits_dir, ds_name, args.split)
        if not split_csv.is_file():
            logger.warning("Split file %s not found. Falling back to synthetic smoke data.", split_csv)
            synth_dir = Path("data/synthetic")
            create_synthetic_dataset(output_dir=synth_dir, num_samples=40, image_size=img_size)
            split_csv = synth_dir / "splits" / "synthetic_test.csv"
            img_dir = synth_dir / "images"
        else:
            img_dir = Path(cfg.get("data", {}).get("images_dir", cfg.get("data", {}).get("data_dir", "")))

    logger.info("Evaluating split: %s (%s)", args.split, split_csv)
    df = pd.read_csv(split_csv)
    transform = get_transforms(split="test", image_size=img_size)
    dataset = ChestXrayDataset(df, images_dir=img_dir, transform=transform, headline_only=headline_only)
    loader = DataLoader(dataset, batch_size=cfg.get("eval", {}).get("batch_size", 32), shuffle=False, num_workers=0)

    # Build model & load checkpoint
    model = build_model(architecture=arch, num_classes=num_classes, pretrained=False)
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        ckpt_dir = Path(cfg.get("train", {}).get("checkpoint_dir", f"results/runs/{ds_name}_{arch}/checkpoints"))
        ckpt_path = ckpt_dir / "checkpoint_best.pt"
        if not ckpt_path.is_file():
            ckpt_path = ckpt_dir / "checkpoint_latest.pt"

    ckpt_path = Path(ckpt_path) if ckpt_path else None
    if ckpt_path and ckpt_path.is_file():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded model weights from checkpoint: %s (epoch %s)", ckpt_path, ckpt.get("epoch", "?"))
    else:
        logger.warning("Checkpoint %s not found. Running evaluation with initialized weights.", ckpt_path)

    model.to(device)

    # Resolve uncertainty evaluation modes
    default_mode = args.uncertainty_mode or "uncertain_as_negative"
    modes_to_eval = [default_mode]
    if args.uncertainty_mode is None and "eval_uncertainty_modes" in cfg.get("eval", {}):
        modes_to_eval = cfg.get("eval", {}).get("eval_uncertainty_modes", [default_mode])

    # Deduce benchmark reference
    if args.benchmark:
        bench_name = args.benchmark
    elif "nih" in ds_name:
        bench_name = "pan_2019_nih"
    elif "chexpert" in ds_name:
        if args.split == "official_val":
            bench_name = "irvin_2019_chexpert"
        else:
            bench_name = "arasteh_2023_chexpert"
    else:
        bench_name = "pan_2019_nih"

    for mode in modes_to_eval:
        logger.info("Executing evaluation under uncertainty mode: %s", mode)
        results = evaluate_split(
            model=model,
            loader=loader,
            df=df,
            label_cols=label_cols,
            device=device,
            compute_bootstrap=not args.no_bootstrap and not args.smoke_test,
            n_resamples=args.n_resamples,
            seed=cfg.get("seed", 42),
            uncertainty_mode=mode,
        )

        mode_suffix = f"_{mode}" if len(modes_to_eval) > 1 else ""

        # Save detailed JSON metrics
        json_path = out_dir / f"eval_metrics_{args.split}{mode_suffix}.json"
        save_json(results, json_path)
        logger.info("Saved evaluation JSON to %s", json_path)

        # Save summary CSV table
        csv_path = out_dir / f"eval_metrics_{args.split}{mode_suffix}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["finding", "auroc", "auroc_ci_lower", "auroc_ci_upper", "auprc", "auprc_ci_lower", "auprc_ci_upper", "sens_at_90_spec", "spec_at_90_sens", "prevalence", "ece"])
            for col, m in results["per_class"].items():
                auc_ci = m.get("auroc_ci_95", ["", ""])
                prc_ci = m.get("auprc_ci_95", ["", ""])
                writer.writerow([
                    col,
                    f"{m['auroc']:.4f}" if not np.isnan(m['auroc']) else "",
                    f"{auc_ci[0]:.4f}" if isinstance(auc_ci[0], float) else "",
                    f"{auc_ci[1]:.4f}" if isinstance(auc_ci[1], float) else "",
                    f"{m['auprc']:.4f}" if not np.isnan(m['auprc']) else "",
                    f"{prc_ci[0]:.4f}" if isinstance(prc_ci[0], float) else "",
                    f"{prc_ci[1]:.4f}" if isinstance(prc_ci[1], float) else "",
                    f"{m.get('sens_at_90_spec', 0.0):.4f}",
                    f"{m.get('spec_at_90_sens', 0.0):.4f}",
                    f"{m.get('prevalence', 0.0):.4f}",
                    f"{m.get('ece', 0.0):.4f}",
                ])
        logger.info("Saved evaluation CSV table to %s", csv_path)

        # Literature benchmark comparison
        try:
            bench_comp = compare_against_benchmarks(
                evaluated_metrics=results["summary_flat"],
                benchmark_name=bench_name,
                headline_class=HEADLINE_CLASS,
            )
            save_json(bench_comp, out_dir / f"benchmark_comparison_{args.split}{mode_suffix}.json")
            with open(out_dir / f"benchmark_comparison_{args.split}{mode_suffix}.md", "w", encoding="utf-8") as f:
                f.write(f"# Benchmark Comparison: {bench_name} ({mode})\n\n")
                f.write(bench_comp["table_markdown"] + "\n\n")
                f.write(f"**Sanity Status**: {bench_comp['headline_sanity_message']}\n")

            print("\n" + "=" * 70)
            print(f"BENCHMARK COMPARISON ({bench_name.upper()} vs EVALUATED - Mode: {mode})")
            print("=" * 70)
            print(bench_comp["table_markdown"])
            print("-" * 70)
            print(f"Sanity Check: {bench_comp['headline_sanity_message']}")
            print("=" * 70 + "\n")
        except Exception as exc:
            logger.warning("Could not run benchmark comparison: %s", exc)
            f.write(f"**Sanity Status**: {bench_comp['headline_sanity_message']}\n")

        print("\n" + "=" * 70)
        print(f"BENCHMARK COMPARISON ({bench_name.upper()} vs EVALUATED)")
        print("=" * 70)
        print(bench_comp["table_markdown"])
        print("-" * 70)
        print(f"Sanity Check: {bench_comp['headline_sanity_message']}")
        print("=" * 70 + "\n")
    except Exception as exc:
        logger.warning("Could not run benchmark comparison: %s", exc)


if __name__ == "__main__":
    main()
