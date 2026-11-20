"""Training CLI entry point for PneumoShift with multi-seed replication support."""

import argparse
import copy
import datetime
import hashlib
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.data.splits import create_synthetic_dataset
from pneumoshift.data.transforms import get_transforms
from pneumoshift.evaluation.benchmarks import compare_against_benchmarks
from pneumoshift.evaluation.thresholds import compute_optimal_thresholds
from pneumoshift.models.factory import build_model
from pneumoshift.training.losses import MaskedBCEWithLogitsLoss, compute_class_weights
from pneumoshift.training.trainer import Trainer
from pneumoshift.utils.io import load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger
from pneumoshift.utils.seed import set_seed

logger = get_logger("scripts.train")


def compute_file_sha256(path: Path) -> str:
    """Computes SHA-256 hex digest for a file."""
    if not path.is_file():
        return "file_not_found"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint_file(path: Path, device: torch.device) -> Dict[str, Any]:
    """Safely loads PyTorch checkpoint with backward compatibility across PyTorch versions."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def seed_worker(worker_id: int) -> None:
    """Ensures PyTorch DataLoader workers have distinct, deterministic seeds."""
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def resolve_split_path(base_dir: Path, ds_name: str, split_name: str) -> Path:
    """Finds split CSV either directly in base_dir or in a dataset subfolder."""
    direct = base_dir / f"{ds_name}_{split_name}.csv"
    if direct.is_file():
        return direct
    sub = base_dir / ds_name / f"{ds_name}_{split_name}.csv"
    if sub.is_file():
        return sub
    return direct


def extract_batch_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Runs forward inference to extract logits, probabilities, ground-truth targets, features, patient IDs, and image IDs."""
    model.eval()
    all_logits: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    all_features: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            tgts = batch["target"].to(device)
            logits, feats = model(imgs, return_features=True)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(tgts.cpu().numpy())
            all_features.append(feats.cpu().numpy())
    logits = np.vstack(all_logits)
    targets = np.vstack(all_targets)
    features = np.vstack(all_features)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))

    # Patient IDs
    if "patient_id" in df.columns:
        patient_ids = df["patient_id"].to_numpy()
    elif "Patient ID" in df.columns:
        patient_ids = df["Patient ID"].to_numpy()
    else:
        patient_ids = np.arange(len(df))

    # Image IDs
    if "image_id" in df.columns:
        image_ids = df["image_id"].astype(str).to_numpy()
    elif "Image Index" in df.columns:
        image_ids = df["Image Index"].astype(str).to_numpy()
    elif "image_path" in df.columns:
        image_ids = df["image_path"].apply(lambda p: Path(str(p)).name).astype(str).to_numpy()
    elif "Path" in df.columns:
        image_ids = df["Path"].apply(lambda p: Path(str(p)).name).astype(str).to_numpy()
    else:
        image_ids = np.array([f"img_{i}" for i in range(len(df))])

    return logits, probs, targets, features, patient_ids, image_ids


def parse_args() -> argparse.Namespace:
    """Parses training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train deep learning model for PneumoShift.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--smoke_test", action="store_true", help="Run 1-epoch smoke test and project execution time")
    parser.add_argument("--resume", action="store_true", help="Resume from last.pt checkpoint if present")
    parser.add_argument("--max_hours", type=float, default=None, help="Maximum execution time in hours before graceful exit")
    parser.add_argument("--seed", type=int, default=None, help="Single random seed override")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated random seeds (e.g. '42,1337,2024')")
    parser.add_argument("--architecture", default=None, help="Model architecture override")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs override")
    parser.add_argument("--eval_test", action="store_true", help="Evaluate best checkpoint on test set after training")
    parser.add_argument("--run_id", type=str, default="repro_v1", help="Unique identifier for run/experiment (default: repro_v1)")
    parser.add_argument("--smoke_dir", type=str, default="outputs/smoke", help="Directory to isolate smoke test outputs")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save raw prediction arrays (.npz)")
    parser.add_argument("--allow_overwrite", action="store_true", help="Permit overwriting existing prediction arrays")
    return parser.parse_args()


def write_run_manifest(
    output_path: Path,
    seed: int,
    epochs: int,
    arch: str,
    config_path: Path,
    splits_dir: Path,
    checkpoint_paths: Dict[str, Path],
    start_time_iso: str,
    end_time_iso: str,
    best_epoch: int,
    best_val_pneumonia_auroc: float,
    run_id: str = "repro_v1",
) -> None:
    """Writes a standardized forensic run manifest documenting exact environment and dataset hashes."""
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        git_commit = "unknown"

    try:
        pip_freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode("utf-8").splitlines()
    except Exception:
        pip_freeze = []

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    cuda_version = torch.version.cuda or "N/A"
    torch_version = torch.__version__

    config_hash = compute_file_sha256(config_path) if config_path.is_file() else "N/A"

    split_hashes = {}
    if splits_dir.is_dir():
        for split_file in sorted(splits_dir.glob("*.csv")):
            split_hashes[split_file.name] = compute_file_sha256(split_file)

    ckpt_hashes = {}
    for name, ckpt_p in checkpoint_paths.items():
        if ckpt_p.is_file():
            ckpt_hashes[name] = compute_file_sha256(ckpt_p)

    manifest = {
        "run_id": f"{run_id}_{arch}_seed_{seed}_{start_time_iso.replace(':', '-')}",
        "architecture": arch,
        "seed": seed,
        "epochs": epochs,
        "start_time": start_time_iso,
        "end_time": end_time_iso,
        "best_epoch": best_epoch,
        "best_val_pneumonia_auroc": best_val_pneumonia_auroc,
        "rng_seeding_protocol": (
            "Full multi-process reseeding enabled via worker_init_fn and torch.Generator. "
            "RNG streams differ from legacy unseeded DataLoader runs (pre-cb8f978 / unseeded worker loops), "
            "guaranteeing strict worker-level determinism."
        ),
        "git_commit": git_commit,
        "gpu_name": gpu_name,
        "cuda_version": cuda_version,
        "torch_version": torch_version,
        "config_path": str(config_path),
        "config_sha256": config_hash,
        "split_sha256": split_hashes,
        "checkpoint_sha256": ckpt_hashes,
        "pip_freeze": pip_freeze,
    }

    save_json(manifest, output_path)
    logger.info("Saved run manifest to %s", output_path)


def main() -> None:
    args = parse_args()

    if args.config != "configs/default.yaml":
        config_path = Path(args.config)
    elif args.smoke_test and Path("configs/smoke_test.yaml").is_file():
        config_path = Path("configs/smoke_test.yaml")
    else:
        config_path = Path(args.config)

    cfg = load_yaml_config(config_path)

    # Resolve seed list
    if args.seeds:
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    elif args.seed is not None:
        seeds = [args.seed]
    elif "seeds" in cfg and isinstance(cfg["seeds"], list):
        seeds = [int(s) for s in cfg["seeds"]]
    else:
        seeds = [int(cfg.get("seed", 42))]

    arch = args.architecture or cfg.get("model", {}).get("architecture", "densenet121")
    epochs = args.epochs or cfg.get("train", {}).get("epochs", 10)
    batch_size = cfg.get("data", {}).get("batch_size", 32)
    num_workers = cfg.get("data", {}).get("num_workers", 0)
    img_size = cfg.get("data", {}).get("image_size", 224)
    headline_only = cfg.get("data", {}).get("headline_only", False)
    cache_preprocessed = cfg.get("data", {}).get("cache_preprocessed", False)
    cache_dir = cfg.get("data", {}).get("cache_dir", None) if cache_preprocessed else None
    ds_name = cfg.get("data", {}).get("dataset_name", "nih")
    splits_dir = Path(cfg.get("data", {}).get("splits_dir", "data/splits"))
    synth_dir = Path("data/synthetic")

    run_id = args.run_id
    if args.smoke_test:
        base_run_out_dir = Path(args.smoke_dir) / "runs" / run_id / arch
        base_checkpoint_dir = Path(args.smoke_dir) / "checkpoints" / run_id / arch
        pred_dir = Path(args.smoke_dir) / "predictions" / run_id
        logger.info("SMOKE TEST MODE ENABLED: Routing outputs strictly to '%s' (zero impact on results/predictions)", args.smoke_dir)
    else:
        base_run_out_dir = Path(cfg.get("train", {}).get("run_output_dir", "results/runs"))
        base_checkpoint_dir = Path(cfg.get("train", {}).get("checkpoint_dir", "results/runs/checkpoints"))
        pred_dir = Path(args.output_dir or "results/predictions")

    base_run_out_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing reproducible training suite:")
    logger.info("  Architecture: %s | Epochs: %d | Headline Only: %s", arch, epochs, headline_only)
    logger.info("  Seeds to execute: %s (%d seeds total)", seeds, len(seeds))
    if args.resume:
        logger.info("  Resume mode: ENABLED (reloading from last.pt if present)")
    if args.max_hours:
        logger.info("  Max execution time: %.2f hours", args.max_hours)

    all_seed_metrics: List[Dict[str, Any]] = []

    for seed_idx, seed in enumerate(seeds):
        logger.info("\n" + "=" * 70)
        logger.info(">>> STARTING SEED %d / %d (seed=%d) <<<", seed_idx + 1, len(seeds), seed)
        logger.info("=" * 70)

        # Full multi-process reseeding
        set_seed(seed)
        g = torch.Generator()
        g.manual_seed(seed)

        seed_cfg = copy.deepcopy(cfg)
        seed_cfg["seed"] = seed

        # Isolate checkpoints and run outputs per seed if multi-seed
        if len(seeds) > 1:
            seed_ckpt_dir = base_checkpoint_dir / f"seed_{seed}"
            seed_run_dir = base_run_out_dir / f"seed_{seed}"
        else:
            seed_ckpt_dir = base_checkpoint_dir
            seed_run_dir = base_run_out_dir

        seed_ckpt_dir.mkdir(parents=True, exist_ok=True)
        seed_run_dir.mkdir(parents=True, exist_ok=True)

        seed_cfg.setdefault("train", {})
        seed_cfg["train"]["checkpoint_dir"] = str(seed_ckpt_dir)
        seed_cfg["train"]["run_output_dir"] = str(seed_run_dir)

        # Assertion 1: Protect existing completed results unless --resume is requested
        done_marker = seed_run_dir / f"DONE_seed_{seed}.json"
        if not args.resume and not args.smoke_test and done_marker.is_file():
            raise RuntimeError(
                f"Completed run marker {done_marker} already exists. "
                "To protect previous empirical results from overwrite, use --resume to continue or clean the target folder."
            )

        test_df: Optional[pd.DataFrame] = None
        if ds_name == "synthetic":
            create_synthetic_dataset(output_dir=synth_dir, num_samples=30, image_size=img_size, seed=seed)
            train_df = pd.read_csv(synth_dir / "splits" / "synthetic_train.csv")
            val_df = pd.read_csv(synth_dir / "splits" / "synthetic_val.csv")
            img_dir = synth_dir / "images"
        else:
            train_csv = resolve_split_path(splits_dir, ds_name, "train")
            val_csv = resolve_split_path(splits_dir, ds_name, "val")

            if not train_csv.is_file() or not val_csv.is_file():
                logger.warning("Split files not found (%s, %s). Falling back to synthetic data.", train_csv, val_csv)
                create_synthetic_dataset(output_dir=synth_dir, num_samples=30, image_size=img_size, seed=seed)
                train_df = pd.read_csv(synth_dir / "splits" / "synthetic_train.csv")
                val_df = pd.read_csv(synth_dir / "splits" / "synthetic_val.csv")
                img_dir = synth_dir / "images"
            else:
                train_df = pd.read_csv(train_csv)
                val_df = pd.read_csv(val_csv)
                img_dir = Path(cfg.get("data", {}).get("images_dir", cfg.get("data", {}).get("data_dir", "")))

                # Resolve test split for leak check
                test_csv = resolve_split_path(splits_dir, ds_name, "test")
                if test_csv.is_file():
                    test_df = pd.read_csv(test_csv)

        # Assertion 2: Patient-level splits with zero overlap
        if "patient_id" in train_df.columns and "patient_id" in val_df.columns:
            train_pats = set(train_df["patient_id"])
            val_pats = set(val_df["patient_id"])
            overlap_val = train_pats.intersection(val_pats)
            assert len(overlap_val) == 0, (
                f"Patient leakage detected: {len(overlap_val)} overlapping patients between train and validation splits!"
            )
            if test_df is not None and "patient_id" in test_df.columns:
                test_pats = set(test_df["patient_id"])
                overlap_test = train_pats.intersection(test_pats)
                overlap_val_test = val_pats.intersection(test_pats)
                assert len(overlap_test) == 0, (
                    f"Patient leakage detected: {len(overlap_test)} overlapping patients between train and test splits!"
                )
                assert len(overlap_val_test) == 0, (
                    f"Patient leakage detected: {len(overlap_val_test)} overlapping patients between val and test splits!"
                )

        train_trans = get_transforms(split="train", image_size=img_size)
        val_trans = get_transforms(split="val", image_size=img_size)

        label_cols = [HEADLINE_CLASS] if headline_only else SHARED_CLASSES
        train_ds = ChestXrayDataset(train_df, images_dir=img_dir, transform=train_trans, headline_only=headline_only, cache_dir=cache_dir)
        val_ds = ChestXrayDataset(val_df, images_dir=img_dir, transform=val_trans, headline_only=headline_only, cache_dir=cache_dir)

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            generator=g,
            worker_init_fn=seed_worker,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            generator=g,
            worker_init_fn=seed_worker,
        )

        # Inverse frequency class weighting
        pos_weight = compute_class_weights(train_df, label_cols=label_cols) if seed_cfg.get("train", {}).get("pos_weight") == "inverse_freq" else None
        criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)

        num_classes = 1 if headline_only else seed_cfg.get("model", {}).get("num_classes", 8)
        model = build_model(
            architecture=arch,
            num_classes=num_classes,
            pretrained=seed_cfg.get("model", {}).get("pretrained", False if args.smoke_test else True),
            dropout=seed_cfg.get("model", {}).get("dropout", 0.2),
        )

        lr = float(seed_cfg.get("train", {}).get("learning_rate", 1e-4))
        wd = float(seed_cfg.get("train", {}).get("weight_decay", 1e-5))
        opt_type = seed_cfg.get("train", {}).get("optimizer", "adamw").lower()
        two_stage = seed_cfg.get("train", {}).get("two_stage_finetune", False)

        if two_stage:
            param_groups = [
                {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
                {"params": list(model.head.parameters()), "lr": lr, "name": "head"},
            ]
        else:
            param_groups = [{"params": model.parameters(), "lr": lr}]

        if opt_type == "adam":
            optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=wd)
        else:
            optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)

        patience = seed_cfg.get("train", {}).get("lr_patience", 2)
        factor = seed_cfg.get("train", {}).get("lr_factor", 0.5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config=seed_cfg,
            label_cols=label_cols,
        )

        # Resume support: check for last.pt
        start_epoch = 1
        last_ckpt_path = seed_ckpt_dir / "last.pt"
        if args.resume and last_ckpt_path.is_file():
            logger.info("Resuming seed %d from last checkpoint: %s", seed, last_ckpt_path)
            ckpt_state = load_checkpoint_file(last_ckpt_path, trainer.device)
            model.load_state_dict(ckpt_state["model_state_dict"])
            optimizer.load_state_dict(ckpt_state["optimizer_state_dict"])
            if scheduler and ckpt_state.get("scheduler_state_dict"):
                scheduler.load_state_dict(ckpt_state["scheduler_state_dict"])
            if hasattr(trainer, "scaler") and trainer.scaler and ckpt_state.get("scaler_state_dict"):
                trainer.scaler.load_state_dict(ckpt_state["scaler_state_dict"])
            if "rng_state_torch" in ckpt_state:
                torch.set_rng_state(ckpt_state["rng_state_torch"])
            if "rng_state_cuda" in ckpt_state and ckpt_state["rng_state_cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(ckpt_state["rng_state_cuda"])
            if "rng_state_numpy" in ckpt_state:
                np.random.set_state(ckpt_state["rng_state_numpy"])
            if "rng_state_python" in ckpt_state:
                random.setstate(ckpt_state["rng_state_python"])

            start_epoch = ckpt_state["epoch"] + 1
            trainer.checkpoint_cb.best_score = ckpt_state.get("best_score", -float("inf"))
            trainer.checkpoint_cb.best_secondary_score = ckpt_state.get("best_secondary_score", -float("inf"))
            trainer.best_epoch = ckpt_state.get("best_epoch", ckpt_state["epoch"])
            logger.info("Resumed state successfully. Next epoch: %d (best epoch so far: %d)", start_epoch, trainer.best_epoch)

        start_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        start_time = time.time()

        if args.smoke_test:
            logger.info(">>> RUNNING SMOKE TEST: 1 epoch execution <<<")
            t_smoke_0 = time.time()
            history = trainer.fit(
                epochs=start_epoch,
                start_epoch=start_epoch,
                max_hours=args.max_hours,
                start_time=start_time,
            )
            smoke_sec = time.time() - t_smoke_0
            logger.info("\n" + "=" * 70)
            logger.info("[SMOKE TEST COMPLETED] 1 epoch took: %.2f seconds (%.2f minutes)", smoke_sec, smoke_sec / 60.0)
            logger.info("[PROJECTION] 10 epochs (1 seed):   %.2f minutes (%.2f hours)", (10 * smoke_sec) / 60.0, (10 * smoke_sec) / 3600.0)
            logger.info("[PROJECTION] 10 epochs x 3 seeds:  %.2f hours (%.2f minutes)", (30 * smoke_sec) / 3600.0, (30 * smoke_sec) / 60.0)
            logger.info("=" * 70 + "\n")
        elif start_epoch <= epochs:
            history = trainer.fit(
                epochs=epochs,
                start_epoch=start_epoch,
                max_hours=args.max_hours,
                start_time=start_time,
            )
        else:
            logger.info("Seed %d has already completed all %d epochs. Skipping fit.", seed, epochs)
            history = {"best_epoch": trainer.best_epoch, "val_loss": [], "train_loss": []}

        if trainer.stopped_due_to_timeout:
            logger.warning("Seed %d stopped cleanly due to time limit. Checkpoints are preserved.", seed)
            continue

        end_time_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()

        # Save history to both per-seed and base run directory
        save_json(history, seed_run_dir / f"history_{arch}_{seed}.json")
        save_json(history, base_run_out_dir / f"history_{arch}_{seed}.json")
        logger.info("Seed %d training complete. Saved history to %s", seed, seed_run_dir)

        # Mirror first seed checkpoint into base_checkpoint_dir for backwards compatibility
        best_ckpt_path = trainer.checkpoint_cb.checkpoint_dir / "checkpoint_best.pt"
        if best_ckpt_path.is_file() and len(seeds) > 1 and seed_idx == 0:
            shutil.copy2(best_ckpt_path, base_checkpoint_dir / "checkpoint_best.pt")
            sec_ckpt = trainer.checkpoint_cb.checkpoint_dir / "checkpoint_best_pneumonia.pt"
            if sec_ckpt.is_file():
                shutil.copy2(sec_ckpt, base_checkpoint_dir / "checkpoint_best_pneumonia.pt")
            latest_ckpt = trainer.checkpoint_cb.checkpoint_dir / "checkpoint_latest.pt"
            if latest_ckpt.is_file():
                shutil.copy2(latest_ckpt, base_checkpoint_dir / "checkpoint_latest.pt")
            last_ckpt = trainer.checkpoint_cb.checkpoint_dir / "last.pt"
            if last_ckpt.is_file():
                shutil.copy2(last_ckpt, base_checkpoint_dir / "last.pt")

        if args.eval_test or args.smoke_test:
            logger.info("Evaluating seed %d best model checkpoint...", seed)
            if best_ckpt_path.is_file():
                ckpt = load_checkpoint_file(best_ckpt_path, trainer.device)
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded best model checkpoint from %s (best epoch %s)", best_ckpt_path, ckpt.get("epoch", "?"))

            # 1. Extract and save Source Validation predictions & select tau strictly on source val
            logger.info("Extracting raw prediction arrays on source validation split (seed=%d)...", seed)
            v_logits, v_probs, v_targets, v_feats, v_patients, v_image_ids = extract_batch_predictions(
                model, val_loader, trainer.device, val_df
            )
            pneu_idx = label_cols.index("Pneumonia") if "Pneumonia" in label_cols else 5
            val_pred_path = pred_dir / f"predictions_val_{arch}_{seed}.npz"
            if val_pred_path.is_file() and not args.smoke_test and not args.allow_overwrite and not args.resume:
                raise RuntimeError(
                    f"Prediction file '{val_pred_path}' already exists! Refusing to overwrite existing empirical results. "
                    "Pass --allow_overwrite to explicitly permit overwriting, or specify a distinct --run_id / --output_dir."
                )

            val_row_order_hash = hashlib.sha256("".join(v_image_ids.astype(str)).encode("utf-8")).hexdigest()
            np.savez_compressed(
                val_pred_path,
                y_true=v_targets,
                y_score=v_probs,
                y_logits=v_logits,
                image_ids=v_image_ids,
                patient_ids=v_patients,
                pneumonia_col_idx=np.int64(pneu_idx),
                label_cols=np.array(label_cols),
                split_csv_row_order_hash=val_row_order_hash,
                features=v_feats,
            )
            shutil.copy2(val_pred_path, seed_run_dir / f"predictions_val_{arch}_{seed}.npz")
            logger.info("Saved source validation prediction arrays (.npz) to %s (row_order_hash: %s)", val_pred_path, val_row_order_hash[:10])

            raw_yt = v_targets[:, pneu_idx]
            yt = np.where(raw_yt == -1.0, 0.0, raw_yt)
            yp = v_probs[:, pneu_idx]
            opt_th = compute_optimal_thresholds(yt, yp)
            tau_data = {
                "architecture": arch,
                "seed": seed,
                "dataset": ds_name,
                "selection_split": "val",
                "source_val_optimal_tau": float(opt_th["youden_threshold"]),
                "youden_tau": float(opt_th["youden_threshold"]),
                "youden_metrics": opt_th["youden"]["metrics"],
                "max_f1_tau": float(opt_th["f1_threshold"]),
                "max_f1_metrics": opt_th["max_f1"]["metrics"],
                "sens90_tau": float(opt_th["sens90_threshold"]),
                "sens90_metrics": opt_th["fixed_sensitivity"]["metrics"],
                "description": "Selected strictly on source validation set (zero target leakage)",
            }
            save_json(tau_data, pred_dir / f"tau_selection_{arch}_{seed}.json")
            save_json(tau_data, seed_run_dir / f"tau_selection_{arch}_{seed}.json")
            logger.info("Selected source validation optimal threshold tau=%.3f (Youden) / %.3f (F1). Saved to %s",
                        tau_data["youden_tau"], tau_data["max_f1_tau"], pred_dir / f"tau_selection_{arch}_{seed}.json")

            if ds_name == "synthetic":
                test_csv = synth_dir / "splits" / "synthetic_test.csv"
            else:
                test_csv = resolve_split_path(splits_dir, ds_name, "test")
                if not test_csv.is_file():
                    test_csv = synth_dir / "splits" / "synthetic_test.csv"

            if test_csv.is_file():
                test_df = pd.read_csv(test_csv)
                test_trans = get_transforms(split="test", image_size=img_size)
                test_ds = ChestXrayDataset(test_df, images_dir=img_dir, transform=test_trans, headline_only=headline_only, cache_dir=cache_dir)
                test_loader = DataLoader(
                    test_ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    generator=g,
                    worker_init_fn=seed_worker,
                )

                # 2. Extract and save Source Test predictions
                logger.info("Extracting raw prediction arrays on test split (seed=%d)...", seed)
                t_logits, t_probs, t_targets, t_feats, t_patients, t_image_ids = extract_batch_predictions(
                    model, test_loader, trainer.device, test_df
                )
                test_pred_path = pred_dir / f"predictions_test_{arch}_{seed}.npz"
                if test_pred_path.is_file() and not args.smoke_test and not args.allow_overwrite and not args.resume:
                    raise RuntimeError(
                        f"Prediction file '{test_pred_path}' already exists! Refusing to overwrite existing empirical results. "
                        "Pass --allow_overwrite to explicitly permit overwriting, or specify a distinct --run_id / --output_dir."
                    )

                test_row_order_hash = hashlib.sha256("".join(t_image_ids.astype(str)).encode("utf-8")).hexdigest()
                np.savez_compressed(
                    test_pred_path,
                    y_true=t_targets,
                    y_score=t_probs,
                    y_logits=t_logits,
                    image_ids=t_image_ids,
                    patient_ids=t_patients,
                    pneumonia_col_idx=np.int64(pneu_idx),
                    label_cols=np.array(label_cols),
                    split_csv_row_order_hash=test_row_order_hash,
                    features=t_feats,
                )
                shutil.copy2(test_pred_path, seed_run_dir / f"predictions_test_{arch}_{seed}.npz")
                logger.info("Saved source test prediction arrays (.npz) to %s (row_order_hash: %s)", test_pred_path, test_row_order_hash[:10])

                eval_modes = seed_cfg.get("eval", {}).get("eval_uncertainty_modes", ["uncertain_excluded"])
                if not isinstance(eval_modes, list):
                    eval_modes = [eval_modes]

                for mode_idx, mode in enumerate(eval_modes):
                    test_metrics = trainer.evaluate(test_loader, uncertainty_mode=mode)
                    save_json(test_metrics, seed_run_dir / f"test_metrics_{arch}_{seed}_{mode}.json")
                    save_json(test_metrics, base_run_out_dir / f"test_metrics_{arch}_{seed}_{mode}.json")
                    if mode_idx == 0:
                        save_json(test_metrics, seed_run_dir / f"test_metrics_{arch}_{seed}.json")
                        save_json(test_metrics, base_run_out_dir / f"test_metrics_{arch}_{seed}.json")
                        all_seed_metrics.append({"seed": seed, "metrics": test_metrics})

                    logger.info("Test Eval (seed=%d, mode=%s): Macro AUROC: %.4f | Pneumonia AUROC: %.4f",
                                seed, mode, test_metrics.get("val_macro_auroc", 0.0), test_metrics.get("val_pneumonia_auroc", 0.0))

                bench_name = "pan_2019_nih" if "nih" in ds_name else "arasteh_2023_chexpert"
                try:
                    bench_comp = compare_against_benchmarks(
                        evaluated_metrics=test_metrics,
                        benchmark_name=bench_name,
                        headline_class=HEADLINE_CLASS,
                    )
                    save_json(bench_comp, seed_run_dir / f"test_benchmark_{arch}_{seed}.json")
                    save_json(bench_comp, base_run_out_dir / f"test_benchmark_{arch}_{seed}.json")
                except Exception as exc:
                    logger.warning("Could not run benchmark comparison for seed %d: %s", seed, exc)

        # Write DONE marker per seed
        done_payload = {
            "seed": seed,
            "architecture": arch,
            "status": "COMPLETED",
            "completed_epochs": epochs if not args.smoke_test else 1,
            "best_epoch": trainer.best_epoch,
            "best_score": float(trainer.checkpoint_cb.best_score),
            "completed_at": end_time_iso,
        }
        save_json(done_payload, seed_run_dir / f"DONE_seed_{seed}.json")
        save_json(done_payload, base_run_out_dir / f"DONE_seed_{seed}.json")
        logger.info("Wrote DONE marker for seed %d to %s", seed, seed_run_dir / f"DONE_seed_{seed}.json")

        # Write run manifest
        ckpt_dict = {
            "checkpoint_best.pt": seed_ckpt_dir / "checkpoint_best.pt",
            "checkpoint_latest.pt": seed_ckpt_dir / "checkpoint_latest.pt",
            "last.pt": seed_ckpt_dir / "last.pt",
        }
        write_run_manifest(
            output_path=seed_run_dir / f"run_manifest_{arch}_{seed}.json",
            seed=seed,
            epochs=epochs if not args.smoke_test else 1,
            arch=arch,
            config_path=config_path,
            splits_dir=splits_dir / ds_name if (splits_dir / ds_name).is_dir() else splits_dir,
            checkpoint_paths=ckpt_dict,
            start_time_iso=start_time_iso,
            end_time_iso=end_time_iso,
            best_epoch=trainer.best_epoch,
            best_val_pneumonia_auroc=float(trainer.checkpoint_cb.best_score),
            run_id=run_id,
        )
        write_run_manifest(
            output_path=base_run_out_dir / f"run_manifest_{arch}_{seed}.json",
            seed=seed,
            epochs=epochs if not args.smoke_test else 1,
            arch=arch,
            config_path=config_path,
            splits_dir=splits_dir / ds_name if (splits_dir / ds_name).is_dir() else splits_dir,
            checkpoint_paths=ckpt_dict,
            start_time_iso=start_time_iso,
            end_time_iso=end_time_iso,
            best_epoch=trainer.best_epoch,
            best_val_pneumonia_auroc=float(trainer.checkpoint_cb.best_score),
            run_id=run_id,
        )

    # Multi-seed aggregate summary if multiple seeds executed
    if len(seeds) > 1 and all_seed_metrics:
        summary_payload = {
            "architecture": arch,
            "seeds": seeds,
            "n_seeds": len(seeds),
            "runs": all_seed_metrics,
        }
        pneu_aurocs = [m["metrics"].get("val_pneumonia_auroc", 0.0) for m in all_seed_metrics if "val_pneumonia_auroc" in m["metrics"]]
        macro_aurocs = [m["metrics"].get("val_macro_auroc", 0.0) for m in all_seed_metrics if "val_macro_auroc" in m["metrics"]]
        if pneu_aurocs:
            summary_payload["pneumonia_auroc_mean"] = float(pd.Series(pneu_aurocs).mean())
            summary_payload["pneumonia_auroc_std"] = float(pd.Series(pneu_aurocs).std())
        if macro_aurocs:
            summary_payload["macro_auroc_mean"] = float(pd.Series(macro_aurocs).mean())
            summary_payload["macro_auroc_std"] = float(pd.Series(macro_aurocs).std())

        multi_summary_path = base_run_out_dir / f"multi_seed_summary_{arch}.json"
        save_json(summary_payload, multi_summary_path)
        logger.info("\n" + "=" * 70)
        logger.info("MULTI-SEED REPLICATION COMPLETE (%d seeds)", len(seeds))
        if "pneumonia_auroc_mean" in summary_payload:
            logger.info("  Pneumonia AUROC: %.4f +/- %.4f", summary_payload["pneumonia_auroc_mean"], summary_payload.get("pneumonia_auroc_std", 0.0))
        if "macro_auroc_mean" in summary_payload:
            logger.info("  Macro AUROC:     %.4f +/- %.4f", summary_payload["macro_auroc_mean"], summary_payload.get("macro_auroc_std", 0.0))
        logger.info("  Saved summary to %s", multi_summary_path)
        logger.info("=" * 70)


if __name__ == "__main__":
    main()
