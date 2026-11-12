"""Training CLI entry point for PneumoShift with multi-seed replication support."""

import argparse
import copy
from pathlib import Path
import shutil
from typing import Any, Dict, List
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.data.splits import create_synthetic_dataset
from pneumoshift.data.transforms import get_transforms
from pneumoshift.evaluation.benchmarks import compare_against_benchmarks
from pneumoshift.models.factory import build_model
from pneumoshift.training.losses import MaskedBCEWithLogitsLoss, compute_class_weights
from pneumoshift.training.trainer import Trainer
from pneumoshift.utils.io import load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger
from pneumoshift.utils.seed import set_seed

logger = get_logger("scripts.train")


def resolve_split_path(base_dir: Path, ds_name: str, split_name: str) -> Path:
    """Finds split CSV either directly in base_dir or in a dataset subfolder."""
    direct = base_dir / f"{ds_name}_{split_name}.csv"
    if direct.is_file():
        return direct
    sub = base_dir / ds_name / f"{ds_name}_{split_name}.csv"
    if sub.is_file():
        return sub
    return direct


def parse_args() -> argparse.Namespace:
    """Parses training command-line arguments."""
    parser = argparse.ArgumentParser(description="Train deep learning model for PneumoShift.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick synthetic smoke test")
    parser.add_argument("--seed", type=int, default=None, help="Single random seed override")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated random seeds (e.g. '42,1337,2024')")
    parser.add_argument("--architecture", default=None, help="Model architecture override")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs override")
    parser.add_argument("--eval_test", action="store_true", help="Evaluate best checkpoint on test set after training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config != "configs/default.yaml":
        config_path = args.config
    elif args.smoke_test:
        config_path = "configs/smoke_test.yaml"
    else:
        config_path = args.config

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

    base_run_out_dir = Path(cfg.get("train", {}).get("run_output_dir", "results/runs"))
    base_checkpoint_dir = Path(cfg.get("train", {}).get("checkpoint_dir", "results/runs/checkpoints"))
    base_run_out_dir.mkdir(parents=True, exist_ok=True)
    base_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Initializing multi-seed training suite:")
    logger.info("  Architecture: %s | Epochs: %d | Headline Only: %s", arch, epochs, headline_only)
    logger.info("  Seeds to execute: %s (%d seeds total)", seeds, len(seeds))

    all_seed_metrics: List[Dict[str, Any]] = []

    for seed_idx, seed in enumerate(seeds):
        logger.info("\n" + "=" * 70)
        logger.info(">>> STARTING SEED %d / %d (seed=%d) <<<", seed_idx + 1, len(seeds), seed)
        logger.info("=" * 70)

        set_seed(seed)
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

        if args.smoke_test or ds_name == "synthetic":
            create_synthetic_dataset(output_dir=synth_dir, num_samples=30, image_size=img_size, seed=seed)
            train_df = pd.read_csv(synth_dir / "splits" / "synthetic_train.csv")
            val_df = pd.read_csv(synth_dir / "splits" / "synthetic_val.csv")
            img_dir = synth_dir / "images"
        else:
            train_csv = resolve_split_path(splits_dir, ds_name, "train")
            val_csv = resolve_split_path(splits_dir, ds_name, "val")

            if not train_csv.is_file() or not val_csv.is_file():
                logger.warning("Split files not found (%s, %s). Falling back to synthetic smoke data.", train_csv, val_csv)
                create_synthetic_dataset(output_dir=synth_dir, num_samples=30, image_size=img_size, seed=seed)
                train_df = pd.read_csv(synth_dir / "splits" / "synthetic_train.csv")
                val_df = pd.read_csv(synth_dir / "splits" / "synthetic_val.csv")
                img_dir = synth_dir / "images"
            else:
                train_df = pd.read_csv(train_csv)
                val_df = pd.read_csv(val_csv)
                img_dir = Path(cfg.get("data", {}).get("images_dir", cfg.get("data", {}).get("data_dir", "")))

        train_trans = get_transforms(split="train", image_size=img_size)
        val_trans = get_transforms(split="val", image_size=img_size)

        label_cols = [HEADLINE_CLASS] if headline_only else SHARED_CLASSES
        train_ds = ChestXrayDataset(train_df, images_dir=img_dir, transform=train_trans, headline_only=headline_only, cache_dir=cache_dir)
        val_ds = ChestXrayDataset(val_df, images_dir=img_dir, transform=val_trans, headline_only=headline_only, cache_dir=cache_dir)

        g = torch.Generator()
        g.manual_seed(seed)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
            backbone_lr = float(seed_cfg.get("train", {}).get("backbone_lr", lr * 0.1))
            param_groups = [
                {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
                {"params": list(model.head.parameters()), "lr": lr, "name": "head"},
            ]
        else:
            param_groups = [{"params": model.parameters(), "lr": lr}]
            backbone_lr = lr

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

        history = trainer.fit(epochs=epochs)

        # Save history to both per-seed and base run directory
        save_json(history, seed_run_dir / f"history_{arch}_{seed}.json")
        save_json(history, base_run_out_dir / f"history_{arch}_{seed}.json")
        logger.info("Seed %d training complete. Saved history to %s", seed, seed_run_dir)

        # Mirror first seed or best checkpoint into base_checkpoint_dir for backwards compatibility
        best_ckpt_path = trainer.checkpoint_cb.checkpoint_dir / "checkpoint_best.pt"
        if best_ckpt_path.is_file() and len(seeds) > 1 and seed_idx == 0:
            shutil.copy2(best_ckpt_path, base_checkpoint_dir / "checkpoint_best.pt")
            sec_ckpt = trainer.checkpoint_cb.checkpoint_dir / "checkpoint_best_pneumonia.pt"
            if sec_ckpt.is_file():
                shutil.copy2(sec_ckpt, base_checkpoint_dir / "checkpoint_best_pneumonia.pt")
            latest_ckpt = trainer.checkpoint_cb.checkpoint_dir / "checkpoint_latest.pt"
            if latest_ckpt.is_file():
                shutil.copy2(latest_ckpt, base_checkpoint_dir / "checkpoint_latest.pt")

        if args.eval_test:
            logger.info("Evaluating seed %d best model checkpoint on test set...", seed)
            if best_ckpt_path.is_file():
                ckpt = torch.load(best_ckpt_path, map_location=trainer.device)
                model.load_state_dict(ckpt["model_state_dict"])
                logger.info("Loaded best model checkpoint from %s", best_ckpt_path)

            if args.smoke_test or seed_cfg.get("data", {}).get("dataset_name") == "synthetic":
                test_csv = synth_dir / "splits" / "synthetic_test.csv"
            else:
                test_csv = resolve_split_path(splits_dir, ds_name, "test")
                if not test_csv.is_file():
                    test_csv = synth_dir / "splits" / "synthetic_test.csv"

            if test_csv.is_file():
                test_df = pd.read_csv(test_csv)
                test_trans = get_transforms(split="test", image_size=img_size)
                test_ds = ChestXrayDataset(test_df, images_dir=img_dir, transform=test_trans, headline_only=headline_only, cache_dir=cache_dir)
                test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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

                # CheXpert official validation benchmark
                if "chexpert" in ds_name:
                    off_val_csv = resolve_split_path(splits_dir, ds_name, "official_val")
                    if not off_val_csv.is_file() and args.smoke_test:
                        off_val_csv = synth_dir / "splits" / "synthetic_official_val.csv"

                    if off_val_csv.is_file():
                        logger.info("Evaluating on official CheXpert validation benchmark: %s", off_val_csv)
                        off_df = pd.read_csv(off_val_csv)
                        off_ds = ChestXrayDataset(off_df, images_dir=img_dir, transform=test_trans, headline_only=headline_only, cache_dir=cache_dir)
                        off_loader = DataLoader(off_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

                        for mode_idx, mode in enumerate(eval_modes):
                            off_metrics = trainer.evaluate(off_loader, uncertainty_mode=mode)
                            save_json(off_metrics, seed_run_dir / f"official_val_metrics_{arch}_{seed}_{mode}.json")
                            save_json(off_metrics, base_run_out_dir / f"official_val_metrics_{arch}_{seed}_{mode}.json")
                            if mode_idx == 0:
                                save_json(off_metrics, seed_run_dir / f"official_val_metrics_{arch}_{seed}.json")
                                save_json(off_metrics, base_run_out_dir / f"official_val_metrics_{arch}_{seed}.json")
                            logger.info("Official Val (seed=%d, mode=%s): Macro AUROC: %.4f | Pneumonia AUROC: %.4f",
                                        seed, mode, off_metrics.get("val_macro_auroc", 0.0), off_metrics.get("val_pneumonia_auroc", 0.0))
            else:
                logger.warning("Test split CSV %s not found. Skipping test evaluation for seed %d.", test_csv, seed)

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
