"""Training CLI entry point for PneumoShift."""

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.data.splits import create_synthetic_dataset
from pneumoshift.data.transforms import get_transforms
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep learning model for PneumoShift.")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--smoke_test", action="store_true", help="Run quick synthetic smoke test")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--architecture", default=None, help="Model architecture override")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs override")
    args = parser.parse_args()

    config_path = "configs/smoke_test.yaml" if args.smoke_test else args.config
    cfg = load_yaml_config(config_path)

    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    set_seed(seed)

    arch = args.architecture or cfg.get("model", {}).get("architecture", "densenet121")
    epochs = args.epochs or cfg.get("train", {}).get("epochs", 20)
    batch_size = cfg.get("data", {}).get("batch_size", 32)
    num_workers = cfg.get("data", {}).get("num_workers", 0)
    img_size = cfg.get("data", {}).get("image_size", 224)
    headline_only = cfg.get("data", {}).get("headline_only", False)
    cache_preprocessed = cfg.get("data", {}).get("cache_preprocessed", False)
    cache_dir = cfg.get("data", {}).get("cache_dir", None) if cache_preprocessed else None

    logger.info("Initializing run with arch=%s, seed=%d, epochs=%d, headline_only=%s", arch, seed, epochs, headline_only)

    if args.smoke_test or cfg.get("data", {}).get("dataset_name") == "synthetic":
        synth_dir = Path("data/synthetic")
        create_synthetic_dataset(output_dir=synth_dir, num_samples=30, image_size=img_size, seed=seed)
        train_df = pd.read_csv(synth_dir / "splits" / "synthetic_train.csv")
        val_df = pd.read_csv(synth_dir / "splits" / "synthetic_val.csv")
        img_dir = synth_dir / "images"
    else:
        splits_dir = Path(cfg.get("data", {}).get("splits_dir", "data/splits"))
        ds_name = cfg.get("data", {}).get("dataset_name", "nih")
        train_csv = resolve_split_path(splits_dir, ds_name, "train")
        val_csv = resolve_split_path(splits_dir, ds_name, "val")

        if not train_csv.is_file() or not val_csv.is_file():
            logger.warning("Split files not found (%s, %s). Falling back to synthetic smoke data.", train_csv, val_csv)
            synth_dir = Path("data/synthetic")
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Inverse frequency class weighting
    pos_weight = compute_class_weights(train_df, label_cols=label_cols) if cfg.get("train", {}).get("pos_weight") == "inverse_freq" else None
    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)

    num_classes = 1 if headline_only else cfg.get("model", {}).get("num_classes", 8)
    model = build_model(
        architecture=arch,
        num_classes=num_classes,
        pretrained=cfg.get("model", {}).get("pretrained", False if args.smoke_test else True),
        dropout=cfg.get("model", {}).get("dropout", 0.2),
    )

    lr = float(cfg.get("train", {}).get("learning_rate", 1e-4))
    wd = float(cfg.get("train", {}).get("weight_decay", 1e-5))
    opt_type = cfg.get("train", {}).get("optimizer", "adamw").lower()
    two_stage = cfg.get("train", {}).get("two_stage_finetune", False)

    # When two-stage fine-tuning is enabled, register backbone and head as
    # separate parameter groups so the optimizer already tracks all params.
    # Backbone LR starts at 0 (frozen phase); trainer.py updates it on unfreeze.
    if two_stage:
        backbone_lr = float(cfg.get("train", {}).get("backbone_lr", lr * 0.1))
        param_groups = [
            {"params": list(model.backbone.parameters()), "lr": 0.0, "name": "backbone"},
            {"params": list(model.head.parameters()), "lr": lr, "name": "head"},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]
        backbone_lr = lr  # unused, kept for clarity

    if opt_type == "adam":
        optimizer = torch.optim.Adam(param_groups, lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)

    patience = cfg.get("train", {}).get("lr_patience", 2)
    factor = cfg.get("train", {}).get("lr_factor", 0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=factor, patience=patience)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=cfg,
        label_cols=label_cols,
    )

    history = trainer.fit(epochs=epochs)
    run_out_dir = Path(cfg.get("train", {}).get("run_output_dir", "results/runs"))
    run_out_dir.mkdir(parents=True, exist_ok=True)
    save_json(history, run_out_dir / f"history_{arch}_{seed}.json")
    logger.info("Training complete. Results saved to %s", run_out_dir)


if __name__ == "__main__":
    main()
