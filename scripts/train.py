"""Training CLI entry point for PneumoShift."""

import argparse
from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.splits import create_synthetic_dataset
from pneumoshift.data.transforms import get_transforms
from pneumoshift.models.factory import build_model
from pneumoshift.training.losses import MaskedBCEWithLogitsLoss, compute_class_weights
from pneumoshift.training.trainer import Trainer
from pneumoshift.utils.io import load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger
from pneumoshift.utils.seed import set_seed

logger = get_logger("scripts.train")


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

    logger.info("Initializing run with arch=%s, seed=%d, epochs=%d", arch, seed, epochs)

    if args.smoke_test or cfg.get("data", {}).get("dataset_name") == "synthetic":
        synth_dir = Path("data/synthetic")
        create_synthetic_dataset(output_dir=synth_dir, num_samples=30, image_size=img_size, seed=seed)
        train_df = pd.read_csv(synth_dir / "splits" / "synthetic_train.csv")
        val_df = pd.read_csv(synth_dir / "splits" / "synthetic_val.csv")
        img_dir = synth_dir / "images"
    else:
        splits_dir = Path(cfg.get("data", {}).get("splits_dir", "data/splits"))
        ds_name = cfg.get("data", {}).get("dataset_name", "nih")
        train_df = pd.read_csv(splits_dir / f"{ds_name}_train.csv")
        val_df = pd.read_csv(splits_dir / f"{ds_name}_val.csv")
        img_dir = Path(cfg.get("data", {}).get("data_dir"))

    train_trans = get_transforms(split="train", image_size=img_size)
    val_trans = get_transforms(split="val", image_size=img_size)

    train_ds = ChestXrayDataset(train_df, images_dir=img_dir, transform=train_trans)
    val_ds = ChestXrayDataset(val_df, images_dir=img_dir, transform=val_trans)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Class weighting
    pos_weight = compute_class_weights(train_df) if cfg.get("train", {}).get("pos_weight") == "inverse_freq" else None
    criterion = MaskedBCEWithLogitsLoss(pos_weight=pos_weight)

    model = build_model(
        architecture=arch,
        num_classes=cfg.get("model", {}).get("num_classes", 8),
        pretrained=cfg.get("model", {}).get("pretrained", False if args.smoke_test else True),
        dropout=cfg.get("model", {}).get("dropout", 0.2),
    )

    lr = float(cfg.get("train", {}).get("learning_rate", 1e-4))
    wd = float(cfg.get("train", {}).get("weight_decay", 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=cfg,
    )

    history = trainer.fit(epochs=epochs)
    out_dir = Path("results/runs")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(history, out_dir / f"history_{arch}_{seed}.json")
    logger.info("Training complete. Results saved.")


if __name__ == "__main__":
    main()

