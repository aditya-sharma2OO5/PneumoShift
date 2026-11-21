"""Generates checkpoint_manifest.json for the 6 verified NIH model checkpoints."""

import hashlib
import json
from pathlib import Path
import torch
import pandas as pd


CHECKPOINTS = [
    {
        "architecture": "densenet121",
        "seed": 42,
        "checkpoint_path": "results/runs/nih_densenet121/checkpoints/seed_42/checkpoint_best.pt",
        "epoch_metrics_path": "results/runs/nih_densenet121/seed_42/epoch_metrics.csv",
    },
    {
        "architecture": "densenet121",
        "seed": 1337,
        "checkpoint_path": "results/runs/nih_densenet121/checkpoints/seed_1337/checkpoint_best.pt",
        "epoch_metrics_path": "results/runs/nih_densenet121/seed_1337/epoch_metrics.csv",
    },
    {
        "architecture": "densenet121",
        "seed": 2024,
        "checkpoint_path": "results/runs/nih_densenet121/checkpoints/seed_2024/checkpoint_best.pt",
        "epoch_metrics_path": "results/runs/nih_densenet121/seed_2024/epoch_metrics.csv",
    },
    {
        "architecture": "efficientnet_b0",
        "seed": 42,
        "checkpoint_path": "results/runs/nih_efficientnet_b0/checkpoints/seed_42/checkpoint_best.pt",
        "epoch_metrics_path": "results/runs/nih_efficientnet_b0/seed_42/epoch_metrics.csv",
    },
    {
        "architecture": "efficientnet_b0",
        "seed": 1337,
        "checkpoint_path": "results/runs/nih_efficientnet_b0/checkpoints/seed_1337/checkpoint_best.pt",
        "epoch_metrics_path": "results/runs/nih_efficientnet_b0/seed_1337/epoch_metrics.csv",
    },
    {
        "architecture": "efficientnet_b0",
        "seed": 2024,
        "checkpoint_path": "results/runs/nih_efficientnet_b0/checkpoints/seed_2024/checkpoint_best.pt",
        "epoch_metrics_path": "results/runs/nih_efficientnet_b0/seed_2024/epoch_metrics.csv",
    },
]


def create_manifest() -> dict:
    manifest = {
        "version": "1.0.0",
        "dataset_origin": "nih_chest_xray",
        "checkpoints": {},
        "valid_sha256s": [],
    }

    for item in CHECKPOINTS:
        p = Path(item["checkpoint_path"])
        if not p.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {p}")

        data = p.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        size_bytes = len(data)

        ckpt_obj = torch.load(p, map_location="cpu", weights_only=False)
        stored_epoch = int(ckpt_obj.get("epoch", 1))

        # Lookup val AUROC from epoch_metrics.csv at stored_epoch
        metrics_df = pd.read_csv(item["epoch_metrics_path"])
        epoch_row = metrics_df[metrics_df["epoch"] == stored_epoch]
        if not epoch_row.empty:
            val_auroc = float(epoch_row["val_pneumonia_auroc"].iloc[0])
            val_macro_auroc = float(epoch_row["val_macro_auroc"].iloc[0])
        else:
            val_auroc = None
            val_macro_auroc = None

        key = f"{item['architecture']}_{item['seed']}"
        entry = {
            "architecture": item["architecture"],
            "seed": item["seed"],
            "relative_path": str(p).replace("\\", "/"),
            "filename": p.name,
            "sha256": sha256,
            "size_bytes": size_bytes,
            "stored_epoch": stored_epoch,
            "stored_val_pneumonia_auroc": val_auroc,
            "stored_val_macro_auroc": val_macro_auroc,
        }
        manifest["checkpoints"][key] = entry
        manifest["valid_sha256s"].append(sha256)

    return manifest


def main():
    manifest = create_manifest()
    
    out_paths = [
        Path("checkpoint_manifest.json"),
        Path("configs/checkpoint_manifest.json"),
        Path("results/checkpoints/checkpoint_manifest.json"),
    ]
    for outp in out_paths:
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved manifest to {outp}")

    print("\nManifest Summary:")
    for k, v in manifest["checkpoints"].items():
        print(f"  {k:25s} Epoch {v['stored_epoch']} | Val AUC {v['stored_val_pneumonia_auroc']:.6f} | SHA256: {v['sha256']}")


if __name__ == "__main__":
    main()
