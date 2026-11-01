"""Cross-dataset evaluation, calibration, and threshold transfer script."""

import argparse
from pathlib import Path
from typing import Dict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pneumoshift.data.datasets import ChestXrayDataset
from pneumoshift.data.transforms import get_transforms
from pneumoshift.evaluation.calibration import TemperatureScaling, compute_calibration_metrics
from pneumoshift.evaluation.domain_gap import compute_feature_mmd, train_domain_classifier_probe
from pneumoshift.evaluation.metrics import compute_discrimination_metrics, compute_generalization_gap
from pneumoshift.evaluation.thresholds import (
    compute_optimal_thresholds,
    compute_score_distribution_shift,
    evaluate_threshold_transfer,
)
from pneumoshift.models.factory import build_model
from pneumoshift.utils.io import load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.evaluate")


def extract_features_and_predictions(model, loader, device, headline_idx: int = 5):
    model.eval()
    all_logits = []
    all_targets = []
    all_features = []

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
    probs = 1.0 / (1.0 + np.exp(-logits))

    return logits, probs, targets, features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PneumoShift model across in-domain and target cohorts.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="results/runs/checkpoints/checkpoint_best.pt")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", default="results/tables")
    args = parser.parse_args()

    cfg = load_yaml_config("configs/smoke_test.yaml" if args.smoke_test else args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = cfg.get("model", {}).get("architecture", "densenet121")
    img_size = cfg.get("data", {}).get("image_size", 224)

    model = build_model(architecture=arch, num_classes=8, pretrained=False)
    if Path(args.checkpoint).is_file():
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded weights from %s", args.checkpoint)
    else:
        logger.warning("Checkpoint %s not found, running with initialized weights.", args.checkpoint)

    model.to(device)
    logger.info("Evaluation framework initialized successfully.")


if __name__ == "__main__":
    main()

