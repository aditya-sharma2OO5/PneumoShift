#!/usr/bin/env python3
"""Verifies that saved NIH prediction logits match classifier head linear projection (logits = features @ W.T + b).

For each of the 6 (architecture, seed) models across validation and test splits (12 files total):
1. Loads the model classifier head weights (W) and bias (b) from the verified checkpoint.
2. Loads saved features and logits from the prediction .npz file.
3. Recomputes logits = features @ W.T + b.
4. Asserts that the max absolute difference is <= 1e-4.
"""

import sys
from pathlib import Path
import numpy as np
import torch

CHECKPOINTS = {
    ("densenet121", 42): "results/runs/nih_densenet121/checkpoints/seed_42/checkpoint_best.pt",
    ("densenet121", 1337): "results/runs/nih_densenet121/checkpoints/seed_1337/checkpoint_best.pt",
    ("densenet121", 2024): "results/runs/nih_densenet121/checkpoints/seed_2024/checkpoint_best.pt",
    ("efficientnet_b0", 42): "results/runs/nih_efficientnet_b0/checkpoints/seed_42/checkpoint_best.pt",
    ("efficientnet_b0", 1337): "results/runs/nih_efficientnet_b0/checkpoints/seed_1337/checkpoint_best.pt",
    ("efficientnet_b0", 2024): "results/runs/nih_efficientnet_b0/checkpoints/seed_2024/checkpoint_best.pt",
}

TOLERANCE = 1e-4


def main():
    print("=" * 80)
    print("VERIFYING NIH PREDICTIONS AGAINST CHECKPOINT CLASSIFIER HEADS")
    print(f"Mathematical invariant: logits == features @ W.T + b (tolerance = {TOLERANCE})")
    print("=" * 80)

    all_passed = True
    total_checked = 0

    for (arch, seed), ckpt_path_str in CHECKPOINTS.items():
        ckpt_path = Path(ckpt_path_str)
        if not ckpt_path.is_file():
            print(f"ERROR: Checkpoint not found: {ckpt_path}")
            all_passed = False
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        if "head.linear.weight" not in state_dict or "head.linear.bias" not in state_dict:
            print(f"ERROR: Classifier head keys missing in {ckpt_path}")
            all_passed = False
            continue

        W = state_dict["head.linear.weight"].numpy()
        b = state_dict["head.linear.bias"].numpy()

        for split in ["val", "test"]:
            pred_path = Path(f"results/predictions/predictions_{split}_{arch}_{seed}.npz")
            if not pred_path.is_file():
                print(f"ERROR: Prediction file not found: {pred_path}")
                all_passed = False
                continue

            data = np.load(pred_path, allow_pickle=True)
            if "features" not in data or "y_logits" not in data:
                print(f"ERROR: 'features' or 'y_logits' missing in {pred_path}")
                all_passed = False
                continue

            feats = data["features"]
            logits = data["y_logits"]

            # Mathematical forward projection
            logits_recomputed = feats @ W.T + b
            max_abs_diff = float(np.max(np.abs(logits - logits_recomputed)))
            mean_abs_diff = float(np.mean(np.abs(logits - logits_recomputed)))

            passed = (max_abs_diff <= TOLERANCE)
            if not passed:
                all_passed = False

            status_str = "PASS" if passed else "FAIL"
            print(f"[{status_str:4s}] Split: {split.upper():4s} | Model: {arch:15s} | Seed: {seed:4d} | "
                  f"MaxDiff: {max_abs_diff:12.8e} | MeanDiff: {mean_abs_diff:12.8e}")
            total_checked += 1

    print("-" * 80)
    print(f"Total files checked: {total_checked}/12")
    if all_passed and total_checked == 12:
        print("VERDICT: ALL 12 NIH PREDICTION ARRAYS MATHEMATICALLY VERIFIED AGAINST CHECKPOINTS.")
        return 0
    else:
        print("VERDICT: VERIFICATION FAILED.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
