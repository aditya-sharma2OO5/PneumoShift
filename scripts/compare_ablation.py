"""CLI script to compare multi-task vs single-task ablation models for pneumonia detection."""

import argparse
from pathlib import Path
from typing import Any, Dict
import numpy as np

from pneumoshift.evaluation.ablation import compare_single_vs_multitask
from pneumoshift.utils.io import load_json, save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.compare_ablation")


def _find_metrics_file(path_str: str) -> Path:
    """Finds evaluation metrics JSON whether pointing to a file or directory."""
    p = Path(path_str)
    if p.is_file():
        return p
    # Try common filename patterns
    patterns = [
        "eval_metrics_test.json",
        "eval_metrics_test_uncertain_as_negative.json",
        "test_metrics_densenet121_42.json",
        "epoch_metrics.json",
    ]
    for pat in patterns:
        candidate = p / pat
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not find metrics JSON in {path_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare multi-task vs single-task ablation models.")
    parser.add_argument("--multitask", default="results/tables/nih_densenet121", help="Multi-task results JSON or dir")
    parser.add_argument("--singletask", default="results/tables/pneumonia_only_nih", help="Single-task results JSON or dir")
    parser.add_argument("--cohort", default=None, help="Cohort name override (e.g. 'NIH ChestX-ray14' or 'Stanford CheXpert')")
    parser.add_argument("--output_dir", default="results/tables/ablation", help="Directory to save comparison outputs")
    parser.add_argument("--smoke_test", action="store_true", help="Run with synthetic test numbers for verification")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cohort = args.cohort
    if cohort is None:
        if "chexpert" in args.multitask.lower() or "chexpert" in args.singletask.lower():
            cohort = "Stanford CheXpert (U-Zeros)"
        else:
            cohort = "NIH ChestX-ray14"

    if args.smoke_test:
        logger.info("Running synthetic ablation comparison smoke test...")
        multi_metrics = {
            "val_pneumonia_auroc": 0.732,
            "val_pneumonia_auprc": 0.385,
            "ece": 0.042,
        }
        single_metrics = {
            "val_pneumonia_auroc": 0.708,
            "val_pneumonia_auprc": 0.351,
            "ece": 0.058,
        }
        # Simulate paired predictions on 50 samples
        rng = np.random.RandomState(42)
        y_true = (rng.rand(50) < 0.15).astype(float)
        y_pred_multi = np.clip(y_true * 0.6 + rng.rand(50) * 0.4, 0.0, 1.0)
        y_pred_single = np.clip(y_true * 0.5 + rng.rand(50) * 0.5, 0.0, 1.0)
        patient_ids = np.array([f"patient_{i // 2:02d}" for i in range(50)])
    else:
        multi_file = _find_metrics_file(args.multitask)
        single_file = _find_metrics_file(args.singletask)
        logger.info("Comparing Multi-Task (%s) vs Single-Task (%s)", multi_file, single_file)
        multi_metrics = load_json(multi_file)
        single_metrics = load_json(single_file)
        y_true = None
        y_pred_multi = None
        y_pred_single = None
        patient_ids = None

    comparison = compare_single_vs_multitask(
        multitask_results=multi_metrics,
        singletask_results=single_metrics,
        cohort_name=cohort,
        patient_ids=patient_ids,
        y_true=y_true,
        y_pred_multi=y_pred_multi,
        y_pred_single=y_pred_single,
    )

    # Save outputs
    save_json(comparison, out_dir / "ablation_comparison.json")
    with open(out_dir / "ablation_comparison.md", "w", encoding="utf-8") as f:
        f.write(comparison["table_markdown"] + "\n")

    print("\n" + "=" * 70)
    print("ABLATION ANALYSIS: SINGLE-TASK VS MULTI-TASK")
    print("=" * 70)
    print(comparison["table_markdown"])
    print("=" * 70 + "\n")
    logger.info("Saved ablation comparison artifacts to %s", out_dir)


if __name__ == "__main__":
    main()

