"""Explainability runner: Grad-CAM, faithfulness curves, stability, and sanity checks."""

import argparse
from pathlib import Path
import numpy as np
import torch
from pneumoshift.models.factory import build_model, get_target_layer_for_xai
from pneumoshift.utils.io import load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger
from pneumoshift.xai.faithfulness import compute_deletion_insertion_curves
from pneumoshift.xai.gradcam import GradCAMExplainer
from pneumoshift.xai.lung_mask import compute_in_lung_attribution_ratio
from pneumoshift.xai.sanity_checks import run_parameter_randomization_test
from pneumoshift.xai.shortcut_analysis import compute_shortcut_energy_ratio
from pneumoshift.xai.stability import compute_attribution_stability

logger = get_logger("scripts.run_xai")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explainability suite for PneumoShift.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default="results/runs/checkpoints/checkpoint_best.pt")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--output_dir", default="results/figures/xai")
    args = parser.parse_args()

    cfg = load_yaml_config("configs/smoke_test.yaml" if args.smoke_test else args.config)
    arch = cfg.get("model", {}).get("architecture", "densenet121")
    img_size = cfg.get("data", {}).get("image_size", 224)

    model = build_model(architecture=arch, num_classes=8, pretrained=False)
    model.eval()

    target_layer = get_target_layer_for_xai(model, arch)
    explainer = GradCAMExplainer(model, target_layer)

    # Synthetic mock image for pipeline verification
    dummy_input = torch.randn(1, 3, img_size, img_size)
    cam = explainer.generate_cam(dummy_input, target_class_idx=5)

    # Faithfulness
    faith_results = compute_deletion_insertion_curves(model, dummy_input, cam, target_class_idx=5, num_steps=5)
    # Sanity checks
    sanity_results = run_parameter_randomization_test(model, arch, dummy_input, cam, target_class_idx=5)
    # Stability
    stability_results = compute_attribution_stability(explainer, dummy_input, target_class_idx=5, num_perturbations=3)
    # Plausibility & shortcuts
    plausibility = compute_in_lung_attribution_ratio(cam)
    shortcuts = compute_shortcut_energy_ratio(cam)

    explainer.release_hooks()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    results = {
        "faithfulness": faith_results,
        "sanity_checks": sanity_results,
        "stability": stability_results,
        "anatomical_plausibility": plausibility,
        "shortcut_analysis": shortcuts,
    }
    save_json(results, out_path / "xai_summary.json")
    logger.info("XAI suite completed successfully. Results saved to %s", out_path)


if __name__ == "__main__":
    main()

