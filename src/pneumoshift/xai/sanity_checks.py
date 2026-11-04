"""Model parameter randomization sanity checks for saliency maps (Adebayo et al., NeurIPS 2018).

Evaluates whether Grad-CAM explanations are sensitive to trained model parameters
rather than acting as data-independent edge detectors or image priors.
Supports classifier-only, cascading layer-wise, and full-network randomization.
"""

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats
import torch
import torch.nn as nn

from pneumoshift.models.factory import PneumoShiftModel, get_target_layer_for_xai
from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger
from pneumoshift.xai.gradcam import GradCAMExplainer

logger = get_logger("pneumoshift.xai.sanity_checks")


def compute_map_similarity(map_a: np.ndarray, map_b: np.ndarray) -> Dict[str, float]:
    """Computes correlation and distance similarity metrics between two attribution heatmaps.

    Handles zero-variance constant maps gracefully without emitting numerical warnings.

    Args:
        map_a: First 2D attribution heatmap of shape (H, W).
        map_b: Second 2D attribution heatmap of shape (H, W).

    Returns:
        Dict with spearman_rank_correlation, pearson_correlation, mean_squared_error,
        and cosine_similarity.
    """
    flat_a = np.array(map_a, dtype=np.float64).ravel()
    flat_b = np.array(map_b, dtype=np.float64).ravel()

    std_a = float(np.std(flat_a))
    std_b = float(np.std(flat_b))

    # Constant / zero-variance handling
    if std_a < 1e-8 and std_b < 1e-8:
        spearman_corr = 1.0
        pearson_corr = 1.0
    elif std_a < 1e-8 or std_b < 1e-8:
        spearman_corr = 0.0
        pearson_corr = 0.0
    else:
        s_corr, _ = scipy.stats.spearmanr(flat_a, flat_b)
        p_corr, _ = scipy.stats.pearsonr(flat_a, flat_b)
        spearman_corr = float(s_corr) if not np.isnan(s_corr) else 0.0
        pearson_corr = float(p_corr) if not np.isnan(p_corr) else 0.0

    # Mean Squared Error
    mse = float(np.mean((flat_a - flat_b) ** 2))

    # Cosine Similarity
    norm_a = float(np.linalg.norm(flat_a))
    norm_b = float(np.linalg.norm(flat_b))
    if norm_a > 1e-8 and norm_b > 1e-8:
        cosine_sim = float(np.dot(flat_a, flat_b) / (norm_a * norm_b))
    else:
        cosine_sim = 1.0 if (norm_a < 1e-8 and norm_b < 1e-8) else 0.0

    return {
        "spearman_rank_correlation": spearman_corr,
        "pearson_correlation": pearson_corr,
        "mean_squared_error": mse,
        "cosine_similarity": cosine_sim,
    }


def _reinit_submodules(module: nn.Module) -> None:
    """Recursively resets weights of linear, conv, and norm submodules."""
    for layer in module.modules():
        if hasattr(layer, "reset_parameters") and callable(layer.reset_parameters):
            layer.reset_parameters()


def randomize_model_weights(
    model: nn.Module,
    architecture: str,
    randomize_mode: str = "all",
    seed: Optional[int] = None,
) -> nn.Module:
    """Clones and re-initializes model parameters according to specified Adebayo scope.

    Args:
        model: Trained PyTorch model.
        architecture: Architecture string ('densenet121', 'efficientnet_b0', etc.).
        randomize_mode: 'classifier', 'cascading', or 'all'.
        seed: Optional random seed for reproducible re-initialization.

    Returns:
        Cloned model with specified parameters randomized.
    """
    random_model = copy.deepcopy(model)
    random_model.eval()

    if seed is not None:
        torch.manual_seed(seed)

    arch = architecture.lower()
    mode = str(randomize_mode).lower().strip()

    if mode == "classifier":
        # Re-initialize only the classification head
        if hasattr(random_model, "head"):
            _reinit_submodules(random_model.head)
        elif hasattr(random_model, "classifier"):
            _reinit_submodules(random_model.classifier)
        else:
            # Fallback to last linear layer
            for m in reversed(list(random_model.modules())):
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                    break

    elif mode == "cascading":
        # Re-initialize classification head AND top feature stage
        if hasattr(random_model, "head"):
            _reinit_submodules(random_model.head)

        backbone = getattr(random_model, "backbone", getattr(random_model, "features", random_model))
        if "densenet" in arch:
            # Re-initialize norm5, denseblock4, and transition3
            for attr in ["norm5", "denseblock4", "transition3"]:
                if hasattr(backbone, attr):
                    _reinit_submodules(getattr(backbone, attr))
        elif "efficientnet" in arch:
            # Re-initialize top two stages (features.8 and features.7)
            if hasattr(backbone, "__getitem__"):
                _reinit_submodules(backbone[-1])
                if len(backbone) > 1:
                    _reinit_submodules(backbone[-2])
            else:
                _reinit_submodules(backbone)
        else:
            # General fallback: reinit last child of backbone
            children = list(backbone.children())
            if children:
                _reinit_submodules(children[-1])

    elif mode == "all":
        # Full network re-initialization
        _reinit_submodules(random_model)

    else:
        raise ValueError(f"Unsupported randomize_mode '{randomize_mode}'. Expected 'classifier', 'cascading', or 'all'")

    return random_model


def run_parameter_randomization_test(
    model: nn.Module,
    architecture: str,
    image_tensor: torch.Tensor,
    original_cam: np.ndarray,
    target_class_idx: int = 5,
    randomize_mode: str = "all",
    threshold_drop_percent: float = 75.0,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Performs Adebayo parameter randomization sanity check on a single radiograph.

    Args:
        model: Original trained model.
        architecture: Architecture string ('densenet121' or 'efficientnet_b0').
        image_tensor: Image tensor of shape (1, C, H, W) or (C, H, W).
        original_cam: Heatmap np.ndarray from unrandomized trained model.
        target_class_idx: Target disease class index.
        randomize_mode: 'classifier', 'cascading', or 'all'.
        threshold_drop_percent: Minimum required relative drop in correlation to pass sanity check.
        seed: Optional random seed for deterministic re-initialization.

    Returns:
        Dict with similarity metrics, pass/fail status, and mode details.
    """
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    # Re-initialize weights
    random_model = randomize_model_weights(
        model=model,
        architecture=architecture,
        randomize_mode=randomize_mode,
        seed=seed,
    )

    target_layer = get_target_layer_for_xai(random_model, architecture)
    with GradCAMExplainer(random_model, target_layer) as explainer:
        random_cam = explainer.generate_cam(image_tensor, target_class_idx=target_class_idx)

    sim = compute_map_similarity(original_cam, random_cam)
    rho = sim["spearman_rank_correlation"]

    # Adebayo Sanity Rule: Randomization must collapse attribution similarity (rho <= 0.25 or drop >= 75%)
    max_allowed_rho = 1.0 - (threshold_drop_percent / 100.0)
    passed = bool(rho <= max_allowed_rho)

    sim["randomize_mode"] = randomize_mode
    sim["sanity_passed"] = passed
    sim["threshold_drop_percent"] = threshold_drop_percent
    sim["architecture"] = architecture
    return sim


def evaluate_cohort_sanity_checks(
    model: nn.Module,
    architecture: str,
    images: Union[List[torch.Tensor], torch.Tensor],
    cams: Union[List[np.ndarray], np.ndarray],
    target_class_idx: int = 5,
    randomize_modes: Optional[List[str]] = None,
    threshold_drop_percent: float = 75.0,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluates Adebayo parameter randomization sanity checks across a cohort of radiographs.

    Args:
        model: Evaluated PyTorch model.
        architecture: Architecture string.
        images: Batch tensor or list of (1, C, H, W) tensors.
        cams: Batch array or list of 2D heatmaps.
        target_class_idx: Target disease class index.
        randomize_modes: List of modes to evaluate (default: ['classifier', 'cascading', 'all']).
        threshold_drop_percent: Minimum percentage correlation drop to pass.
        seed: Random seed for re-initialization.

    Returns:
        Dict mapping mode -> aggregated cohort statistics (mean rho, mean r, pass_rate, verdict).
    """
    if randomize_modes is None:
        randomize_modes = ["classifier", "cascading", "all"]

    n_samples = len(images)
    mode_results: Dict[str, Any] = {}

    for mode in randomize_modes:
        spearman_scores: List[float] = []
        pearson_scores: List[float] = []
        cosine_scores: List[float] = []
        passes: List[bool] = []

        for i in range(n_samples):
            img_i = images[i]
            cam_i = cams[i]
            res = run_parameter_randomization_test(
                model=model,
                architecture=architecture,
                image_tensor=img_i,
                original_cam=cam_i,
                target_class_idx=target_class_idx,
                randomize_mode=mode,
                threshold_drop_percent=threshold_drop_percent,
                seed=seed + i,
            )
            spearman_scores.append(res["spearman_rank_correlation"])
            pearson_scores.append(res["pearson_correlation"])
            cosine_scores.append(res["cosine_similarity"])
            passes.append(res["sanity_passed"])

        pass_rate = float(np.mean(passes) * 100.0)
        mean_spearman = float(np.mean(spearman_scores))
        verdict = "PASSED" if (pass_rate >= 80.0 or mean_spearman <= 0.25) else "FAILED"

        mode_results[mode] = {
            "num_samples": n_samples,
            "mean_spearman_correlation": mean_spearman,
            "std_spearman_correlation": float(np.std(spearman_scores)),
            "mean_pearson_correlation": float(np.mean(pearson_scores)),
            "mean_cosine_similarity": float(np.mean(cosine_scores)),
            "pass_rate_percent": pass_rate,
            "verdict": verdict,
        }

    return {
        "architecture": architecture,
        "num_samples": n_samples,
        "threshold_drop_percent": threshold_drop_percent,
        "modes": mode_results,
    }


def format_sanity_check_markdown_table(
    results: Union[List[Dict[str, Any]], Dict[str, Any]],
    title: str = "Model Parameter Randomization Sanity Checks (Adebayo et al., 2018)",
) -> str:
    """Formats sanity check results into a publication-ready Markdown table."""
    if isinstance(results, dict) and "modes" in results:
        results = [results]

    lines = [
        f"### {title}",
        "",
        "| Architecture | Randomization Mode | Mean Spearman ρ | Mean Pearson r | Cosine Sim | Pass Rate (%) | Adebayo Sanity Verdict |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for item in results:
        arch = item["architecture"]
        for mode, m_res in item["modes"].items():
            verdict_badge = f"**{m_res['verdict']}**"
            lines.append(
                f"| **{arch}** | `{mode}` | "
                f"{m_res['mean_spearman_correlation']:.4f} ± {m_res['std_spearman_correlation']:.4f} | "
                f"{m_res['mean_pearson_correlation']:.4f} | {m_res['mean_cosine_similarity']:.4f} | "
                f"{m_res['pass_rate_percent']:.1f}% | {verdict_badge} |"
            )

    lines.append("")
    lines.append(
        "*Adebayo Criterion*: Saliency explanations must be sensitive to model parameters. "
        "Parameter randomization collapsing attribution correlation (Spearman ρ ≤ 0.25 or drop ≥ 75%) "
        "confirms explanations reflect learned neural representations rather than raw image priors."
    )
    return "\n".join(lines)


def save_sanity_check_report(
    benchmark_results: Union[List[Dict[str, Any]], Dict[str, Any]],
    output_dir: Union[str, Path] = "results/tables/xai",
    filename_prefix: str = "sanity_checks_adebayo",
) -> Tuple[Path, Path]:
    """Saves sanity check benchmark results as structured JSON and formatted Markdown."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{filename_prefix}.json"
    md_path = out_dir / f"{filename_prefix}.md"

    save_json(benchmark_results, json_path)
    md_content = format_sanity_check_markdown_table(benchmark_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved Adebayo sanity check report to %s and %s", json_path, md_path)
    return json_path, md_path
