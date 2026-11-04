"""Attribution stability analysis across seeds, geometric/photometric perturbations, and architectures."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from pneumoshift.models.factory import get_target_layer_for_xai
from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger
from pneumoshift.xai.gradcam import GradCAMExplainer

logger = get_logger("pneumoshift.xai.stability")


def compute_binarized_iou(
    map_a: np.ndarray,
    map_b: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Computes the Intersection-over-Union (IoU) of two attribution maps binarized at a threshold.

    Args:
        map_a: 2D numpy array normalized in [0, 1].
        map_b: 2D numpy array normalized in [0, 1].
        threshold: Binarization threshold in (0.0, 1.0).

    Returns:
        IoU score in [0.0, 1.0]. Returns 1.0 if both maps are completely below threshold.
    """
    if map_a.shape != map_b.shape:
        # Resize map_b to match map_a shape using bilinear interpolation
        t_b = torch.tensor(map_b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_b_resized = TF.resize(t_b, list(map_a.shape), antialias=True)
        map_b = t_b_resized.squeeze().numpy()

    mask_a = (map_a >= threshold)
    mask_b = (map_b >= threshold)

    intersection = float(np.logical_and(mask_a, mask_b).sum())
    union = float(np.logical_or(mask_a, mask_b).sum())

    if union < 1e-8:
        return 1.0
    return float(intersection / union)


def compute_similarity_metrics(
    map_a: np.ndarray,
    map_b: np.ndarray,
    binarize_threshold: float = 0.5,
) -> Dict[str, float]:
    """Computes Spearman rho, Pearson r, Cosine similarity, MSE, and binarized IoU between two maps.

    Handles zero-variance/constant maps safely without raising RuntimeWarnings or returning NaNs.

    Args:
        map_a: First 2D attribution heatmap of shape (H, W).
        map_b: Second 2D attribution heatmap of shape (H, W).
        binarize_threshold: Cutoff for binarized IoU computation.

    Returns:
        Dictionary of correlation and alignment metrics.
    """
    if map_a.shape != map_b.shape:
        t_b = torch.tensor(map_b, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t_b_resized = TF.resize(t_b, list(map_a.shape), antialias=True)
        map_b = t_b_resized.squeeze().numpy()

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

    # Binarized IoU
    iou = compute_binarized_iou(map_a, map_b, threshold=binarize_threshold)

    return {
        "spearman": spearman_corr,
        "pearson": pearson_corr,
        "cosine_similarity": cosine_sim,
        "mse": mse,
        "iou": iou,
    }


def compute_attribution_stability(
    explainer: GradCAMExplainer,
    image_tensor: torch.Tensor,
    target_class_idx: int = 5,
    num_perturbations: int = 5,
    max_rotation_deg: float = 5.0,
    brightness_factor: float = 0.05,
    translation_percent: float = 0.05,
    binarize_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Measures attribution stability under minor geometric and photometric perturbations.

    Applies affine transformations (small rotations +/-5 deg, spatial translations +/-5%)
    and photometric jitter (+/-5% brightness), computes Grad-CAM on the perturbed image,
    inverses the geometric transform to align back to base coordinates, and measures
    attribution correlation and IoU against the unperturbed explanation.

    Args:
        explainer: Initialized GradCAMExplainer.
        image_tensor: Input radiograph tensor of shape (1, C, H, W) or (C, H, W).
        target_class_idx: Target disease index (default: 5 for Pneumonia).
        num_perturbations: Number of stochastic perturbation trials.
        max_rotation_deg: Maximum rotation angle in degrees (+/-).
        brightness_factor: Maximum brightness shift fraction (+/-).
        translation_percent: Maximum translation shift as fraction of image dimension (+/-).
        binarize_threshold: Threshold for binarized IoU overlap.

    Returns:
        Dictionary containing mean and std of Spearman rho, Pearson r, Cosine similarity,
        MSE, and IoU across perturbation trials, plus per-trial metrics.
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    _, _, h, w = image_tensor.shape
    base_cam = explainer.generate_cam(image_tensor, target_class_idx=target_class_idx)

    if num_perturbations <= 0:
        return {
            "mean_spearman": 1.0,
            "std_spearman": 0.0,
            "mean_pearson": 1.0,
            "std_pearson": 0.0,
            "mean_cosine": 1.0,
            "std_cosine": 0.0,
            "mean_mse": 0.0,
            "std_mse": 0.0,
            "mean_iou": 1.0,
            "std_iou": 0.0,
            "mean_perturbation_stability": 1.0,
            "std_perturbation_stability": 0.0,
            "trials": [],
        }

    trials: List[Dict[str, Any]] = []

    for trial_idx in range(num_perturbations):
        rot = float(np.random.uniform(-max_rotation_deg, max_rotation_deg)) if max_rotation_deg > 0 else 0.0
        tx = float(np.random.uniform(-translation_percent, translation_percent)) if translation_percent > 0 else 0.0
        ty = float(np.random.uniform(-translation_percent, translation_percent)) if translation_percent > 0 else 0.0
        dx = int(round(tx * w))
        dy = int(round(ty * h))
        bf = float(np.random.uniform(1.0 - brightness_factor, 1.0 + brightness_factor)) if brightness_factor > 0 else 1.0

        # Apply forward geometric and photometric perturbation
        perturbed = TF.affine(image_tensor, angle=rot, translate=[dx, dy], scale=1.0, shear=0.0)
        perturbed = TF.adjust_brightness(perturbed, bf)

        pert_cam = explainer.generate_cam(perturbed, target_class_idx=target_class_idx)

        # Invert geometric transform: step 1 undo translation, step 2 undo rotation
        pert_tensor = torch.tensor(pert_cam, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        untranslated = TF.affine(pert_tensor, angle=0.0, translate=[-dx, -dy], scale=1.0, shear=0.0)
        aligned_tensor = TF.affine(untranslated, angle=-rot, translate=[0, 0], scale=1.0, shear=0.0)
        pert_cam_aligned = np.clip(aligned_tensor.squeeze().cpu().numpy(), 0.0, 1.0)

        metrics = compute_similarity_metrics(base_cam, pert_cam_aligned, binarize_threshold=binarize_threshold)
        trial_record = {
            "trial": trial_idx + 1,
            "rotation_deg": rot,
            "translation_pixels": [dx, dy],
            "brightness_factor": bf,
            **metrics,
        }
        trials.append(trial_record)

    spearmans = [t["spearman"] for t in trials]
    pearsons = [t["pearson"] for t in trials]
    cosines = [t["cosine_similarity"] for t in trials]
    mses = [t["mse"] for t in trials]
    ious = [t["iou"] for t in trials]

    mean_spearman = float(np.mean(spearmans))
    std_spearman = float(np.std(spearmans))

    return {
        "mean_spearman": mean_spearman,
        "std_spearman": std_spearman,
        "mean_pearson": float(np.mean(pearsons)),
        "std_pearson": float(np.std(pearsons)),
        "mean_cosine": float(np.mean(cosines)),
        "std_cosine": float(np.std(cosines)),
        "mean_mse": float(np.mean(mses)),
        "std_mse": float(np.std(mses)),
        "mean_iou": float(np.mean(ious)),
        "std_iou": float(np.std(ious)),
        # Backwards compatibility aliases
        "mean_perturbation_stability": mean_spearman,
        "std_perturbation_stability": std_spearman,
        "trials": trials,
    }


def compare_inter_seed_stability(
    models_or_explainers: Dict[Union[int, str], Any],
    image_tensor: torch.Tensor,
    architecture: Optional[str] = "densenet121",
    target_class_idx: int = 5,
    binarize_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Quantifies attribution reproducibility across models trained with different random seeds.

    Generates Grad-CAM explanations for identical radiographs across seed checkpoints
    (e.g., seeds 42, 1337, 2024) and computes all pairwise correlation and IoU metrics.

    Args:
        models_or_explainers: Mapping from seed (int or str) to either nn.Module or GradCAMExplainer.
        image_tensor: Radiograph input tensor of shape (1, C, H, W) or (C, H, W).
        architecture: Model architecture string (used to locate target layer if models are passed).
        target_class_idx: Target disease class index (default: 5 for Pneumonia).
        binarize_threshold: Threshold for binarized IoU overlap.

    Returns:
        Dictionary containing pairwise seed comparisons, mean/std Spearman rho, Pearson r,
        Cosine similarity, MSE, and IoU.
    """
    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    seed_keys = list(models_or_explainers.keys())
    if len(seed_keys) < 2:
        return {
            "seeds": [str(s) for s in seed_keys],
            "pairwise_comparisons": {},
            "mean_inter_seed_spearman": 1.0,
            "std_inter_seed_spearman": 0.0,
            "mean_inter_seed_pearson": 1.0,
            "std_inter_seed_pearson": 0.0,
            "mean_inter_seed_cosine": 1.0,
            "std_inter_seed_cosine": 0.0,
            "mean_inter_seed_mse": 0.0,
            "std_inter_seed_mse": 0.0,
            "mean_inter_seed_iou": 1.0,
            "std_inter_seed_iou": 0.0,
        }

    cams: Dict[Union[int, str], np.ndarray] = {}
    temp_explainers: List[GradCAMExplainer] = []

    try:
        for seed, item in models_or_explainers.items():
            if isinstance(item, GradCAMExplainer):
                expl = item
            elif isinstance(item, nn.Module):
                arch = architecture or "densenet121"
                target_layer = get_target_layer_for_xai(item, arch)
                expl = GradCAMExplainer(item, target_layer)
                temp_explainers.append(expl)
            else:
                raise TypeError(f"Expected GradCAMExplainer or nn.Module for seed {seed}, got {type(item)}")

            cam = expl.generate_cam(image_tensor, target_class_idx=target_class_idx)
            cams[seed] = cam

        pairwise: Dict[str, Dict[str, float]] = {}
        spearmans: List[float] = []
        pearsons: List[float] = []
        cosines: List[float] = []
        mses: List[float] = []
        ious: List[float] = []

        for i in range(len(seed_keys)):
            for j in range(i + 1, len(seed_keys)):
                s1, s2 = seed_keys[i], seed_keys[j]
                pair_name = f"seed_{s1}_vs_seed_{s2}"
                metrics = compute_similarity_metrics(cams[s1], cams[s2], binarize_threshold=binarize_threshold)
                pairwise[pair_name] = metrics

                spearmans.append(metrics["spearman"])
                pearsons.append(metrics["pearson"])
                cosines.append(metrics["cosine_similarity"])
                mses.append(metrics["mse"])
                ious.append(metrics["iou"])

        return {
            "seeds": [str(s) for s in seed_keys],
            "pairwise_comparisons": pairwise,
            "mean_inter_seed_spearman": float(np.mean(spearmans)),
            "std_inter_seed_spearman": float(np.std(spearmans)),
            "mean_inter_seed_pearson": float(np.mean(pearsons)),
            "std_inter_seed_pearson": float(np.std(pearsons)),
            "mean_inter_seed_cosine": float(np.mean(cosines)),
            "std_inter_seed_cosine": float(np.std(cosines)),
            "mean_inter_seed_mse": float(np.mean(mses)),
            "std_inter_seed_mse": float(np.std(mses)),
            "mean_inter_seed_iou": float(np.mean(ious)),
            "std_inter_seed_iou": float(np.std(ious)),
        }
    finally:
        for expl in temp_explainers:
            expl.release_hooks()


def compare_inter_architecture_stability(
    model_or_explainer_a: Any,
    model_or_explainer_b: Any,
    arch_a: str = "densenet121",
    arch_b: str = "efficientnet_b0",
    image_tensor: Optional[torch.Tensor] = None,
    target_class_idx: int = 5,
    binarize_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compares visual explanation consistency between two distinct model architectures.

    Quantifies attribution overlap between DenseNet121 (dense concatenation inductive bias)
    and EfficientNet-B0 (depthwise separable convolution inductive bias).

    Args:
        model_or_explainer_a: First model (nn.Module) or GradCAMExplainer.
        model_or_explainer_b: Second model (nn.Module) or GradCAMExplainer.
        arch_a: Architecture name of model A.
        arch_b: Architecture name of model B.
        image_tensor: Radiograph input tensor. If None, raises ValueError.
        target_class_idx: Target disease class index (default: 5 for Pneumonia).
        binarize_threshold: Threshold for binarized IoU overlap.

    Returns:
        Dictionary of comparative metrics between the two architectures.
    """
    if image_tensor is None:
        raise ValueError("image_tensor cannot be None")

    if image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)

    temp_explainers: List[GradCAMExplainer] = []
    try:
        if isinstance(model_or_explainer_a, GradCAMExplainer):
            expl_a = model_or_explainer_a
        elif isinstance(model_or_explainer_a, nn.Module):
            layer_a = get_target_layer_for_xai(model_or_explainer_a, arch_a)
            expl_a = GradCAMExplainer(model_or_explainer_a, layer_a)
            temp_explainers.append(expl_a)
        else:
            raise TypeError(f"Invalid type for model_or_explainer_a: {type(model_or_explainer_a)}")

        if isinstance(model_or_explainer_b, GradCAMExplainer):
            expl_b = model_or_explainer_b
        elif isinstance(model_or_explainer_b, nn.Module):
            layer_b = get_target_layer_for_xai(model_or_explainer_b, arch_b)
            expl_b = GradCAMExplainer(model_or_explainer_b, layer_b)
            temp_explainers.append(expl_b)
        else:
            raise TypeError(f"Invalid type for model_or_explainer_b: {type(model_or_explainer_b)}")

        cam_a = expl_a.generate_cam(image_tensor, target_class_idx=target_class_idx)
        cam_b = expl_b.generate_cam(image_tensor, target_class_idx=target_class_idx)

        metrics = compute_similarity_metrics(cam_a, cam_b, binarize_threshold=binarize_threshold)

        return {
            "architecture_a": arch_a,
            "architecture_b": arch_b,
            "spearman": metrics["spearman"],
            "pearson": metrics["pearson"],
            "cosine_similarity": metrics["cosine_similarity"],
            "mse": metrics["mse"],
            "iou": metrics["iou"],
        }
    finally:
        for expl in temp_explainers:
            expl.release_hooks()


def evaluate_cohort_stability(
    models: Dict[str, Any],
    images: List[torch.Tensor],
    target_class_idx: int = 5,
    architecture: Optional[str] = "densenet121",
    num_perturbations: int = 3,
    max_rotation_deg: float = 5.0,
    brightness_factor: float = 0.05,
    translation_percent: float = 0.05,
    binarize_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Evaluates attribution stability across an entire cohort of radiographs.

    Computes perturbation invariance per model and inter-seed reproducibility across models
    for each patient radiograph in the evaluation cohort.

    Args:
        models: Dictionary mapping model identifier (e.g. "seed_42", "DenseNet121") to nn.Module or GradCAMExplainer.
        images: List of radiograph tensors of shape (1, C, H, W) or (C, H, W).
        target_class_idx: Target disease index.
        architecture: Default architecture name.
        num_perturbations: Number of perturbation trials per image.
        max_rotation_deg: Max rotation in degrees.
        brightness_factor: Max brightness shift factor.
        translation_percent: Max translation shift percent.
        binarize_threshold: Threshold for binarized IoU overlap.

    Returns:
        Consolidated dictionary of cohort-level stability statistics.
    """
    if not images:
        raise ValueError("Cohort images list cannot be empty")

    cohort_perturbations: Dict[str, List[Dict[str, float]]] = {name: [] for name in models.keys()}
    cohort_inter_seed: List[Dict[str, Any]] = []

    # Initialize explainers if modules passed
    active_explainers: Dict[str, GradCAMExplainer] = {}
    temp_explainers: List[GradCAMExplainer] = []

    try:
        for name, item in models.items():
            if isinstance(item, GradCAMExplainer):
                active_explainers[name] = item
            elif isinstance(item, nn.Module):
                arch = architecture or "densenet121"
                target_layer = get_target_layer_for_xai(item, arch)
                expl = GradCAMExplainer(item, target_layer)
                active_explainers[name] = expl
                temp_explainers.append(expl)
            else:
                raise TypeError(f"Invalid model entry for '{name}': {type(item)}")

        for img in images:
            # Perturbation stability for each model
            for name, expl in active_explainers.items():
                p_res = compute_attribution_stability(
                    explainer=expl,
                    image_tensor=img,
                    target_class_idx=target_class_idx,
                    num_perturbations=num_perturbations,
                    max_rotation_deg=max_rotation_deg,
                    brightness_factor=brightness_factor,
                    translation_percent=translation_percent,
                    binarize_threshold=binarize_threshold,
                )
                cohort_perturbations[name].append({
                    "spearman": p_res["mean_spearman"],
                    "pearson": p_res["mean_pearson"],
                    "cosine": p_res["mean_cosine"],
                    "iou": p_res["mean_iou"],
                    "mse": p_res["mean_mse"],
                })

            # Inter-seed reproducibility across models if >= 2 models
            if len(active_explainers) >= 2:
                seed_res = compare_inter_seed_stability(
                    models_or_explainers=active_explainers,
                    image_tensor=img,
                    architecture=architecture,
                    target_class_idx=target_class_idx,
                    binarize_threshold=binarize_threshold,
                )
                cohort_inter_seed.append(seed_res)

        # Aggregate perturbation stats per model
        aggregated_perturbations: Dict[str, Dict[str, float]] = {}
        for name, records in cohort_perturbations.items():
            aggregated_perturbations[name] = {
                "mean_spearman": float(np.mean([r["spearman"] for r in records])),
                "std_spearman": float(np.std([r["spearman"] for r in records])),
                "mean_pearson": float(np.mean([r["pearson"] for r in records])),
                "std_pearson": float(np.std([r["pearson"] for r in records])),
                "mean_cosine": float(np.mean([r["cosine"] for r in records])),
                "std_cosine": float(np.std([r["cosine"] for r in records])),
                "mean_iou": float(np.mean([r["iou"] for r in records])),
                "std_iou": float(np.std([r["iou"] for r in records])),
                "mean_mse": float(np.mean([r["mse"] for r in records])),
                "std_mse": float(np.std([r["mse"] for r in records])),
            }

        # Aggregate inter-seed stats across cohort
        aggregated_inter_seed: Dict[str, Any] = {}
        if cohort_inter_seed:
            aggregated_inter_seed = {
                "seeds": cohort_inter_seed[0]["seeds"],
                "cohort_mean_spearman": float(np.mean([s["mean_inter_seed_spearman"] for s in cohort_inter_seed])),
                "cohort_std_spearman": float(np.std([s["mean_inter_seed_spearman"] for s in cohort_inter_seed])),
                "cohort_mean_pearson": float(np.mean([s["mean_inter_seed_pearson"] for s in cohort_inter_seed])),
                "cohort_std_pearson": float(np.std([s["mean_inter_seed_pearson"] for s in cohort_inter_seed])),
                "cohort_mean_cosine": float(np.mean([s["mean_inter_seed_cosine"] for s in cohort_inter_seed])),
                "cohort_std_cosine": float(np.std([s["mean_inter_seed_cosine"] for s in cohort_inter_seed])),
                "cohort_mean_iou": float(np.mean([s["mean_inter_seed_iou"] for s in cohort_inter_seed])),
                "cohort_std_iou": float(np.std([s["mean_inter_seed_iou"] for s in cohort_inter_seed])),
                "cohort_mean_mse": float(np.mean([s["mean_inter_seed_mse"] for s in cohort_inter_seed])),
                "cohort_std_mse": float(np.std([s["mean_inter_seed_mse"] for s in cohort_inter_seed])),
            }

        return {
            "num_cohort_samples": len(images),
            "perturbation_stability": aggregated_perturbations,
            "inter_seed_reproducibility": aggregated_inter_seed,
        }
    finally:
        for expl in temp_explainers:
            expl.release_hooks()


def format_stability_markdown_table(results: Dict[str, Any]) -> str:
    """Formats attribution stability results into publication-ready GitHub Flavored Markdown.

    Args:
        results: Dictionary containing stability benchmarks (perturbation, inter-seed, inter-arch).

    Returns:
        Formatted markdown string.
    """
    lines: List[str] = [
        "# Attribution Stability & Inter-Seed Reproducibility Analysis",
        "",
        "> **Hypothesis 5C Evaluation**: Evaluates visual explanation stability against geometric/photometric",
        "> perturbations, reproducibility across initialization seeds (42, 1337, 2024), and inductive bias",
        "> concordance between DenseNet121 and EfficientNet-B0 architectures.",
        "",
    ]

    # Section 1: Perturbation Invariance
    pert = results.get("perturbation_stability", {})
    if pert:
        lines.extend([
            "## 1. Geometric & Photometric Perturbation Invariance",
            "",
            "| Model / Architecture | Spearman $\\rho$ | Pearson $r$ | Cosine Similarity | Binarized IoU ($\\tau=0.5$) | MSE |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
        ])
        for model_name, m in pert.items():
            lines.append(
                f"| `{model_name}` | {m.get('mean_spearman', 0.0):.4f} ± {m.get('std_spearman', 0.0):.4f} | "
                f"{m.get('mean_pearson', 0.0):.4f} ± {m.get('std_pearson', 0.0):.4f} | "
                f"{m.get('mean_cosine', 0.0):.4f} ± {m.get('std_cosine', 0.0):.4f} | "
                f"{m.get('mean_iou', 0.0):.4f} ± {m.get('std_iou', 0.0):.4f} | "
                f"{m.get('mean_mse', 0.0):.6f} |"
            )
        lines.append("")

    # Section 2: Inter-Seed Reproducibility
    seed_data = results.get("inter_seed_reproducibility", {})
    if seed_data:
        lines.extend([
            "## 2. Multi-Seed Attribution Reproducibility (Seeds 42, 1337, 2024)",
            "",
            f"**Seeds Evaluated**: `{', '.join(str(s) for s in seed_data.get('seeds', []))}`",
            "",
            "| Metric | Mean Across Cohort | Standard Deviation | Description |",
            "|:---|:---:|:---:|:---|",
            f"| **Spearman Rank Correlation ($\\rho$)** | {seed_data.get('cohort_mean_spearman', seed_data.get('mean_inter_seed_spearman', 0.0)):.4f} | "
            f"± {seed_data.get('cohort_std_spearman', seed_data.get('std_inter_seed_spearman', 0.0)):.4f} | Monotonic attribution rank concordance across seeds |",
            f"| **Pearson Correlation ($r$)** | {seed_data.get('cohort_mean_pearson', seed_data.get('mean_inter_seed_pearson', 0.0)):.4f} | "
            f"± {seed_data.get('cohort_std_pearson', seed_data.get('std_inter_seed_pearson', 0.0)):.4f} | Linear intensity correlation across seeds |",
            f"| **Cosine Similarity** | {seed_data.get('cohort_mean_cosine', seed_data.get('mean_inter_seed_cosine', 0.0)):.4f} | "
            f"± {seed_data.get('cohort_std_cosine', seed_data.get('std_inter_seed_cosine', 0.0)):.4f} | Directional alignment of visual explanation vectors |",
            f"| **Binarized IoU ($\\tau=0.5$)** | {seed_data.get('cohort_mean_iou', seed_data.get('mean_inter_seed_iou', 0.0)):.4f} | "
            f"± {seed_data.get('cohort_std_iou', seed_data.get('std_inter_seed_iou', 0.0)):.4f} | Spatial overlap of top-50% salient lesion attributions |",
            f"| **Mean Squared Error (MSE)** | {seed_data.get('cohort_mean_mse', seed_data.get('mean_inter_seed_mse', 0.0)):.6f} | "
            f"± {seed_data.get('cohort_std_mse', seed_data.get('std_inter_seed_mse', 0.0)):.6f} | Direct pixel-wise error across random seeds |",
            "",
        ])

        # If pairwise comparisons are available
        pairwise = seed_data.get("pairwise_comparisons", {})
        if pairwise:
            lines.extend([
                "### Pairwise Seed Comparisons",
                "",
                "| Seed Comparison Pair | Spearman $\\rho$ | Pearson $r$ | Cosine Similarity | Binarized IoU | MSE |",
                "|:---|:---:|:---:|:---:|:---:|:---:|",
            ])
            for pair_name, m in pairwise.items():
                lines.append(
                    f"| `{pair_name}` | {m.get('spearman', 0.0):.4f} | {m.get('pearson', 0.0):.4f} | "
                    f"{m.get('cosine_similarity', 0.0):.4f} | {m.get('iou', 0.0):.4f} | {m.get('mse', 0.0):.6f} |"
                )
            lines.append("")

    # Section 3: Inter-Architecture Attribution Consistency
    arch_data = results.get("inter_architecture_consistency", {})
    if arch_data:
        lines.extend([
            "## 3. Inter-Architecture Attribution Consistency",
            "",
            "| Architecture Pair | Spearman $\\rho$ | Pearson $r$ | Cosine Similarity | Binarized IoU ($\\tau=0.5$) | MSE |",
            "|:---|:---:|:---:|:---:|:---:|:---:|",
            f"| `{arch_data.get('architecture_a')}` vs `{arch_data.get('architecture_b')}` | "
            f"{arch_data.get('spearman', 0.0):.4f} | {arch_data.get('pearson', 0.0):.4f} | "
            f"{arch_data.get('cosine_similarity', 0.0):.4f} | {arch_data.get('iou', 0.0):.4f} | "
            f"{arch_data.get('mse', 0.0):.6f} |",
            "",
        ])

    return "\n".join(lines)


def save_stability_report(
    stability_results: Dict[str, Any],
    output_path: Union[str, Path] = "results/tables/xai/stability_comparison.json",
) -> Tuple[Path, Path]:
    """Saves stability evaluation results as structured JSON and academic Markdown.

    Args:
        stability_results: Output dictionary from stability benchmark evaluations.
        output_path: Target path for the JSON report (default: results/tables/xai/stability_comparison.json).

    Returns:
        Tuple of (json_path, markdown_path).
    """
    json_path = Path(output_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(stability_results, json_path)

    md_path = json_path.with_suffix(".md")
    md_content = format_stability_markdown_table(stability_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved stability JSON report to %s and Markdown to %s", json_path, md_path)
    return json_path, md_path
