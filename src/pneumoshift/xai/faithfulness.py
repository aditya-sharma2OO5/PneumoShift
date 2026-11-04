"""Quantitative faithfulness evaluation: progressive deletion and insertion dynamics with multi-baseline support."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.xai.faithfulness")


def _build_baseline_image(
    image: torch.Tensor,
    baseline_mode: str = "zero",
    mask_value: float = 0.0,
    blur_kernel_size: int = 11,
    blur_sigma: float = 5.0,
) -> torch.Tensor:
    """Constructs baseline comparison image for progressive masking."""
    mode = str(baseline_mode).lower().strip()
    if mode == "zero":
        return torch.full_like(image, fill_value=mask_value)
    elif mode == "blur":
        k = blur_kernel_size if blur_kernel_size % 2 == 1 else blur_kernel_size + 1
        return TF.gaussian_blur(image, kernel_size=[k, k], sigma=[blur_sigma, blur_sigma])
    elif mode == "mean":
        return torch.mean(image, dim=(-2, -1), keepdim=True).expand_as(image)
    else:
        raise ValueError(f"Unsupported baseline mode '{mode}'. Expected one of ['zero', 'blur', 'mean']")


def compute_deletion_insertion_curves(
    model: nn.Module,
    image_tensor: torch.Tensor,
    saliency_map: np.ndarray,
    target_class_idx: int = 5,
    num_steps: int = 10,
    max_mask_percent: float = 50.0,
    baseline_mode: str = "zero",
    mask_value: float = 0.0,
    blur_kernel_size: int = 11,
    blur_sigma: float = 5.0,
) -> Dict[str, Any]:
    """Computes progressive deletion and insertion curves and their Area Under the Curve (AUC).

    Args:
        model: PyTorch model in eval mode.
        image_tensor: Input image tensor of shape (1, C, H, W) or (C, H, W).
        saliency_map: 2D attribution heatmap of shape (H, W).
        target_class_idx: Index of target disease class (e.g. 5 for Pneumonia).
        num_steps: Number of masking intervals from 0% to max_mask_percent.
        max_mask_percent: Maximum cumulative percentage of salient pixels to mask (e.g. 50.0%).
        baseline_mode: Baseline imputation strategy: 'zero', 'blur', or 'mean'.
        mask_value: Replacement value used when baseline_mode='zero'.
        blur_kernel_size: Kernel size for Gaussian blur baseline.
        blur_sigma: Sigma for Gaussian blur baseline.

    Returns:
        Dict with percentages, deletion_curve, insertion_curve, deletion_auc, insertion_auc,
        normalized_deletion_auc, normalized_insertion_auc, and faithfulness_ratio.
    """
    was_training = model.training
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)

    device = next(model.parameters()).device
    img = image_tensor.to(device).clone()

    h, w = saliency_map.shape
    total_pixels = h * w
    flat_indices = np.argsort(-saliency_map.ravel())  # Descending saliency rank order

    max_frac = float(np.clip(max_mask_percent / 100.0, 0.01, 1.0))
    percentages = np.linspace(0.0, max_frac, num_steps + 1)
    deletion_scores: List[float] = []
    insertion_scores: List[float] = []

    # Construct baseline neutral image
    baseline_img = _build_baseline_image(
        img,
        baseline_mode=baseline_mode,
        mask_value=mask_value,
        blur_kernel_size=blur_kernel_size,
        blur_sigma=blur_sigma,
    )

    try:
        with torch.no_grad():
            for p in percentages:
                k = int(p * total_pixels)
                mask_idx = flat_indices[:k]

                # Deletion: mask top-k% salient pixels from original image with baseline
                del_img = img.clone()
                c_channels = del_img.shape[1]
                for c in range(c_channels):
                    flat_del = del_img[0, c].view(-1)
                    flat_base = baseline_img[0, c].view(-1)
                    flat_del[mask_idx] = flat_base[mask_idx]

                del_logits = model(del_img)
                if isinstance(del_logits, tuple):
                    del_logits = del_logits[0]
                del_prob = float(torch.sigmoid(del_logits)[0, target_class_idx].item())
                deletion_scores.append(del_prob)

                # Insertion: restore top-k% salient pixels into blank baseline image
                ins_img = baseline_img.clone()
                for c in range(c_channels):
                    flat_ins = ins_img[0, c].view(-1)
                    flat_orig = img[0, c].view(-1)
                    flat_ins[mask_idx] = flat_orig[mask_idx]

                ins_logits = model(ins_img)
                if isinstance(ins_logits, tuple):
                    ins_logits = ins_logits[0]
                ins_prob = float(torch.sigmoid(ins_logits)[0, target_class_idx].item())
                insertion_scores.append(ins_prob)

        # Numerical trapezoidal AUC (NumPy 1.x / 2.0+ compatible)
        trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
        raw_del_auc = float(trapz_fn(deletion_scores, percentages))
        raw_ins_auc = float(trapz_fn(insertion_scores, percentages))

        norm_del_auc = float(raw_del_auc / max_frac) if max_frac > 0 else 0.0
        norm_ins_auc = float(raw_ins_auc / max_frac) if max_frac > 0 else 0.0
        faithfulness_ratio = float(norm_ins_auc / max(norm_del_auc, 1e-6))

        return {
            "percentages": [round(float(p * 100.0), 2) for p in percentages],
            "deletion_curve": deletion_scores,
            "insertion_curve": insertion_scores,
            "raw_deletion_auc": raw_del_auc,
            "raw_insertion_auc": raw_ins_auc,
            "deletion_auc": norm_del_auc,
            "insertion_auc": norm_ins_auc,
            "faithfulness_ratio": faithfulness_ratio,
            "baseline_mode": baseline_mode,
            "max_mask_percent": max_mask_percent,
        }

    finally:
        if was_training:
            model.train()


def evaluate_cohort_faithfulness(
    model: nn.Module,
    images: Union[List[torch.Tensor], torch.Tensor],
    cams: Union[List[np.ndarray], np.ndarray],
    target_class_idx: int = 5,
    num_steps: int = 10,
    max_mask_percent: float = 50.0,
    baseline_mode: str = "zero",
    blur_kernel_size: int = 11,
    blur_sigma: float = 5.0,
) -> Dict[str, Any]:
    """Computes mean and standard deviation of faithfulness dynamics across a cohort of images.

    Args:
        model: Evaluated PyTorch model.
        images: Batch tensor (N, C, H, W) or list of (1, C, H, W) tensors.
        cams: Batch array (N, H, W) or list of 2D heatmaps.
        target_class_idx: Index of target pathology.
        num_steps: Steps along masking curve.
        max_mask_percent: Upper percentage limit for salient pixel budget.
        baseline_mode: Baseline imputation mode.
        blur_kernel_size: Kernel dimension for blur mode.
        blur_sigma: Gaussian sigma for blur mode.

    Returns:
        Dict of cohort aggregated statistics and average curves.
    """
    n_samples = len(images)
    del_aucs: List[float] = []
    ins_aucs: List[float] = []
    ratios: List[float] = []
    all_del_curves: List[List[float]] = []
    all_ins_curves: List[List[float]] = []

    percentages_pct: List[float] = []

    for i in range(n_samples):
        img_i = images[i]
        cam_i = cams[i]
        res = compute_deletion_insertion_curves(
            model=model,
            image_tensor=img_i,
            saliency_map=cam_i,
            target_class_idx=target_class_idx,
            num_steps=num_steps,
            max_mask_percent=max_mask_percent,
            baseline_mode=baseline_mode,
            blur_kernel_size=blur_kernel_size,
            blur_sigma=blur_sigma,
        )
        del_aucs.append(res["deletion_auc"])
        ins_aucs.append(res["insertion_auc"])
        ratios.append(res["faithfulness_ratio"])
        all_del_curves.append(res["deletion_curve"])
        all_ins_curves.append(res["insertion_curve"])
        if not percentages_pct:
            percentages_pct = res["percentages"]

    avg_del_curve = np.mean(all_del_curves, axis=0).tolist()
    avg_ins_curve = np.mean(all_ins_curves, axis=0).tolist()

    return {
        "num_samples": n_samples,
        "baseline_mode": baseline_mode,
        "max_mask_percent": max_mask_percent,
        "percentages": percentages_pct,
        "mean_deletion_auc": float(np.mean(del_aucs)),
        "std_deletion_auc": float(np.std(del_aucs)),
        "mean_insertion_auc": float(np.mean(ins_aucs)),
        "std_insertion_auc": float(np.std(ins_aucs)),
        "mean_faithfulness_ratio": float(np.mean(ratios)),
        "std_faithfulness_ratio": float(np.std(ratios)),
        "avg_deletion_curve": avg_del_curve,
        "avg_insertion_curve": avg_ins_curve,
    }


def compare_cross_domain_faithfulness(
    in_domain_results: Dict[str, Any],
    cross_domain_results: Dict[str, Any],
    model_name: str = "DenseNet121",
    shift_name: str = "NIH_to_CheXpert",
) -> Dict[str, Any]:
    """Quantifies attribution faithfulness degradation between in-domain and out-of-domain deployments.

    Args:
        in_domain_results: Output of evaluate_cohort_faithfulness on source in-domain set.
        cross_domain_results: Output of evaluate_cohort_faithfulness on target cross-domain set.
        model_name: Name of evaluated architecture.
        shift_name: Distribution shift label.

    Returns:
        Dict comparing faithfulness metrics, deltas, and relative degradation percentages.
    """
    in_del = in_domain_results["mean_deletion_auc"]
    cross_del = cross_domain_results["mean_deletion_auc"]
    delta_del = cross_del - in_del  # Positive delta implies cross-domain failed to collapse upon deletion

    in_ins = in_domain_results["mean_insertion_auc"]
    cross_ins = cross_domain_results["mean_insertion_auc"]
    delta_ins = cross_ins - in_ins  # Negative delta implies cross-domain failed to recover upon insertion

    in_ratio = in_domain_results["mean_faithfulness_ratio"]
    cross_ratio = cross_domain_results["mean_faithfulness_ratio"]
    delta_ratio = cross_ratio - in_ratio

    rel_ratio_deg = (
        float((in_ratio - cross_ratio) / max(in_ratio, 1e-6) * 100.0) if in_ratio > 0 else 0.0
    )

    return {
        "model_name": model_name,
        "shift_name": shift_name,
        "baseline_mode": in_domain_results.get("baseline_mode", "zero"),
        "in_domain": {
            "deletion_auc": in_del,
            "insertion_auc": in_ins,
            "faithfulness_ratio": in_ratio,
        },
        "cross_domain": {
            "deletion_auc": cross_del,
            "insertion_auc": cross_ins,
            "faithfulness_ratio": cross_ratio,
        },
        "deltas": {
            "delta_deletion_auc": delta_del,
            "delta_insertion_auc": delta_ins,
            "delta_faithfulness_ratio": delta_ratio,
            "faithfulness_degradation_percent": rel_ratio_deg,
        },
    }


def format_faithfulness_markdown_table(
    comparisons: Union[List[Dict[str, Any]], Dict[str, Any]],
    title: str = "Attribution Faithfulness Dynamics (Petsiuk et al., 2018)",
) -> str:
    """Formats faithfulness comparison results into a publication-ready Markdown table."""
    comp_list: List[Dict[str, Any]] = []
    if isinstance(comparisons, dict):
        if "comparison" in comparisons:
            comp_obj = comparisons["comparison"]
            comp_list = [comp_obj] if isinstance(comp_obj, dict) else comp_obj
        elif "model_name" in comparisons:
            comp_list = [comparisons]
        else:
            comp_list = [v for v in comparisons.values() if isinstance(v, dict) and "model_name" in v]
    elif isinstance(comparisons, list):
        comp_list = comparisons

    lines = [
        f"### {title}",
        "",
        "| Model Architecture | Distribution Shift | Mode | In-Domain Del AUC | In-Domain Ins AUC | In-Domain Ratio | Cross Del AUC | Cross Ins AUC | Cross Ratio | Δ Ratio | Degradation (%) |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for comp in comp_list:
        m_name = comp.get("model_name", "Model")
        shift = comp.get("shift_name", "In_to_Cross")
        mode = comp.get("baseline_mode", "zero")
        in_d = comp.get("in_domain", {})
        cr_d = comp.get("cross_domain", {})
        deltas = comp.get("deltas", {})

        lines.append(
            f"| **{m_name}** | {shift} | `{mode}` | "
            f"{in_d.get('deletion_auc', 0.0):.4f} | {in_d.get('insertion_auc', 0.0):.4f} | **{in_d.get('faithfulness_ratio', 0.0):.2f}x** | "
            f"{cr_d.get('deletion_auc', 0.0):.4f} | {cr_d.get('insertion_auc', 0.0):.4f} | **{cr_d.get('faithfulness_ratio', 0.0):.2f}x** | "
            f"{deltas.get('delta_faithfulness_ratio', 0.0):+.2f} | **{deltas.get('faithfulness_degradation_percent', 0.0):.1f}%** |"
        )

    lines.append("")
    lines.append(
        "*Notes*: Deletion AUC (lower is better, reflecting necessary salient features). "
        "Insertion AUC (higher is better, reflecting sufficient salient features). "
        "Faithfulness Ratio = Insertion AUC / Deletion AUC."
    )
    return "\n".join(lines)


def save_faithfulness_report(
    benchmark_results: Union[List[Dict[str, Any]], Dict[str, Any]],
    output_dir: Union[str, Path] = "results/tables/xai",
    filename_prefix: str = "faithfulness_dynamics",
) -> Tuple[Path, Path]:
    """Saves faithfulness benchmark results as structured JSON and formatted Markdown."""
    out_path = Path(output_dir)
    if out_path.suffix == ".json":
        json_path = out_path
        md_path = out_path.with_suffix(".md")
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path.mkdir(parents=True, exist_ok=True)
        json_path = out_path / f"{filename_prefix}.json"
        md_path = out_path / f"{filename_prefix}.md"

    save_json(benchmark_results, json_path)
    md_content = format_faithfulness_markdown_table(benchmark_results)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    logger.info("Saved faithfulness report to %s and %s", json_path, md_path)
    return json_path, md_path
