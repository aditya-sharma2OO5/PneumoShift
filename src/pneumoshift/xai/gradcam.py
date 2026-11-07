"""Self-contained, robust Grad-CAM implementation with multi-architecture support and overlay generation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
import matplotlib.cm as cm
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from pneumoshift.models.factory import get_target_layer_for_xai
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.xai.gradcam")


class GradCAMExplainer:
    """Computes Gradient-weighted Class Activation Mapping (Grad-CAM) with context management."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hooks_released = False

        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

    def __enter__(self) -> "GradCAMExplainer":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.release_hooks()

    def _save_activations(self, module: nn.Module, input: Any, output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module: nn.Module, grad_input: Any, grad_output: Tuple[torch.Tensor, ...]) -> None:
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_idx: int = 5,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """Generates normalized [0, 1] 2D Grad-CAM heatmap for a single image or batch.

        Args:
            input_tensor: Image tensor of shape (B, C, H, W) where B >= 1.
            target_class_idx: Index of target class (e.g. 5 for Pneumonia).
            target_size: Output spatial resolution (H, W). Defaults to input spatial dimensions.

        Returns:
            np.ndarray heatmap: (H, W) if B == 1, or (B, H, W) if B > 1, scaled to [0, 1].
        """
        was_training = self.model.training
        self.model.eval()
        try:
            self.model.zero_grad()
            batch_size = input_tensor.shape[0]
            if target_size is None:
                target_size = (input_tensor.shape[2], input_tensor.shape[3])

            device = next(self.model.parameters()).device
            if input_tensor.device != device:
                input_tensor = input_tensor.to(device)

            logits = self.model(input_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]

            score = logits[:, target_class_idx]
            # Summing across independent batch items gives exact per-item gradients
            score.sum().backward(retain_graph=True)

            if self.gradients is None or self.activations is None:
                raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

            # Channel-wise mean of gradients: (B, C, 1, 1)
            weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
            cam = F.relu(cam)

            # Interpolate to target size
            cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
            cam_np = cam.squeeze(1).cpu().numpy()

            # Normalize each sample independently to [0, 1]
            if batch_size == 1:
                sample_cam = cam_np[0] if cam_np.ndim == 3 else cam_np
                c_min, c_max = sample_cam.min(), sample_cam.max()
                if c_max > c_min:
                    return (sample_cam - c_min) / (c_max - c_min)
                return np.zeros_like(sample_cam)

            normalized_batch = np.zeros_like(cam_np)
            for b in range(batch_size):
                item = cam_np[b]
                i_min, i_max = item.min(), item.max()
                if i_max > i_min:
                    normalized_batch[b] = (item - i_min) / (i_max - i_min)
                else:
                    normalized_batch[b] = np.zeros_like(item)
            return normalized_batch

        finally:
            if was_training:
                self.model.train()

    def generate_multi_class_cam(
        self,
        input_tensor: torch.Tensor,
        target_class_indices: List[int],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[int, np.ndarray]:
        """Generates CAM heatmaps for multiple disease classes on the same input image.

        Args:
            input_tensor: Image tensor of shape (1, C, H, W).
            target_class_indices: List of target class indices.
            target_size: Output spatial resolution.

        Returns:
            Dict mapping class_idx -> 2D heatmap np.ndarray in [0, 1].
        """
        results: Dict[int, np.ndarray] = {}
        for c_idx in target_class_indices:
            cam = self.generate_cam(input_tensor, target_class_idx=c_idx, target_size=target_size)
            results[c_idx] = cam
        return results

    def release_hooks(self) -> None:
        """Removes PyTorch forward and backward hooks."""
        if not self._hooks_released:
            self._forward_hook.remove()
            self._backward_hook.remove()
            self._hooks_released = True


def _normalize_image_to_rgb(image: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Converts torch tensor or numpy array to a float RGB image of shape (H, W, 3) in [0, 1]."""
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().squeeze().numpy()
    else:
        img = np.array(image, copy=True).squeeze()

    # Handle (C, H, W) -> (H, W, C)
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))

    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)

    # Scale to [0, 1]
    img = img.astype(np.float32)
    i_min, i_max = img.min(), img.max()
    if i_max > 1.0 or i_min < 0.0:
        if i_max > i_min:
            img = (img - i_min) / (i_max - i_min)
        else:
            img = np.zeros_like(img)
    return img


def overlay_cam_on_image(
    image: Union[np.ndarray, torch.Tensor],
    cam: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
) -> np.ndarray:
    """Blends a 2D Grad-CAM heatmap over an input radiograph.

    Args:
        image: Radiograph as tensor or array of shape (C, H, W), (H, W, C), or (H, W).
        cam: 2D normalized heatmap of shape (H, W) in [0, 1].
        colormap: Matplotlib colormap string ('jet', 'turbo', 'viridis', etc.).
        alpha: Blending weight for heatmap overlay (0.0 = original image, 1.0 = heatmap).

    Returns:
        np.ndarray of shape (H, W, 3) with uint8 values in [0, 255].
    """
    img_rgb = _normalize_image_to_rgb(image)
    h, w = img_rgb.shape[:2]

    # Resize CAM if spatial dimensions differ
    if cam.shape[:2] != (h, w):
        cam_pil = Image.fromarray((cam * 255.0).astype(np.uint8))
        cam = np.array(cam_pil.resize((w, h), Image.Resampling.BILINEAR)) / 255.0

    # Apply colormap
    try:
        cmap = matplotlib.colormaps.get_cmap(colormap)
    except AttributeError:
        cmap = cm.get_cmap(colormap)

    cam_color = cmap(np.clip(cam, 0.0, 1.0))[:, :, :3].astype(np.float32)

    # Alpha blending
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = alpha * cam_color + (1.0 - alpha) * img_rgb
    blended_uint8 = np.clip(blended * 255.0, 0, 255).astype(np.uint8)
    return blended_uint8


def save_cam_visualization(
    output_path: Union[str, Path],
    image: Union[np.ndarray, torch.Tensor],
    cam: np.ndarray,
    colormap: str = "jet",
    alpha: float = 0.5,
    side_by_side: bool = True,
) -> Path:
    """Saves Grad-CAM overlay or side-by-side composite visualization to disk.

    Args:
        output_path: Path to write image file (e.g. .png).
        image: Input radiograph.
        cam: Normalized heatmap in [0, 1].
        colormap: Colormap string.
        alpha: Overlay blending transparency.
        side_by_side: If True, saves composite [Raw Image | Colored CAM | Blended Overlay].

    Returns:
        Path to written image file.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    img_rgb = _normalize_image_to_rgb(image)
    overlay = overlay_cam_on_image(image, cam, colormap=colormap, alpha=alpha)

    if not side_by_side:
        pil_img = Image.fromarray(overlay)
        pil_img.save(out_file)
        return out_file

    h, w = img_rgb.shape[:2]
    # Colored CAM
    try:
        cmap = matplotlib.colormaps.get_cmap(colormap)
    except AttributeError:
        cmap = cm.get_cmap(colormap)
    cam_color = np.clip(cmap(np.clip(cam, 0.0, 1.0))[:, :, :3] * 255.0, 0, 255).astype(np.uint8)
    img_uint8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)

    # 3-panel side-by-side composite
    composite = np.concatenate([img_uint8, cam_color, overlay], axis=1)
    pil_img = Image.fromarray(composite)
    pil_img.save(out_file)
    logger.info("Saved Grad-CAM visualization to %s", out_file)
    return out_file


def explain_image_with_model(
    model: nn.Module,
    architecture: str,
    image_tensor: torch.Tensor,
    target_class_idx: int = 5,
    layer_name: Optional[str] = None,
    target_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Generates Grad-CAM attribution and model prediction for an image in a safe context.

    Args:
        model: Trained model instance.
        architecture: Architecture string ('densenet121', 'efficientnet_b0', etc.).
        image_tensor: Tensor of shape (1, C, H, W).
        target_class_idx: Class index to explain.
        layer_name: Optional explicit layer string.
        target_size: Optional spatial resolution.

    Returns:
        Dict with 'cam', 'probability', 'logit', 'target_class_idx', 'architecture'.
    """
    target_layer = get_target_layer_for_xai(model, architecture, layer_name=layer_name)

    with GradCAMExplainer(model, target_layer) as explainer:
        cam = explainer.generate_cam(image_tensor, target_class_idx=target_class_idx, target_size=target_size)

        with torch.no_grad():
            logits = model(image_tensor)
            if isinstance(logits, tuple):
                logits = logits[0]
            prob = float(torch.sigmoid(logits)[0, target_class_idx].item())
            logit = float(logits[0, target_class_idx].item())

    return {
        "cam": cam,
        "probability": prob,
        "logit": logit,
        "target_class_idx": target_class_idx,
        "architecture": architecture,
    }
