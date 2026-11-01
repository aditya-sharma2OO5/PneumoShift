"""Self-contained, robust Grad-CAM implementation with forward/backward hook management."""

from typing import Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMExplainer:
    """Computes Gradient-weighted Class Activation Mapping (Grad-CAM)."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_hook = self.target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)

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
            input_tensor: Image tensor of shape (1, C, H, W).
            target_class_idx: Index of target class (e.g. Pneumonia).
            target_size: Output spatial resolution (H, W). Defaults to input spatial dimensions.

        Returns:
            np.ndarray heatmap of shape (H, W) scaled to [0, 1].
        """
        self.model.zero_grad()
        if target_size is None:
            target_size = (input_tensor.shape[2], input_tensor.shape[3])

        logits = self.model(input_tensor)
        score = logits[:, target_class_idx]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture gradients or activations.")

        # Channel-wise mean of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=target_size, mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Min-max normalization
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def release_hooks(self) -> None:
        """Removes PyTorch hooks."""
        self._forward_hook.remove()
        self._backward_hook.remove()
