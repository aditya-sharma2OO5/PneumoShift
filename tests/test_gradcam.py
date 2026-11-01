"""Tests for Grad-CAM output shape, normalization, and hook cleanup."""

import numpy as np
import pytest
import torch
from pneumoshift.models.factory import build_model, get_target_layer_for_xai
from pneumoshift.xai.gradcam import GradCAMExplainer


def test_gradcam_output_shape_and_bounds():
    model = build_model(architecture="densenet121", num_classes=8, pretrained=False)
    model.eval()

    target_layer = get_target_layer_for_xai(model, "densenet121")
    explainer = GradCAMExplainer(model, target_layer)

    dummy_image = torch.randn(1, 3, 64, 64, requires_grad=True)
    cam = explainer.generate_cam(dummy_image, target_class_idx=5, target_size=(64, 64))

    assert isinstance(cam, np.ndarray)
    assert cam.shape == (64, 64)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0

    explainer.release_hooks()

