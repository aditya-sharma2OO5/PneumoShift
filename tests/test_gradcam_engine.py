"""Unit tests for multi-architecture Grad-CAM attribution engine and overlay visualizations."""

from pathlib import Path
import numpy as np
from PIL import Image
import pytest
import torch

from pneumoshift.models.factory import build_model, get_target_layer_for_xai
from pneumoshift.xai.gradcam import (
    GradCAMExplainer,
    overlay_cam_on_image,
    save_cam_visualization,
    explain_image_with_model,
)


@pytest.fixture
def densenet_model():
    model = build_model(architecture="densenet121", num_classes=8, pretrained=False)
    model.eval()
    return model


@pytest.fixture
def efficientnet_model():
    model = build_model(architecture="efficientnet_b0", num_classes=8, pretrained=False)
    model.eval()
    return model


class TestGradCAMMultiArchitecture:
    """Verifies Grad-CAM attribution generation across DenseNet121 and EfficientNet-B0."""

    def test_densenet121_attribution(self, densenet_model):
        target_layer = get_target_layer_for_xai(densenet_model, "densenet121")
        explainer = GradCAMExplainer(densenet_model, target_layer)

        img = torch.randn(1, 3, 64, 64)
        cam = explainer.generate_cam(img, target_class_idx=5)
        explainer.release_hooks()

        assert isinstance(cam, np.ndarray)
        assert cam.shape == (64, 64)
        assert 0.0 <= cam.min() <= cam.max() <= 1.0

    def test_efficientnet_b0_attribution(self, efficientnet_model):
        target_layer = get_target_layer_for_xai(efficientnet_model, "efficientnet_b0")
        explainer = GradCAMExplainer(efficientnet_model, target_layer)

        img = torch.randn(1, 3, 64, 64)
        cam = explainer.generate_cam(img, target_class_idx=5)
        explainer.release_hooks()

        assert isinstance(cam, np.ndarray)
        assert cam.shape == (64, 64)
        assert 0.0 <= cam.min() <= cam.max() <= 1.0

    def test_explicit_layer_name_targeting(self, densenet_model, efficientnet_model):
        layer_dn = get_target_layer_for_xai(densenet_model, "densenet121", layer_name="features.denseblock4")
        assert layer_dn is not None

        layer_eff = get_target_layer_for_xai(efficientnet_model, "efficientnet_b0", layer_name="features.8")
        assert layer_eff is not None


class TestGradCAMBatchAndMultiClass:
    """Tests multi-sample batch processing and multi-disease class heatmaps."""

    def test_batch_attribution_generation(self, densenet_model):
        target_layer = get_target_layer_for_xai(densenet_model, "densenet121")
        with GradCAMExplainer(densenet_model, target_layer) as explainer:
            batch_img = torch.randn(3, 3, 48, 48)
            cams = explainer.generate_cam(batch_img, target_class_idx=5)

            assert isinstance(cams, np.ndarray)
            assert cams.shape == (3, 48, 48)
            for b in range(3):
                assert 0.0 <= cams[b].min() <= cams[b].max() <= 1.0

    def test_multi_class_attribution(self, densenet_model):
        target_layer = get_target_layer_for_xai(densenet_model, "densenet121")
        with GradCAMExplainer(densenet_model, target_layer) as explainer:
            img = torch.randn(1, 3, 48, 48)
            multi_cams = explainer.generate_multi_class_cam(img, target_class_indices=[0, 5])

            assert set(multi_cams.keys()) == {0, 5}
            assert multi_cams[0].shape == (48, 48)
            assert multi_cams[5].shape == (48, 48)


class TestGradCAMContextAndHookManagement:
    """Tests context manager and safe model training mode restoration."""

    def test_context_manager_hook_release(self, densenet_model):
        target_layer = get_target_layer_for_xai(densenet_model, "densenet121")
        with GradCAMExplainer(densenet_model, target_layer) as explainer:
            assert not explainer._hooks_released
        assert explainer._hooks_released

    def test_training_mode_restored_after_cam(self, densenet_model):
        densenet_model.train()
        assert densenet_model.training

        target_layer = get_target_layer_for_xai(densenet_model, "densenet121")
        with GradCAMExplainer(densenet_model, target_layer) as explainer:
            img = torch.randn(2, 3, 32, 32)
            _ = explainer.generate_cam(img, target_class_idx=5)

        # Must be restored to training mode
        assert densenet_model.training


class TestOverlayAndVisualization:
    """Tests alpha blending, colormaps, and side-by-side composite generation."""

    def test_overlay_cam_on_image_tensor(self):
        img_tensor = torch.randn(3, 64, 64)
        cam = np.random.rand(64, 64).astype(np.float32)

        overlay = overlay_cam_on_image(img_tensor, cam, colormap="jet", alpha=0.5)
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == (64, 64, 3)
        assert overlay.dtype == np.uint8
        assert 0 <= overlay.min() <= overlay.max() <= 255

    def test_overlay_cam_spatial_resizing(self):
        # Image is 64x64, CAM is 32x32
        img = np.random.rand(64, 64)
        cam = np.random.rand(32, 32).astype(np.float32)

        overlay = overlay_cam_on_image(img, cam, colormap="turbo", alpha=0.4)
        assert overlay.shape == (64, 64, 3)

    def test_save_cam_visualization_side_by_side(self, tmp_path: Path):
        img = torch.randn(3, 48, 48)
        cam = np.random.rand(48, 48).astype(np.float32)

        out_file = tmp_path / "xai" / "test_cam_panel.png"
        res = save_cam_visualization(out_file, img, cam, colormap="jet", side_by_side=True)

        assert res.is_file()
        loaded = Image.open(res)
        # Width should be 3 * 48 = 144
        assert loaded.size == (144, 48)

    def test_save_cam_visualization_overlay_only(self, tmp_path: Path):
        img = np.random.rand(48, 48)
        cam = np.random.rand(48, 48).astype(np.float32)

        out_file = tmp_path / "overlay_only.png"
        res = save_cam_visualization(out_file, img, cam, side_by_side=False)

        assert res.is_file()
        loaded = Image.open(res)
        assert loaded.size == (48, 48)


class TestExplainImageWithModel:
    """Tests high-level single-function explanation pipeline."""

    def test_explain_image_with_densenet(self, densenet_model):
        img = torch.randn(1, 3, 48, 48)
        result = explain_image_with_model(densenet_model, "densenet121", img, target_class_idx=5)

        assert "cam" in result
        assert "probability" in result
        assert "logit" in result
        assert result["architecture"] == "densenet121"
        assert result["cam"].shape == (48, 48)
        assert 0.0 <= result["probability"] <= 1.0

    def test_explain_image_with_efficientnet(self, efficientnet_model):
        img = torch.randn(1, 3, 48, 48)
        result = explain_image_with_model(efficientnet_model, "efficientnet_b0", img, target_class_idx=5)

        assert "cam" in result
        assert "probability" in result
        assert result["architecture"] == "efficientnet_b0"
        assert result["cam"].shape == (48, 48)
        assert 0.0 <= result["probability"] <= 1.0

