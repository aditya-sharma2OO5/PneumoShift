"""Unit tests for clinical bounding box localization, anatomical lung masks, and shortcut energy analysis."""

from pathlib import Path
import numpy as np
import pytest
import torch

from pneumoshift.utils.io import load_json
from pneumoshift.xai.localization import (
    compute_pointing_game_and_iou,
    evaluate_image_localization,
    evaluate_cohort_localization,
    compare_cross_domain_localization,
)
from pneumoshift.xai.lung_mask import (
    generate_anatomical_lung_mask,
    extract_adaptive_lung_mask,
    compute_in_lung_attribution_ratio,
    evaluate_cohort_lung_plausibility,
)
from pneumoshift.xai.shortcut_analysis import (
    compute_shortcut_energy_ratio,
    evaluate_cohort_shortcut_energy,
    compare_shortcut_escalation,
    format_localization_and_shortcuts_table,
    save_localization_and_shortcuts_report,
)


class TestBoundingBoxLocalization:
    """Verifies Pointing Game hit detection, Energy in Box, and multi-threshold IoU."""

    def test_single_bbox_peak_inside(self):
        saliency = np.zeros((100, 100), dtype=np.float32)
        saliency[40:60, 40:60] = 0.9
        saliency[50, 50] = 1.0  # Peak is at (50, 50)

        bbox = (30, 30, 40, 40)  # x_min=30, y_min=30, w=40, h=40
        res = compute_pointing_game_and_iou(saliency, bbox, binarize_threshold=0.5)

        assert res["pointing_game_hit"] is True
        assert res["energy_in_box"] > 0.95
        assert res["iou"] > 0.0

    def test_single_bbox_peak_outside(self):
        saliency = np.zeros((100, 100), dtype=np.float32)
        saliency[10, 10] = 1.0  # Peak at (10, 10)

        bbox = (50, 50, 30, 30)  # Box at [50, 80] x [50, 80]
        res = compute_pointing_game_and_iou(saliency, bbox, binarize_threshold=0.5)

        assert res["pointing_game_hit"] is False
        assert res["energy_in_box"] == pytest.approx(0.0)

    def test_multi_bbox_evaluation(self):
        saliency = np.zeros((100, 100), dtype=np.float32)
        # Two foci: left lower and right upper
        saliency[70:80, 20:30] = 0.8
        saliency[75, 25] = 1.0  # Peak

        bboxes = [
            (60, 10, 20, 20),   # Box 1
            (15, 65, 20, 20),   # Box 2: contains (25, 75)
        ]

        res = evaluate_image_localization(saliency, bboxes, binarize_thresholds=[0.3, 0.5, 0.7])

        assert res["pointing_game_hit"] is True
        assert res["energy_in_box"] > 0.9
        assert "iou_0.3" in res["iou_by_threshold"]
        assert "iou_0.5" in res["iou_by_threshold"]
        assert "iou_0.7" in res["iou_by_threshold"]
        assert res["mean_iou"] > 0.0

    def test_empty_bboxes_graceful(self):
        saliency = np.ones((50, 50), dtype=np.float32)
        res = evaluate_image_localization(saliency, bboxes=[])

        assert res["pointing_game_hit"] is False
        assert res["energy_in_box"] == 0.0
        assert res["mean_iou"] == 0.0
        assert res["num_bboxes"] == 0

    def test_cohort_localization_evaluation(self):
        saliency_maps = [np.zeros((100, 100), dtype=np.float32) for _ in range(4)]
        bboxes_list = []

        for i, m in enumerate(saliency_maps):
            if i < 3:
                m[45:55, 45:55] = 1.0
                bboxes_list.append([(40, 40, 20, 20)])
            else:
                m[10:15, 10:15] = 1.0
                bboxes_list.append([(70, 70, 20, 20)])

        res = evaluate_cohort_localization(saliency_maps, bboxes_list, binarize_thresholds=[0.3, 0.5])

        assert res["num_samples"] == 4
        assert res["pointing_game_hits"] == 3
        assert res["pointing_game_hit_rate"] == 0.75
        assert res["mean_energy_in_box"] > 0.70
        assert "iou_0.3" in res["mean_iou_by_threshold"]

    def test_cross_domain_localization_comparison(self):
        in_domain = {
            "num_samples": 20,
            "pointing_game_hits": 18,
            "pointing_game_hit_rate": 0.90,
            "mean_energy_in_box": 0.82,
            "overall_mean_iou": 0.65,
            "sample_results": [{"energy_in_box": 0.85, "mean_iou": 0.68} for _ in range(20)],
        }
        transferred = {
            "num_samples": 20,
            "pointing_game_hits": 8,
            "pointing_game_hit_rate": 0.40,
            "mean_energy_in_box": 0.35,
            "overall_mean_iou": 0.22,
            "sample_results": [{"energy_in_box": 0.35, "mean_iou": 0.22} for _ in range(20)],
        }

        comp = compare_cross_domain_localization(in_domain, transferred)

        assert comp["pointing_game_in_domain"] == 0.90
        assert comp["pointing_game_transferred"] == 0.40
        assert comp["pointing_game_p_value"] < 0.05
        assert comp["significant_pointing_collapse"] is True
        assert comp["hypothesis_5d_supported"] is True
        assert "SUPPORTED" in comp["verdict"]


class TestLungMaskPlausibility:
    """Verifies anatomical lung field segmentation and in-lung attribution quantification."""

    def test_generate_anatomical_lung_mask(self):
        mask = generate_anatomical_lung_mask((128, 128))
        assert mask.shape == (128, 128)
        assert mask.dtype == np.float32
        assert set(np.unique(mask)).issubset({0.0, 1.0})

        # Lung fields must occupy a reasonable thoracic fraction (between 20% and 55%)
        coverage = float(mask.sum() / mask.size)
        assert 0.20 <= coverage <= 0.55

    def test_extract_adaptive_lung_mask_tensor(self):
        img_tensor = torch.zeros(1, 1, 64, 64)
        mask = extract_adaptive_lung_mask(img_tensor)
        assert mask.shape == (64, 64)
        assert mask.sum() > 0

    def test_in_lung_attribution_pure_lung(self):
        lung_mask = generate_anatomical_lung_mask((100, 100))
        # Place attribution completely inside lungs
        saliency = lung_mask.copy()
        res = compute_in_lung_attribution_ratio(saliency, lung_mask)

        assert res["in_lung_ratio"] == pytest.approx(1.0)
        assert res["out_lung_ratio"] == pytest.approx(0.0)

    def test_in_lung_attribution_pure_mediastinum(self):
        lung_mask = generate_anatomical_lung_mask((100, 100))
        # Place attribution strictly in midline cardiac/mediastinum strip
        saliency = np.zeros((100, 100), dtype=np.float32)
        saliency[40:60, 48:52] = 1.0

        res = compute_in_lung_attribution_ratio(saliency, lung_mask)
        assert res["in_lung_ratio"] == pytest.approx(0.0)
        assert res["mediastinum_ratio"] > 0.90
        assert res["out_lung_ratio"] > 0.90

    def test_cohort_lung_plausibility(self):
        maps = [np.ones((64, 64), dtype=np.float32) for _ in range(5)]
        res = evaluate_cohort_lung_plausibility(maps)

        assert res["num_samples"] == 5
        assert 0.0 <= res["mean_in_lung_ratio"] <= 1.0
        assert len(res["in_lung_ratios"]) == 5


class TestShortcutAnalysis:
    """Verifies detection of border, corner, and lateral radiopaque marker shortcuts."""

    def test_central_pathology_low_shortcut_energy(self):
        saliency = np.zeros((100, 100), dtype=np.float32)
        saliency[45:55, 45:55] = 1.0  # Center of image

        res = compute_shortcut_energy_ratio(saliency, border_fraction=0.05, corner_fraction=0.08)

        assert res["border_energy_ratio"] == pytest.approx(0.0)
        assert res["corner_energy_ratio"] == pytest.approx(0.0)
        assert res["total_shortcut_energy_ratio"] == pytest.approx(0.0)
        assert res["central_clinical_energy_ratio"] == pytest.approx(1.0)

    def test_corner_text_marker_high_shortcut_energy(self):
        saliency = np.zeros((100, 100), dtype=np.float32)
        # Saliency in top-left corner
        saliency[0:5, 0:5] = 1.0

        res = compute_shortcut_energy_ratio(saliency, border_fraction=0.05, corner_fraction=0.08)

        assert res["corner_energy_ratio"] == pytest.approx(1.0)
        assert res["total_shortcut_energy_ratio"] == pytest.approx(1.0)
        assert res["central_clinical_energy_ratio"] == pytest.approx(0.0)

    def test_cohort_shortcut_energy_evaluation(self):
        maps = [np.random.rand(64, 64).astype(np.float32) for _ in range(4)]
        res = evaluate_cohort_shortcut_energy(maps)

        assert res["num_samples"] == 4
        assert 0.0 <= res["mean_total_shortcut_energy_ratio"] <= 1.0
        assert 0.0 <= res["mean_central_clinical_energy_ratio"] <= 1.0
        assert len(res["total_shortcut_ratios"]) == 4

    def test_compare_shortcut_escalation_hypothesis_5e(self):
        # In-domain: low shortcut energy (mean 5%)
        in_domain_shortcuts = [0.03, 0.04, 0.05, 0.06, 0.05, 0.04]
        # Transferred: high shortcut energy (mean 35%)
        transferred_shortcuts = [0.32, 0.35, 0.40, 0.28, 0.38, 0.36]

        in_domain_lung = [0.85, 0.88, 0.82, 0.90, 0.86, 0.84]
        transferred_lung = [0.45, 0.50, 0.42, 0.52, 0.48, 0.46]

        comp = compare_shortcut_escalation(
            in_domain_shortcut_ratios=in_domain_shortcuts,
            transferred_shortcut_ratios=transferred_shortcuts,
            in_domain_in_lung_ratios=in_domain_lung,
            transferred_in_lung_ratios=transferred_lung,
        )

        assert comp["mean_shortcut_in_domain"] < 0.10
        assert comp["mean_shortcut_transferred"] > 0.30
        assert comp["p_value_shortcut_escalation"] < 0.05
        assert comp["significant_shortcut_escalation"] is True
        assert comp["hypothesis_5e_supported"] is True
        assert "SUPPORTED" in comp["verdict"]


class TestLocalizationAndShortcutReporting:
    """Verifies Markdown formatting and file exports for localization & shortcut reports."""

    def test_format_localization_and_shortcuts_table(self):
        loc_res = {
            "num_samples": 50,
            "pointing_game_hits": 41,
            "pointing_game_hit_rate": 0.82,
            "mean_energy_in_box": 0.742,
            "std_energy_in_box": 0.081,
            "overall_mean_iou": 0.584,
            "std_mean_iou": 0.062,
            "mean_iou_by_threshold": {"iou_0.3": 0.65, "iou_0.5": 0.58, "iou_0.7": 0.52},
        }
        sc_res = {
            "mean_central_clinical_energy_ratio": 0.884,
            "std_central_clinical_energy_ratio": 0.042,
            "mean_total_shortcut_energy_ratio": 0.116,
            "std_total_shortcut_energy_ratio": 0.042,
            "mean_border_energy_ratio": 0.065,
            "std_border_energy_ratio": 0.021,
            "mean_corner_energy_ratio": 0.045,
            "std_corner_energy_ratio": 0.018,
            "mean_lateral_marker_energy_ratio": 0.051,
            "std_lateral_marker_energy_ratio": 0.020,
        }
        comp_res = {
            "verdict": "SUPPORTED: Cross-domain transfer significantly escalates shortcut energy.",
            "mean_shortcut_in_domain": 0.116,
            "mean_shortcut_transferred": 0.342,
            "relative_escalation_percent": 194.8,
            "p_value_shortcut_escalation": 0.0001,
        }

        md = format_localization_and_shortcuts_table(loc_res, sc_res, comp_res)

        assert "# Clinical Localization Benchmarking & Shortcut Energy Quantification" in md
        assert "## 1. Ground-Truth Bounding Box Localization" in md
        assert "Pointing Game Hit Rate" in md
        assert "## 2. Anatomical Plausibility & Peripheral Shortcut Energy Breakdown" in md
        assert "## 3. Cross-Domain Transfer Shortcut Escalation Audit" in md
        assert "Hypothesis 5E Verdict" in md

    def test_save_localization_and_shortcuts_report(self, tmp_path):
        data = {
            "localization": {"num_samples": 10, "pointing_game_hit_rate": 0.8},
            "shortcuts": {"mean_total_shortcut_energy_ratio": 0.12},
        }

        target_json = tmp_path / "test_report.json"
        saved_json, saved_md = save_localization_and_shortcuts_report(data, target_json)

        assert saved_json.is_file()
        assert saved_md.is_file()

        loaded = load_json(saved_json)
        assert "localization" in loaded
        assert loaded["localization"]["pointing_game_hit_rate"] == 0.8

        md_content = saved_md.read_text(encoding="utf-8")
        assert "Clinical Localization Benchmarking" in md_content

