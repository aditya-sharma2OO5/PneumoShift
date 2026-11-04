"""Unit tests for Phase 6 Results Ingestion and Harmonization Engine."""

from pathlib import Path
import tempfile
import pytest

from pneumoshift.aggregation.collector import (
    build_synthetic_manifest,
    collect_all_results,
    ingest_phase3_results,
    ingest_phase4_results,
    ingest_phase5_results,
    save_run_manifest,
    validate_manifest_completeness,
)
from pneumoshift.utils.io import load_json, save_json


class TestResultsCollector:
    """Verifies results discovery, ingestion, normalization, and manifest saving."""

    def test_ingest_phase3_existing(self):
        res = ingest_phase3_results("results/tables/cross_domain")
        assert res["available"] is True
        assert "nih_to_chexpert" in res["directions"]
        assert "chexpert_to_nih" in res["directions"]

    def test_ingest_phase3_nonexistent_dir(self):
        res = ingest_phase3_results("nonexistent/cross_domain")
        assert res["available"] is False
        assert res["directions"] == {}

    def test_ingest_phase4_existing(self):
        res = ingest_phase4_results("results/tables/phase4_ablations")
        assert res["available"] is True
        assert bool(res["master_report"]) or bool(res["architecture_comparison"])

    def test_ingest_phase4_nonexistent_dir(self):
        res = ingest_phase4_results("nonexistent/phase4")
        assert res["available"] is False

    def test_ingest_phase5_existing(self):
        res = ingest_phase5_results("results/tables/xai")
        assert res["available"] is True
        assert bool(res["master_report"]) or bool(res["faithfulness"])

    def test_ingest_phase5_nonexistent_dir(self):
        res = ingest_phase5_results("nonexistent/xai")
        assert res["available"] is False

    def test_build_synthetic_manifest(self):
        manifest = build_synthetic_manifest()
        assert manifest["metadata"]["mode"] == "synthetic_smoke_test"
        assert manifest["phase3_cross_domain"]["available"] is True
        assert manifest["phase4_ablations"]["available"] is True
        assert manifest["phase5_xai"]["available"] is True

        # Verify key hypotheses data presence
        nih_to_chx = manifest["phase3_cross_domain"]["directions"]["nih_to_chexpert"]
        assert nih_to_chx["densenet121"]["generalization_gap"] > 0
        assert nih_to_chx["densenet121"]["calibration"]["oracle_recovery_pct"] > 40.0

        p4 = manifest["phase4_ablations"]
        assert p4["architecture_comparison"]["more_robust"] == "EfficientNet-B0"
        assert p4["uncertainty_ablation"]["u_ones"]["over_pred_ratio"] > 2.0

        p5 = manifest["phase5_xai"]
        assert p5["sanity_checks"]["spearman_drop_pct"] >= 75.0
        assert p5["faithfulness"]["in_domain_fdr"] > p5["faithfulness"]["transferred_fdr"]

    def test_collect_all_results_smoke_test(self):
        manifest = collect_all_results(smoke_test=True)
        assert manifest["metadata"]["mode"] == "synthetic_smoke_test"
        checks = validate_manifest_completeness(manifest)
        assert checks["phase3_cross_domain"] is True
        assert checks["phase4_ablations"] is True
        assert checks["phase5_xai"] is True
        assert checks["metadata_valid"] is True

    def test_collect_all_results_real_mode(self):
        manifest = collect_all_results(
            cross_domain_dir="results/tables/cross_domain",
            phase4_dir="results/tables/phase4_ablations",
            xai_dir="results/tables/xai",
            smoke_test=False,
        )
        assert manifest["metadata"]["mode"] in ["real_data", "synthetic_smoke_test"]
        checks = validate_manifest_completeness(manifest)
        assert checks["metadata_valid"] is True

    def test_collect_all_results_fallback(self):
        # All invalid dirs should fall back gracefully to synthetic
        manifest = collect_all_results(
            cross_domain_dir="nonexistent/p3",
            phase4_dir="nonexistent/p4",
            xai_dir="nonexistent/p5",
            smoke_test=False,
        )
        assert manifest["metadata"]["mode"] == "synthetic_smoke_test"

    def test_save_run_manifest(self):
        manifest = build_synthetic_manifest()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "test_manifest.json"
            saved_p = save_run_manifest(manifest, out_file)
            assert saved_p.is_file()
            loaded = load_json(saved_p)
            assert loaded["metadata"]["mode"] == "synthetic_smoke_test"

    def test_validate_manifest_completeness_missing_phase(self):
        partial_manifest = {
            "metadata": {"timestamp": "2026-11-05T00:00:00Z"},
            "phase3_cross_domain": {"available": True, "directions": {"nih_to_chexpert": {}}},
            "phase4_ablations": {"available": False},
            "phase5_xai": {"available": False},
        }
        checks = validate_manifest_completeness(partial_manifest)
        assert checks["metadata_valid"] is True
        assert checks["phase3_cross_domain"] is True
        assert checks["phase4_ablations"] is False
        assert checks["phase5_xai"] is False

