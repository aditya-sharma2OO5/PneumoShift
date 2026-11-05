"""Unit tests for Phase 7 Quality Gate & Final Deliverables Verification CLI."""

from pathlib import Path
import subprocess
import sys
from typing import Any, Dict
import pytest

from scripts.verify_phase7_gate import (
    check_data_onboarding_tooling,
    check_gallery_dashboard,
    check_manuscript_completeness,
    check_one_click_reproduction,
    check_phase7_configurations,
    check_phase7_guardrails,
    check_publication_figures_integrity,
    run_phase7_quality_gate,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestPhase7QualityGateChecks:
    """Tests each individual gate check function for pass and fail conditions."""

    def test_check_phase7_configurations_pass(self) -> None:
        res = check_phase7_configurations()
        assert res["passed"] is True
        assert res["validated_count"] == 5
        assert len(res["failures"]) == 0

    def test_check_phase7_configurations_fail(self, tmp_path: Path) -> None:
        res = check_phase7_configurations(configs_dir=tmp_path)
        assert res["passed"] is False
        assert len(res["failures"]) > 0

    def test_check_publication_figures_integrity_pass(self) -> None:
        res = check_publication_figures_integrity()
        assert res["passed"] is True
        assert res["verified_files"] == 18
        assert len(res["failures"]) == 0

    def test_check_publication_figures_integrity_fail(self, tmp_path: Path) -> None:
        res = check_publication_figures_integrity(figures_dir=tmp_path)
        assert res["passed"] is False
        assert res["verified_files"] == 0

    def test_check_manuscript_completeness_pass(self) -> None:
        res = check_manuscript_completeness()
        assert res["passed"] is True
        assert res["size_kb"] >= 30.0
        assert res["citation_count"] >= 15
        assert len(res["failures"]) == 0

    def test_check_manuscript_completeness_placeholder_fail(self, tmp_path: Path) -> None:
        bad_doc = tmp_path / "MANUSCRIPT.md"
        bad_doc.write_text("# Title\nTODO: add text here\n[INSERT TABLE]\n", encoding="utf-8")
        res = check_manuscript_completeness(manuscript_path=bad_doc)
        assert res["passed"] is False
        assert any("placeholder" in f.lower() for f in res["failures"])

    def test_check_gallery_dashboard_pass(self) -> None:
        res = check_gallery_dashboard()
        assert res["passed"] is True
        assert res["size_kb"] >= 30.0
        assert len(res["failures"]) == 0

    def test_check_gallery_dashboard_cdn_violation_fail(self, tmp_path: Path) -> None:
        bad_html = tmp_path / "gallery.html"
        bad_html.write_text(
            "<!DOCTYPE html><html><head><script src='https://cdnjs.cloudflare.com/lib.js'></script></head><body>tablesData hypothesesData tab-btn figures-grid</body></html>"
            + (" " * 31000),
            encoding="utf-8",
        )
        res = check_gallery_dashboard(gallery_path=bad_html)
        assert res["passed"] is False
        assert any("zero-cdn" in f.lower() for f in res["failures"])

    def test_check_data_onboarding_tooling_pass(self) -> None:
        res = check_data_onboarding_tooling()
        assert res["passed"] is True
        assert len(res["failures"]) == 0

    def test_check_one_click_reproduction_pass(self) -> None:
        res = check_one_click_reproduction()
        assert res["passed"] is True
        assert len(res["failures"]) == 0

    def test_check_phase7_guardrails_pass(self) -> None:
        res = check_phase7_guardrails()
        assert res["passed"] is True
        assert len(res["failures"]) == 0


class TestPhase7QualityGateEndToEnd:
    """Tests the full orchestrator and CLI entry points."""

    def test_run_phase7_quality_gate_all_passed(self) -> None:
        results = run_phase7_quality_gate()
        assert results["overall_passed"] is True
        for gate_name, gate_res in results["checks"].items():
            assert gate_res["passed"] is True, f"Gate {gate_name} failed: {gate_res.get('message')}"

    def test_cli_execution_success(self) -> None:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "verify_phase7_gate.py")]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert "PNEUMOSHIFT PHASE 7 QUALITY GATE VERDICT: PASSED" in proc.stdout

