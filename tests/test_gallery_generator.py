"""Unit tests for Phase 7 interactive HTML results dashboard and gallery generator."""

from pathlib import Path
import pytest

from pneumoshift.figures.gallery import (
    _get_fallback_tables_data,
    build_gallery_html,
    generate_gallery,
)


def test_fallback_tables_data_structure():
    data = _get_fallback_tables_data()
    assert "kpis" in data
    assert "hypotheses" in data
    assert len(data["hypotheses"]) == 5
    for h in data["hypotheses"]:
        assert "id" in h
        assert "name" in h
        assert "raw_p" in h
        assert "fdr_q" in h


def test_build_gallery_html_structure():
    html = build_gallery_html(smoke_test=True)
    assert isinstance(html, str)
    assert html.strip().startswith("<!DOCTYPE html>")
    assert "</html>" in html
    assert "PneumoShift" in html

    # Assert presence of all 5 tabs
    assert 'id="tab-overview"' in html
    assert 'id="tab-figures"' in html
    assert 'id="tab-tables"' in html
    assert 'id="tab-saliency"' in html
    assert 'id="tab-hypotheses"' in html

    # Assert presence of KPI tiles
    assert "Discrimination Gap" in html
    assert "Calibration Escalation" in html
    assert "Threshold Discordance" in html
    assert "Architecture Bias" in html
    assert "Shortcut Energy Surge" in html

    # Assert presence of all 6 publication figure stems
    assert "fig1_study_overview" in html
    assert "fig2_roc_pr_curves" in html
    assert "fig3_calibration_reliability" in html
    assert "fig4_threshold_transfer_discordance" in html
    assert "fig5_architecture_and_uncertainty" in html
    assert "fig6_explainability_and_shortcuts" in html


def test_build_gallery_zero_cdn_dependencies():
    """Validates the architectural guardrail: zero external CDN / network dependencies."""
    html = build_gallery_html(smoke_test=True)

    # Check for external script sources
    assert '<script src="http' not in html.lower()
    assert '<script src="https' not in html.lower()

    # Check for external stylesheet links
    assert '<link rel="stylesheet" href="http' not in html.lower()
    assert '<link rel="stylesheet" href="https' not in html.lower()

    # Must contain self-contained style block
    assert "<style>" in html
    assert "</style>" in html


def test_generate_gallery_persists_file(tmp_path):
    out_file = tmp_path / "reports" / "gallery.html"
    saved = generate_gallery(output_file=out_file, smoke_test=True)

    assert saved.is_file()
    assert saved == out_file
    # Must be non-empty and reasonably sized (embedded css/js/tables > 15 KB)
    assert saved.stat().st_size > 15 * 1024

    with open(saved, "r", encoding="utf-8") as f:
        content = f.read()
    assert "<!DOCTYPE html>" in content
    assert "masterDataTable" in content


def test_gallery_with_live_master_tables(tmp_path):
    # Using existing results/tables/master directory
    out_file = tmp_path / "gallery.html"
    saved = generate_gallery(
        output_file=out_file,
        tables_dir="results/tables/master",
        figures_dir="results/figures",
        smoke_test=False,
    )
    assert saved.is_file()
    with open(saved, "r", encoding="utf-8") as f:
        content = f.read()
    assert "table1_cross_domain_discrimination" in content or "tablesData" in content


def test_saliency_comparator_elements():
    html = build_gallery_html(smoke_test=True)
    # Check interactive controls
    assert 'id="camOpacity"' in html
    assert 'id="toggleBBox"' in html
    assert 'id="toggleShortcut"' in html
    assert "in-domain-cam" in html
    assert "cross-domain-cam" in html
    assert "Pointing Game:" in html

