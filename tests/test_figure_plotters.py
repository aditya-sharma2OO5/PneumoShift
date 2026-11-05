"""Unit tests for Phase 7 publication figures generation engine and exporters."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from pneumoshift.figures.plotters import (
    _synthetic_pr_curve,
    _synthetic_roc_curve,
    apply_publication_theme,
    export_figure,
    generate_all_figures,
    plot_fig1_study_overview,
    plot_fig2_roc_pr_curves,
    plot_fig3_calibration_reliability,
    plot_fig4_threshold_transfer_discordance,
    plot_fig5_architecture_and_uncertainty,
    plot_fig6_explainability_and_shortcuts,
)


def test_apply_publication_theme():
    apply_publication_theme({"font_size_title": 12, "font_size_label": 10})
    assert plt.rcParams["axes.titlesize"] == 12
    assert plt.rcParams["axes.labelsize"] == 10
    assert plt.rcParams["pdf.fonttype"] == 42


def test_synthetic_curve_helpers():
    fpr, tpr = _synthetic_roc_curve(target_auc=0.75, n_points=50)
    assert len(fpr) == 50
    assert len(tpr) == 50
    assert np.all(np.diff(fpr) >= 0), "FPR must be monotonically non-decreasing"
    assert np.all(np.diff(tpr) >= 0), "TPR must be monotonically non-decreasing"

    rec, prec = _synthetic_pr_curve(target_auprc=0.35, n_points=50)
    assert len(rec) == 50
    assert len(prec) == 50
    assert np.all(prec >= 0.0) and np.all(prec <= 1.0)


def test_plot_fig1_study_overview():
    fig = plot_fig1_study_overview()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 1
    plt.close(fig)


def test_plot_fig2_roc_pr_curves():
    fig = plot_fig2_roc_pr_curves()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    # Verify titles
    assert "ROC" in fig.axes[0].get_title()
    assert "Precision-Recall" in fig.axes[1].get_title()
    plt.close(fig)


def test_plot_fig3_calibration_reliability():
    fig = plot_fig3_calibration_reliability()
    assert isinstance(fig, plt.Figure)
    # 4 main subpanels + 4 inset axes = 8 total axes
    assert len(fig.axes) >= 4
    plt.close(fig)


def test_plot_fig4_threshold_transfer_discordance():
    fig = plot_fig4_threshold_transfer_discordance()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 2
    assert "In-Domain" in fig.axes[0].get_title()
    assert "Cross-Domain" in fig.axes[1].get_title()
    plt.close(fig)


def test_plot_fig5_architecture_and_uncertainty():
    fig = plot_fig5_architecture_and_uncertainty()
    assert isinstance(fig, plt.Figure)
    # 2 subpanels + 1 twin axis = 3 axes
    assert len(fig.axes) >= 2
    plt.close(fig)


def test_plot_fig6_explainability_and_shortcuts():
    fig = plot_fig6_explainability_and_shortcuts()
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 4
    plt.close(fig)


def test_export_figure(tmp_path):
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot([0, 1], [0, 1], label="Test")
    ax.legend()

    saved = export_figure(fig, figure_stem="test_figure", output_dir=tmp_path, formats=["pdf", "svg", "png"])
    plt.close(fig)

    assert "pdf" in saved
    assert "svg" in saved
    assert "png" in saved

    for fmt, p in saved.items():
        assert p.is_file(), f"{p} should exist"
        assert p.stat().st_size > 0, f"{p} should be non-empty"


def test_generate_all_figures_smoke_test(tmp_path):
    results = generate_all_figures(
        output_dir=tmp_path,
        formats=["pdf", "svg", "png"],
        smoke_test=True,
    )

    expected_stems = [
        "fig1_study_overview",
        "fig2_roc_pr_curves",
        "fig3_calibration_reliability",
        "fig4_threshold_transfer_discordance",
        "fig5_architecture_and_uncertainty",
        "fig6_explainability_and_shortcuts",
    ]

    assert len(results) == len(expected_stems)
    for stem in expected_stems:
        assert stem in results, f"Missing figure: {stem}"
        for fmt in ["pdf", "svg", "png"]:
            assert fmt in results[stem]
            p = results[stem][fmt]
            assert p.is_file(), f"File {p} should exist"
            assert p.stat().st_size > 0, f"File {p} should not be 0 bytes"

