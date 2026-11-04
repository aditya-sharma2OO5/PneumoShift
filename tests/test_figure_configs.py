"""Unit tests for Phase 7 publication figure configurations and schema validators."""

from pathlib import Path
import pytest

from pneumoshift.figures.config_schemas import (
    DEFAULT_PRESET_DIMS,
    FigureExportSpec,
    GlobalFigureConfig,
    PlotDimensions,
    PlotStyleConfig,
    get_preset_dimensions,
    load_figure_config,
    validate_dimensions_config,
    validate_export_config,
    validate_figure_spec,
    validate_global_figure_config,
    validate_plot_style,
)


def test_default_yaml_loads_and_validates():
    cfg_path = Path("configs/figures/default.yaml")
    assert cfg_path.is_file(), "configs/figures/default.yaml must exist"
    cfg = load_figure_config(cfg_path)
    assert "style" in cfg
    assert "dimensions" in cfg
    assert "export" in cfg
    assert cfg["export"]["dpi"] >= 300
    assert "pdf" in cfg["export"]["formats"]
    assert "png" in cfg["export"]["formats"]


def test_roc_pr_yaml_loads_and_validates():
    cfg_path = Path("configs/figures/roc_pr.yaml")
    assert cfg_path.is_file()
    cfg = load_figure_config(cfg_path)
    assert cfg["figure_id"] == "fig2_roc_pr_curves"
    assert "roc_panel" in cfg
    assert "pr_panel" in cfg


def test_calibration_yaml_loads_and_validates():
    cfg_path = Path("configs/figures/calibration.yaml")
    assert cfg_path.is_file()
    cfg = load_figure_config(cfg_path)
    assert cfg["figure_id"] == "fig3_calibration_reliability"
    assert cfg["calibration"]["n_bins"] == 10
    assert len(cfg["regimes"]) == 4


def test_thresholds_yaml_loads_and_validates():
    cfg_path = Path("configs/figures/thresholds.yaml")
    assert cfg_path.is_file()
    cfg = load_figure_config(cfg_path)
    assert cfg["figure_id"] == "fig4_threshold_transfer_discordance"
    assert "threshold_sweep" in cfg
    assert "metrics" in cfg


def test_xai_yaml_loads_and_validates():
    cfg_path = Path("configs/figures/xai.yaml")
    assert cfg_path.is_file()
    cfg = load_figure_config(cfg_path)
    assert cfg["figure_id"] == "fig6_explainability_and_shortcuts"
    assert "saliency_heatmaps" in cfg
    assert "faithfulness" in cfg


def test_plot_dimensions_dataclass_valid():
    dims = PlotDimensions(width=7.2, height=4.8)
    assert dims.width == 7.2
    assert dims.height == 4.8


def test_plot_dimensions_dataclass_invalid():
    with pytest.raises(ValueError, match="width must be strictly positive"):
        PlotDimensions(width=0.0, height=4.8)

    with pytest.raises(ValueError, match="height must be strictly positive"):
        PlotDimensions(width=7.2, height=-1.0)


def test_plot_style_config_dataclass_valid():
    style = PlotStyleConfig()
    assert style.font_family == "sans-serif"
    assert style.font_size_title == 11
    assert style.line_width == 1.8


def test_plot_style_config_dataclass_invalid():
    with pytest.raises(ValueError, match="grid_alpha must be in"):
        PlotStyleConfig(grid_alpha=1.5)

    with pytest.raises(ValueError, match="line_width must be positive"):
        PlotStyleConfig(line_width=0.0)

    with pytest.raises(ValueError, match="marker_size must be positive"):
        PlotStyleConfig(marker_size=-2.0)

    with pytest.raises(ValueError, match="font_size_title must be positive"):
        PlotStyleConfig(font_size_title=0)


def test_figure_export_spec_dataclass_valid():
    spec = FigureExportSpec()
    assert spec.dpi == 300
    assert "pdf" in spec.formats


def test_figure_export_spec_dataclass_invalid():
    with pytest.raises(ValueError, match="DPI must be >= 150"):
        FigureExportSpec(dpi=100)

    with pytest.raises(ValueError, match="formats cannot be empty"):
        FigureExportSpec(formats=[])

    with pytest.raises(ValueError, match="Invalid format"):
        FigureExportSpec(formats=["bmp"])


def test_global_figure_config_dataclass():
    cfg = GlobalFigureConfig()
    assert isinstance(cfg.style, PlotStyleConfig)
    assert "single_column" in cfg.dimensions
    assert cfg.export.dpi == 300


def test_validate_plot_style_errors():
    with pytest.raises(ValueError, match="style must be a dictionary"):
        validate_plot_style("not_a_dict")

    with pytest.raises(ValueError, match="grid_alpha must be in"):
        validate_plot_style({"grid_alpha": -0.1})

    with pytest.raises(ValueError, match="line_width must be positive"):
        validate_plot_style({"line_width": -1.0})

    with pytest.raises(ValueError, match="marker_size must be positive"):
        validate_plot_style({"marker_size": 0.0})


def test_validate_dimensions_config_errors():
    with pytest.raises(ValueError, match="dimensions must be a dictionary"):
        validate_dimensions_config([])

    with pytest.raises(ValueError, match="must be a dictionary"):
        validate_dimensions_config({"single_column": "bad"})

    with pytest.raises(ValueError, match="missing 'width' or 'height'"):
        validate_dimensions_config({"single_column": {"width": 3.5}})

    with pytest.raises(ValueError, match="must be strictly positive"):
        validate_dimensions_config({"single_column": {"width": -3.5, "height": 3.2}})


def test_validate_export_config_errors():
    with pytest.raises(ValueError, match="export must be a dictionary"):
        validate_export_config(None)

    with pytest.raises(ValueError, match="dpi must be >= 150"):
        validate_export_config({"dpi": 72})

    with pytest.raises(ValueError, match="formats must be a non-empty list"):
        validate_export_config({"formats": []})

    with pytest.raises(ValueError, match="Invalid format"):
        validate_export_config({"formats": ["tiff"]})


def test_validate_figure_spec_errors():
    with pytest.raises(ValueError, match="Figure spec must be a dictionary"):
        validate_figure_spec(123)

    with pytest.raises(ValueError, match="missing mandatory 'figure_id'"):
        validate_figure_spec({"title": "Test"})

    with pytest.raises(ValueError, match="Invalid layout preset"):
        validate_figure_spec({"figure_id": "test", "layout": {"preset": "nonexistent"}})

    with pytest.raises(ValueError, match="must be >= 1"):
        validate_figure_spec({"figure_id": "test", "layout": {"preset": "single_column", "nrows": 0}})


def test_get_preset_dimensions():
    w, h = get_preset_dimensions("single_column")
    assert (w, h) == DEFAULT_PRESET_DIMS["single_column"]

    w_dh, h_dh = get_preset_dimensions("double_column")
    assert (w_dh, h_dh) == DEFAULT_PRESET_DIMS["double_column"]

    custom_cfg = {"dimensions": {"single_column": {"width": 4.0, "height": 3.5}}}
    w_c, h_c = get_preset_dimensions("single_column", custom_cfg)
    assert (w_c, h_c) == (4.0, 3.5)


def test_load_figure_config_missing_file():
    with pytest.raises(FileNotFoundError):
        load_figure_config("nonexistent_figures_config.yaml")

