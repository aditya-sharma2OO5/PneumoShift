"""Configuration validation schemas and dataclasses for Phase 7 publication figures.

Validates figure layout presets, typography specifications, color palettes,
DPI bounds, and multi-format export parameters (vector PDF/SVG and raster PNG).
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import yaml

from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.figures.config_schemas")

VALID_FORMATS: List[str] = ["pdf", "svg", "png", "eps"]
VALID_PRESETS: List[str] = ["single_column", "one_and_half_column", "double_column", "custom"]
DEFAULT_PRESET_DIMS: Dict[str, Tuple[float, float]] = {
    "single_column": (3.5, 3.2),
    "one_and_half_column": (5.5, 4.0),
    "double_column": (7.2, 4.8),
}


@dataclass
class PlotDimensions:
    """Stores figure canvas width and height in inches."""

    width: float
    height: float

    def __post_init__(self) -> None:
        if self.width <= 0.0:
            raise ValueError(f"Figure width must be strictly positive, got {self.width}")
        if self.height <= 0.0:
            raise ValueError(f"Figure height must be strictly positive, got {self.height}")


@dataclass
class PlotStyleConfig:
    """Stores matplotlib typography, line styles, and color palette parameters."""

    font_family: str = "sans-serif"
    font_sans_serif: List[str] = field(default_factory=lambda: ["DejaVu Sans", "Arial", "Helvetica"])
    font_size_title: int = 11
    font_size_label: int = 10
    font_size_tick: int = 8
    font_size_legend: int = 8
    palette: str = "colorblind"
    primary_color: str = "#0173B2"
    shift_color: str = "#DE8F05"
    secondary_color: str = "#029E73"
    accent_color: str = "#D55E00"
    neutral_color: str = "#7F7F7F"
    grid_alpha: float = 0.30
    line_width: float = 1.8
    marker_size: float = 6.0
    use_latex: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.grid_alpha <= 1.0):
            raise ValueError(f"grid_alpha must be in [0, 1], got {self.grid_alpha}")
        if self.line_width <= 0.0:
            raise ValueError(f"line_width must be positive, got {self.line_width}")
        if self.marker_size <= 0.0:
            raise ValueError(f"marker_size must be positive, got {self.marker_size}")
        for sz_name, sz_val in [
            ("font_size_title", self.font_size_title),
            ("font_size_label", self.font_size_label),
            ("font_size_tick", self.font_size_tick),
            ("font_size_legend", self.font_size_legend),
        ]:
            if sz_val <= 0:
                raise ValueError(f"{sz_name} must be positive, got {sz_val}")


@dataclass
class FigureExportSpec:
    """Stores destination directory, resolution, and format targets for figure saving."""

    output_dir: str = "results/figures"
    dpi: int = 300
    formats: List[str] = field(default_factory=lambda: ["pdf", "svg", "png"])
    bbox_inches: str = "tight"
    transparent: bool = False
    facecolor: str = "white"

    def __post_init__(self) -> None:
        if self.dpi < 150:
            raise ValueError(f"Publication DPI must be >= 150 (standard >= 300), got {self.dpi}")
        if not self.formats:
            raise ValueError("Export formats cannot be empty")
        for fmt in self.formats:
            fmt_clean = fmt.lower().lstrip(".")
            if fmt_clean not in VALID_FORMATS:
                raise ValueError(f"Invalid format '{fmt}'. Expected subset of {VALID_FORMATS}")


@dataclass
class GlobalFigureConfig:
    """Consolidated configuration wrapping style, dimensional presets, and export specifications."""

    style: PlotStyleConfig = field(default_factory=PlotStyleConfig)
    dimensions: Dict[str, PlotDimensions] = field(
        default_factory=lambda: {
            k: PlotDimensions(width=v[0], height=v[1]) for k, v in DEFAULT_PRESET_DIMS.items()
        }
    )
    export: FigureExportSpec = field(default_factory=FigureExportSpec)


def validate_plot_style(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates plot style configuration dictionary against boundary assertions."""
    if not isinstance(cfg, dict):
        raise ValueError(f"style must be a dictionary, got {type(cfg)}")

    grid_alpha = cfg.get("grid_alpha", 0.3)
    if not (0.0 <= float(grid_alpha) <= 1.0):
        raise ValueError(f"grid_alpha must be in [0, 1], got {grid_alpha}")

    line_width = cfg.get("line_width", 1.8)
    if float(line_width) <= 0.0:
        raise ValueError(f"line_width must be positive, got {line_width}")

    marker_size = cfg.get("marker_size", 6.0)
    if float(marker_size) <= 0.0:
        raise ValueError(f"marker_size must be positive, got {marker_size}")

    return cfg


def validate_dimensions_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates figure dimension presets dictionary."""
    if not isinstance(cfg, dict):
        raise ValueError(f"dimensions must be a dictionary, got {type(cfg)}")

    for preset_name, dims in cfg.items():
        if not isinstance(dims, dict):
            raise ValueError(f"Preset '{preset_name}' must be a dictionary with 'width' and 'height'")
        if "width" not in dims or "height" not in dims:
            raise ValueError(f"Preset '{preset_name}' missing 'width' or 'height'")
        w = float(dims["width"])
        h = float(dims["height"])
        if w <= 0.0 or h <= 0.0:
            raise ValueError(f"Preset '{preset_name}' dimensions must be strictly positive: ({w}, {h})")

    return cfg


def validate_export_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates figure export specification dictionary."""
    if not isinstance(cfg, dict):
        raise ValueError(f"export must be a dictionary, got {type(cfg)}")

    dpi = cfg.get("dpi", 300)
    if int(dpi) < 150:
        raise ValueError(f"dpi must be >= 150, got {dpi}")

    formats = cfg.get("formats", ["pdf", "svg", "png"])
    if not isinstance(formats, list) or not formats:
        raise ValueError("formats must be a non-empty list of string formats")

    for fmt in formats:
        fmt_clean = str(fmt).lower().lstrip(".")
        if fmt_clean not in VALID_FORMATS:
            raise ValueError(f"Invalid format '{fmt}'. Expected one of {VALID_FORMATS}")

    return cfg


def validate_global_figure_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates top-level global figures YAML configuration."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Global figure config must be a dictionary, got {type(cfg)}")

    if "style" in cfg:
        validate_plot_style(cfg["style"])

    if "dimensions" in cfg:
        validate_dimensions_config(cfg["dimensions"])

    if "export" in cfg:
        validate_export_config(cfg["export"])

    return cfg


def validate_figure_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates individual figure layout and panel configuration YAML."""
    if not isinstance(cfg, dict):
        raise ValueError(f"Figure spec must be a dictionary, got {type(cfg)}")

    if "figure_id" not in cfg:
        raise ValueError("Figure spec missing mandatory 'figure_id'")

    layout = cfg.get("layout", {})
    if isinstance(layout, dict):
        preset = layout.get("preset", "single_column")
        if preset not in VALID_PRESETS:
            raise ValueError(f"Invalid layout preset '{preset}'. Expected one of {VALID_PRESETS}")
        nrows = int(layout.get("nrows", 1))
        ncols = int(layout.get("ncols", 1))
        if nrows <= 0 or ncols <= 0:
            raise ValueError(f"Layout nrows ({nrows}) and ncols ({ncols}) must be >= 1")

    return cfg


def load_figure_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates a YAML figure configuration file.

    Args:
        config_path: Path to the YAML file.

    Returns:
        Validated dictionary.
    """
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Figure configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if "style" in cfg or "dimensions" in cfg:
        return validate_global_figure_config(cfg)
    return validate_figure_spec(cfg)


def get_preset_dimensions(
    preset_name: str,
    global_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[float, float]:
    """Retrieves width and height in inches for a named layout preset.

    Args:
        preset_name: Preset key ('single_column', 'one_and_half_column', 'double_column').
        global_cfg: Optional loaded global figure config.

    Returns:
        Tuple of (width, height) in inches.
    """
    if global_cfg and "dimensions" in global_cfg:
        dims = global_cfg["dimensions"].get(preset_name)
        if dims and isinstance(dims, dict):
            return float(dims.get("width", 7.2)), float(dims.get("height", 4.8))

    return DEFAULT_PRESET_DIMS.get(preset_name, (7.2, 4.8))

