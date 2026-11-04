"""Configuration validation schemas and data models for Phase 6 Results Aggregation.

Validates master aggregation configurations, clustered bootstrap parameters,
formal hypothesis testing specifications, and master publication table definitions.
"""

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.aggregation.config_schemas")

VALID_BOOTSTRAP_METHODS: List[str] = [
    "patient_clustered",
    "stratified",
    "standard",
]

VALID_CORRECTION_METHODS: List[str] = [
    "benjamini_hochberg",
    "bonferroni",
    "holm",
]

VALID_EXPORT_FORMATS: List[str] = [
    "csv",
    "json",
    "md",
    "tex",
]

VALID_TEST_TYPES: List[str] = [
    "paired_bootstrap",
    "two_sample_bootstrap",
    "delong",
    "wilcoxon",
    "mann_whitney",
    "fisher_exact",
    "paired_t_test",
]

VALID_ALTERNATIVES: List[str] = [
    "two-sided",
    "greater",
    "less",
]


@dataclass
class BootstrapCIResult:
    """Stores non-parametric bootstrap point estimate and confidence bounds."""

    point_estimate: float
    ci_lower: float
    ci_upper: float
    standard_error: float = 0.0
    n_resamples: int = 1000
    confidence_level: float = 0.95
    method: str = "patient_clustered"

    def __post_init__(self) -> None:
        if self.ci_lower > self.ci_upper:
            raise ValueError(
                f"ci_lower ({self.ci_lower}) cannot be strictly greater than ci_upper ({self.ci_upper})"
            )
        if not (0.0 < self.confidence_level < 1.0):
            raise ValueError(f"confidence_level must be in (0, 1), got {self.confidence_level}")
        if self.n_resamples < 1:
            raise ValueError(f"n_resamples must be >= 1, got {self.n_resamples}")

    def format_ci(self, precision: int = 3) -> str:
        """Formats the result as 'point_estimate [ci_lower, ci_upper]'."""
        fmt = f"{{:.{precision}f}} [{{:.{precision}f}}, {{:.{precision}f}}]"
        return fmt.format(self.point_estimate, self.ci_lower, self.ci_upper)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HypothesisTestRecord:
    """Stores structured record of an empirical hypothesis test and FDR correction."""

    hypothesis_id: str
    name: str
    null_hypothesis: str
    alternative: str
    test_type: str
    effect_size: float
    raw_p_value: float
    fdr_q_value: float
    is_significant: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.raw_p_value <= 1.0):
            raise ValueError(f"raw_p_value must be in [0, 1], got {self.raw_p_value}")
        if not (0.0 <= self.fdr_q_value <= 1.0):
            raise ValueError(f"fdr_q_value must be in [0, 1], got {self.fdr_q_value}")
        if self.alternative not in VALID_ALTERNATIVES:
            raise ValueError(f"Invalid alternative: '{self.alternative}'. Expected one of {VALID_ALTERNATIVES}")
        if self.test_type not in VALID_TEST_TYPES:
            raise ValueError(f"Invalid test_type: '{self.test_type}'. Expected one of {VALID_TEST_TYPES}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MasterTableSpec:
    """Specification for a publication master table."""

    id: str
    name: str
    title: str
    description: str
    columns: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregationConfig:
    """Master configuration dataclass for Phase 6 results aggregation."""

    experiment_name: str = "master_results_aggregation"
    description: str = ""
    bootstrap: Dict[str, Any] = field(default_factory=dict)
    hypothesis_testing: Dict[str, Any] = field(default_factory=dict)
    paths: Dict[str, str] = field(default_factory=dict)
    exports: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.experiment_name or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def validate_bootstrap_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates clustered bootstrap parameters."""
    n_resamples = cfg.get("n_resamples", 1000)
    if not isinstance(n_resamples, int) or n_resamples < 10:
        raise ValueError(f"n_resamples must be an integer >= 10, got {n_resamples}")

    confidence_level = cfg.get("confidence_level", 0.95)
    if not isinstance(confidence_level, (int, float)) or not (0.0 < confidence_level < 1.0):
        raise ValueError(f"confidence_level must be in (0, 1), got {confidence_level}")

    seed = cfg.get("seed", 42)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"seed must be a non-negative integer, got {seed}")

    method = str(cfg.get("method", "patient_clustered")).lower()
    if method not in VALID_BOOTSTRAP_METHODS:
        raise ValueError(f"Invalid bootstrap method: '{method}'. Expected one of {VALID_BOOTSTRAP_METHODS}")

    cluster_by = cfg.get("cluster_by", "patient_id")
    if not isinstance(cluster_by, str) or not cluster_by.strip():
        raise ValueError("cluster_by must be a non-empty string identifier")

    return cfg


def validate_hypothesis_testing_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates hypothesis testing and multiple testing correction parameters."""
    fdr_alpha = cfg.get("fdr_alpha", 0.05)
    if not isinstance(fdr_alpha, (int, float)) or not (0.0 < fdr_alpha < 1.0):
        raise ValueError(f"fdr_alpha must be in (0, 1), got {fdr_alpha}")

    fw_alpha = cfg.get("family_wise_alpha", 0.05)
    if not isinstance(fw_alpha, (int, float)) or not (0.0 < fw_alpha < 1.0):
        raise ValueError(f"family_wise_alpha must be in (0, 1), got {fw_alpha}")

    methods = cfg.get("correction_methods", ["benjamini_hochberg"])
    if not isinstance(methods, list) or not methods:
        raise ValueError("correction_methods must be a non-empty list of strings")
    for m in methods:
        if str(m).lower() not in VALID_CORRECTION_METHODS:
            raise ValueError(f"Invalid correction method: '{m}'. Expected one of {VALID_CORRECTION_METHODS}")

    default_test = str(cfg.get("default_test_type", "paired_bootstrap")).lower()
    if default_test not in VALID_TEST_TYPES:
        raise ValueError(f"Invalid default_test_type: '{default_test}'. Expected one of {VALID_TEST_TYPES}")

    return cfg


def validate_export_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates export formatting parameters."""
    formats = cfg.get("formats", ["csv", "json", "md", "tex"])
    if not isinstance(formats, list) or not formats:
        raise ValueError("formats must be a non-empty list")
    for fmt in formats:
        if str(fmt).lower() not in VALID_EXPORT_FORMATS:
            raise ValueError(f"Invalid export format: '{fmt}'. Expected one of {VALID_EXPORT_FORMATS}")

    precision = cfg.get("precision", 4)
    if not isinstance(precision, int) or precision < 0:
        raise ValueError(f"precision must be a non-negative integer, got {precision}")

    latex_booktabs = cfg.get("latex_booktabs", True)
    if not isinstance(latex_booktabs, bool):
        raise ValueError(f"latex_booktabs must be a boolean, got {latex_booktabs}")

    return cfg


def validate_aggregation_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates master results aggregation configuration.

    Args:
        cfg: Raw configuration dictionary.

    Returns:
        Validated configuration dictionary.

    Raises:
        ValueError: If mandatory parameters are missing or out of valid bounds.
    """
    if "experiment_name" not in cfg:
        raise ValueError("Aggregation config missing mandatory key: 'experiment_name'")

    if "bootstrap" in cfg:
        validate_bootstrap_config(cfg["bootstrap"])

    if "hypothesis_testing" in cfg:
        validate_hypothesis_testing_config(cfg["hypothesis_testing"])

    if "exports" in cfg:
        validate_export_config(cfg["exports"])

    paths = cfg.get("paths", {})
    if not isinstance(paths, dict):
        raise ValueError("paths must be a dictionary of directory paths")

    return cfg


def validate_hypothesis_tests_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates formal hypothesis specifications YAML."""
    hypotheses = cfg.get("hypotheses", [])
    if not isinstance(hypotheses, list) or not hypotheses:
        raise ValueError("Hypothesis specification missing non-empty 'hypotheses' list")

    seen_ids = set()
    for h in hypotheses:
        if "id" not in h or "name" not in h or "null_hypothesis" not in h:
            raise ValueError("Each hypothesis must define 'id', 'name', and 'null_hypothesis'")
        hid = h["id"]
        if hid in seen_ids:
            raise ValueError(f"Duplicate hypothesis id detected: '{hid}'")
        seen_ids.add(hid)

        test_type = str(h.get("test_type", "paired_bootstrap")).lower()
        if test_type not in VALID_TEST_TYPES:
            raise ValueError(f"Invalid test_type in hypothesis {hid}: '{test_type}'. Expected one of {VALID_TEST_TYPES}")

        alternative = str(h.get("alternative", "two-sided")).lower()
        if alternative not in VALID_ALTERNATIVES:
            raise ValueError(f"Invalid alternative in hypothesis {hid}: '{alternative}'. Expected one of {VALID_ALTERNATIVES}")

    return cfg


def validate_master_tables_spec(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Validates master table layout specifications YAML."""
    tables = cfg.get("tables", [])
    if not isinstance(tables, list) or not tables:
        raise ValueError("Master tables specification missing non-empty 'tables' list")

    seen_ids = set()
    for t in tables:
        if "id" not in t or "title" not in t or "columns" not in t:
            raise ValueError("Each table spec must define 'id', 'title', and 'columns'")
        tid = t["id"]
        if tid in seen_ids:
            raise ValueError(f"Duplicate table id detected: '{tid}'")
        seen_ids.add(tid)

        cols = t["columns"]
        if not isinstance(cols, list) or not cols:
            raise ValueError(f"Table '{tid}' must define a non-empty list of 'columns'")

    return cfg


def load_aggregation_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads and validates aggregation configuration from YAML file.

    Args:
        config_path: Path to YAML configuration.

    Returns:
        Validated configuration dictionary.
    """
    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Aggregation configuration file not found: {p}")

    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return validate_aggregation_config(cfg)
