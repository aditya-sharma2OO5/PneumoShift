"""Evaluation, calibration, threshold transfer, bootstrapping, and domain gap probing."""

from pneumoshift.evaluation.metrics import compute_discrimination_metrics, compute_generalization_gap
from pneumoshift.evaluation.calibration import (
    compute_ece,
    compute_calibration_metrics,
    TemperatureScaling,
)
from pneumoshift.evaluation.thresholds import (
    compute_optimal_thresholds,
    evaluate_threshold_transfer,
    compute_score_distribution_shift,
)
from pneumoshift.evaluation.ablation import compare_single_vs_multitask
from pneumoshift.evaluation.benchmarks import (
    LITERATURE_BENCHMARKS,
    compare_against_benchmarks,
)
from pneumoshift.evaluation.bootstrap import (
    patient_bootstrap_ci,
    paired_bootstrap_auroc_diff,
    two_sample_bootstrap_diff,
    benjamini_hochberg_fdr,
)
from pneumoshift.evaluation.cross_domain import (
    VALID_SHIFT_DIRECTIONS,
    compute_generalization_gap_report,
    extract_cross_domain_predictions,
    resolve_cross_domain_paths,
    validate_cross_domain_config,
)
from pneumoshift.evaluation.domain_gap import (
    train_domain_classifier_probe,
    compute_feature_mmd,
    evaluate_subgroups,
)

__all__ = [
    "compute_discrimination_metrics",
    "compute_generalization_gap",
    "compute_ece",
    "compute_calibration_metrics",
    "TemperatureScaling",
    "compute_optimal_thresholds",
    "evaluate_threshold_transfer",
    "compute_score_distribution_shift",
    "patient_bootstrap_ci",
    "paired_bootstrap_auroc_diff",
    "two_sample_bootstrap_diff",
    "benjamini_hochberg_fdr",
    "train_domain_classifier_probe",
    "compute_feature_mmd",
    "evaluate_subgroups",
    "LITERATURE_BENCHMARKS",
    "compare_against_benchmarks",
    "compare_single_vs_multitask",
    "VALID_SHIFT_DIRECTIONS",
    "validate_cross_domain_config",
    "resolve_cross_domain_paths",
    "extract_cross_domain_predictions",
    "compute_generalization_gap_report",
]

