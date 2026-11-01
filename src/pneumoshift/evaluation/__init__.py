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
from pneumoshift.evaluation.benchmarks import (
    LITERATURE_BENCHMARKS,
    compare_against_benchmarks,
)
from pneumoshift.evaluation.bootstrap import (
    patient_bootstrap_ci,
    paired_bootstrap_auroc_diff,
    benjamini_hochberg_fdr,
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
    "benjamini_hochberg_fdr",
    "train_domain_classifier_probe",
    "compute_feature_mmd",
    "evaluate_subgroups",
    "LITERATURE_BENCHMARKS",
    "compare_against_benchmarks",
]

