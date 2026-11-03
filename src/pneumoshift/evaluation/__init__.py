"""Evaluation, calibration, threshold transfer, bootstrapping, and domain gap probing."""

from pneumoshift.evaluation.metrics import compute_discrimination_metrics, compute_generalization_gap
from pneumoshift.evaluation.calibration import (
    compute_ece,
    compute_calibration_metrics,
    compute_reliability_diagram_data,
    evaluate_calibration_breakdown,
    TemperatureScaling,
)
from pneumoshift.evaluation.thresholds import (
    compute_binary_metrics_at_threshold,
    compute_optimal_thresholds,
    evaluate_threshold_transfer,
    compute_score_distribution_shift,
    evaluate_threshold_transfer_suite,
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
from pneumoshift.evaluation.config_schemas import (
    validate_model_config,
    validate_train_config,
    validate_arch_comparison_config,
    validate_uncertainty_ablation_config,
)
from pneumoshift.evaluation.architecture_comparison import (
    compare_architecture_generalization_gaps,
    compare_architecture_calibration,
    compare_architecture_domain_divergence,
    generate_architecture_comparison_tables,
    run_architecture_comparison_suite,
)
from pneumoshift.evaluation.uncertainty_ablation import (
    compare_uncertainty_policy_discrimination,
    quantify_probability_overconfidence,
    quantify_threshold_transfer_collapse,
    generate_uncertainty_ablation_tables,
    run_uncertainty_policy_ablation_suite,
)
from pneumoshift.evaluation.domain_gap import (
    train_domain_classifier_probe,
    compute_feature_mmd,
    compute_feature_space_divergence,
    prepare_subgroup_metadata,
    evaluate_subgroups,
    evaluate_subgroup_disparities,
    evaluate_domain_gap_suite,
)

__all__ = [
    "compute_discrimination_metrics",
    "compute_generalization_gap",
    "compute_ece",
    "compute_calibration_metrics",
    "compute_reliability_diagram_data",
    "evaluate_calibration_breakdown",
    "TemperatureScaling",
    "compute_binary_metrics_at_threshold",
    "compute_optimal_thresholds",
    "evaluate_threshold_transfer",
    "compute_score_distribution_shift",
    "evaluate_threshold_transfer_suite",
    "patient_bootstrap_ci",
    "paired_bootstrap_auroc_diff",
    "two_sample_bootstrap_diff",
    "benjamini_hochberg_fdr",
    "train_domain_classifier_probe",
    "compute_feature_mmd",
    "compute_feature_space_divergence",
    "prepare_subgroup_metadata",
    "evaluate_subgroups",
    "evaluate_subgroup_disparities",
    "evaluate_domain_gap_suite",
    "LITERATURE_BENCHMARKS",
    "compare_against_benchmarks",
    "compare_single_vs_multitask",
    "VALID_SHIFT_DIRECTIONS",
    "validate_cross_domain_config",
    "resolve_cross_domain_paths",
    "extract_cross_domain_predictions",
    "compute_generalization_gap_report",
    "validate_model_config",
    "validate_train_config",
    "validate_arch_comparison_config",
    "validate_uncertainty_ablation_config",
    "compare_architecture_generalization_gaps",
    "compare_architecture_calibration",
    "compare_architecture_domain_divergence",
    "generate_architecture_comparison_tables",
    "run_architecture_comparison_suite",
    "compare_uncertainty_policy_discrimination",
    "quantify_probability_overconfidence",
    "quantify_threshold_transfer_collapse",
    "generate_uncertainty_ablation_tables",
    "run_uncertainty_policy_ablation_suite",
]

