"""Unit tests for multi-seed replication suite and statistical significance engine."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.multi_seed import (
    aggregate_seed_metrics,
    compare_multi_seed_architectures,
    compare_multi_seed_uncertainty_policies,
    compute_sample_ci_95,
    generate_multi_seed_tables,
    paired_seed_significance_test,
    run_multi_seed_replication_suite,
)


class TestConfidenceIntervalsAndPairedTesting:
    """Verifies sample statistics, Student's t CIs, and paired hypothesis tests."""

    def test_compute_sample_ci_95_correctness(self):
        vals = [0.80, 0.82, 0.84]
        res = compute_sample_ci_95(vals)

        assert pytest.approx(res["mean"], abs=1e-4) == 0.82
        assert pytest.approx(res["std"], abs=1e-4) == 0.02
        assert res["n_seeds"] == 3
        assert res["ci_lower"] < res["mean"] < res["ci_upper"]

    def test_compute_sample_ci_95_single_or_empty(self):
        res_single = compute_sample_ci_95([0.75])
        assert res_single["mean"] == 0.75
        assert res_single["std"] == 0.0
        assert res_single["ci_lower"] == 0.75

        res_empty = compute_sample_ci_95([])
        assert np.isnan(res_empty["mean"])
        assert res_empty["n_seeds"] == 0

    def test_paired_seed_significance_test_significant(self):
        # A is consistently higher than B across all seeds
        vals_a = [0.15, 0.14, 0.16]
        vals_b = [0.08, 0.07, 0.09]
        res = paired_seed_significance_test(vals_a, vals_b)

        assert pytest.approx(res["mean_diff"], abs=1e-3) == 0.07
        assert res["t_pvalue"] < 0.05
        assert res["is_significant_ttest"] is True
        assert res["n_pairs"] == 3

    def test_paired_seed_significance_test_non_significant(self):
        # Fluctuates around zero
        vals_a = [0.10, 0.12, 0.09]
        vals_b = [0.11, 0.09, 0.10]
        res = paired_seed_significance_test(vals_a, vals_b)

        assert res["t_pvalue"] > 0.05
        assert res["is_significant_ttest"] is False

    def test_paired_mismatched_length_raises(self):
        with pytest.raises(ValueError, match="Mismatched paired sample lengths"):
            paired_seed_significance_test([0.1, 0.2], [0.1])


@pytest.fixture
def mock_densenet_multi_seed() -> dict:
    """Mock evaluations across 3 seeds for DenseNet121."""
    return {
        42: {
            "macro_auroc": 0.85,
            "ece_uniform": 0.09,
            "per_class": {
                "Pneumonia": {"auroc": 0.80},
                "Atelectasis": {"auroc": 0.84},
            },
            "cross_domain": {
                "macro_auroc": 0.74,  # Gap: 0.11
                "ece_uniform": 0.14,
                "per_class": {
                    "Pneumonia": {"auroc": 0.69},  # Gap: 0.11
                    "Atelectasis": {"auroc": 0.73},  # Gap: 0.11
                },
            },
        },
        1337: {
            "macro_auroc": 0.84,
            "ece_uniform": 0.088,
            "per_class": {
                "Pneumonia": {"auroc": 0.79},
                "Atelectasis": {"auroc": 0.83},
            },
            "cross_domain": {
                "macro_auroc": 0.73,  # Gap: 0.11
                "ece_uniform": 0.142,
                "per_class": {
                    "Pneumonia": {"auroc": 0.68},  # Gap: 0.11
                    "Atelectasis": {"auroc": 0.72},  # Gap: 0.11
                },
            },
        },
        2024: {
            "macro_auroc": 0.86,
            "ece_uniform": 0.092,
            "per_class": {
                "Pneumonia": {"auroc": 0.81},
                "Atelectasis": {"auroc": 0.85},
            },
            "cross_domain": {
                "macro_auroc": 0.74,  # Gap: 0.12
                "ece_uniform": 0.145,
                "per_class": {
                    "Pneumonia": {"auroc": 0.69},  # Gap: 0.12
                    "Atelectasis": {"auroc": 0.73},  # Gap: 0.12
                },
            },
        },
    }


@pytest.fixture
def mock_efficientnet_multi_seed() -> dict:
    """Mock evaluations across 3 seeds for EfficientNet-B0 (consistently narrower gap)."""
    return {
        42: {
            "macro_auroc": 0.84,
            "ece_uniform": 0.08,
            "per_class": {
                "Pneumonia": {"auroc": 0.79},
                "Atelectasis": {"auroc": 0.83},
            },
            "cross_domain": {
                "macro_auroc": 0.78,  # Gap: 0.06
                "ece_uniform": 0.11,
                "per_class": {
                    "Pneumonia": {"auroc": 0.74},  # Gap: 0.05
                    "Atelectasis": {"auroc": 0.77},  # Gap: 0.06
                },
            },
        },
        1337: {
            "macro_auroc": 0.83,
            "ece_uniform": 0.078,
            "per_class": {
                "Pneumonia": {"auroc": 0.78},
                "Atelectasis": {"auroc": 0.82},
            },
            "cross_domain": {
                "macro_auroc": 0.77,  # Gap: 0.06
                "ece_uniform": 0.108,
                "per_class": {
                    "Pneumonia": {"auroc": 0.73},  # Gap: 0.05
                    "Atelectasis": {"auroc": 0.76},  # Gap: 0.06
                },
            },
        },
        2024: {
            "macro_auroc": 0.85,
            "ece_uniform": 0.082,
            "per_class": {
                "Pneumonia": {"auroc": 0.80},
                "Atelectasis": {"auroc": 0.84},
            },
            "cross_domain": {
                "macro_auroc": 0.79,  # Gap: 0.06
                "ece_uniform": 0.112,
                "per_class": {
                    "Pneumonia": {"auroc": 0.75},  # Gap: 0.05
                    "Atelectasis": {"auroc": 0.78},  # Gap: 0.06
                },
            },
        },
    }


class TestMultiSeedAggregations:
    """Verifies multi-seed metric aggregation and paired architecture tests."""

    def test_aggregate_seed_metrics(self, mock_densenet_multi_seed):
        findings = ["Pneumonia", "Atelectasis"]
        agg = aggregate_seed_metrics(mock_densenet_multi_seed, findings=findings)

        assert agg["n_seeds"] == 3
        pneu_in = agg["per_class"]["Pneumonia"]["in_domain_auroc"]
        assert pytest.approx(pneu_in["mean"], abs=1e-3) == 0.80
        assert pneu_in["std"] > 0.0

        pneu_gap = agg["per_class"]["Pneumonia"]["generalization_gap"]
        assert pytest.approx(pneu_gap["mean"], abs=1e-3) == 0.1133

    def test_compare_multi_seed_architectures(self, mock_densenet_multi_seed, mock_efficientnet_multi_seed):
        findings = ["Pneumonia", "Atelectasis"]
        comp = compare_multi_seed_architectures(
            densenet_seed_results=mock_densenet_multi_seed,
            efficientnet_seed_results=mock_efficientnet_multi_seed,
            findings=findings,
            shift_direction="nih_to_chexpert",
        )

        assert comp["shift_direction"] == "nih_to_chexpert"
        pneu_comp = comp["per_class_comparison"]["Pneumonia"]
        # DenseNet mean gap ~0.113 vs EfficientNet ~0.050 -> diff > 0
        assert pneu_comp["significance_test"]["mean_diff"] > 0.05
        assert pneu_comp["significance_test"]["is_significant_ttest"] is True

        macro_sig = comp["macro_gap_significance"]
        assert macro_sig["mean_diff"] > 0.0
        assert macro_sig["is_significant_ttest"] is True


class TestMultiSeedUncertaintyComparison:
    """Verifies multi-seed comparisons across CheXpert uncertainty policies."""

    def test_compare_multi_seed_uncertainty_policies(self, mock_densenet_multi_seed):
        # Create variations of mock_densenet_multi_seed for u_ones and u_ignore
        zeros = mock_densenet_multi_seed
        # U-Ones: larger gaps
        ones = {
            s: {
                "in_domain": {"macro_auroc": d["macro_auroc"]},
                "cross_domain": {"macro_auroc": d["cross_domain"]["macro_auroc"] - 0.04},
            }
            for s, d in zeros.items()
        }
        # U-Ignore: smaller gaps
        ignore = {
            s: {
                "in_domain": {"macro_auroc": d["macro_auroc"]},
                "cross_domain": {"macro_auroc": d["cross_domain"]["macro_auroc"] + 0.03},
            }
            for s, d in zeros.items()
        }

        policy_results = {
            "u_zeros": zeros,
            "u_ones": ones,
            "u_ignore": ignore,
        }

        comp = compare_multi_seed_uncertainty_policies(
            policy_seed_results=policy_results,
            baseline_policy="u_zeros",
            findings=["Pneumonia", "Atelectasis"],
            shift_direction="chexpert_to_nih",
        )

        assert "u_ones_vs_u_zeros" in comp["paired_comparisons_to_baseline"]
        assert "u_ignore_vs_u_zeros" in comp["paired_comparisons_to_baseline"]

        # U-Ones gap is larger than U-Zeros (mean_diff > 0)
        ones_sig = comp["paired_comparisons_to_baseline"]["u_ones_vs_u_zeros"]["gap_significance"]
        assert ones_sig["mean_diff"] > 0.0


class TestMultiSeedSuiteExecution:
    """Verifies end-to-end multi-seed replication suite execution and report emission."""

    def test_run_multi_seed_replication_suite(
        self,
        tmp_path: Path,
        mock_densenet_multi_seed,
        mock_efficientnet_multi_seed,
    ):
        findings = ["Pneumonia", "Atelectasis"]
        results = run_multi_seed_replication_suite(
            densenet_seed_results=mock_densenet_multi_seed,
            efficientnet_seed_results=mock_efficientnet_multi_seed,
            findings=findings,
            shift_direction="nih_to_chexpert",
            output_dir=tmp_path,
        )

        assert "table_markdown" in results
        md = results["table_markdown"]
        assert "Multi-Seed Statistical Replication" in md
        assert "DenseNet121 Gap" in md
        assert "EfficientNet-B0 Gap" in md
        assert "Pneumonia" in md
        assert "Macro Average" in md

        # Verify disk outputs
        json_file = tmp_path / "multi_seed_replication_nih_to_chexpert.json"
        md_file = tmp_path / "multi_seed_replication_nih_to_chexpert.md"
        assert json_file.is_file()
        assert md_file.is_file()

