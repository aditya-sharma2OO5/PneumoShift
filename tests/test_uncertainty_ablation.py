"""Unit tests for CheXpert uncertainty policy ablation engine (U-Zeros vs U-Ones vs U-Ignore)."""

from pathlib import Path
import tempfile
import numpy as np
import pytest

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.uncertainty_ablation import (
    compare_uncertainty_policy_discrimination,
    generate_uncertainty_ablation_tables,
    quantify_probability_overconfidence,
    quantify_threshold_transfer_collapse,
    run_uncertainty_policy_ablation_suite,
)


@pytest.fixture
def mock_policy_eval_results() -> dict:
    """Mock evaluation results across the three uncertainty policies."""
    return {
        "u_zeros": {
            "in_domain": {
                "macro_auroc": 0.85,
                "per_class": {
                    "Pneumonia": {"auroc": 0.82, "auprc": 0.45},
                    "Atelectasis": {"auroc": 0.84, "auprc": 0.50},
                },
            },
            "cross_domain": {
                "macro_auroc": 0.77,
                "per_class": {
                    "Pneumonia": {"auroc": 0.74, "auprc": 0.35},  # Gap: 0.08
                    "Atelectasis": {"auroc": 0.76, "auprc": 0.40},  # Gap: 0.08
                },
            },
        },
        "u_ones": {
            "in_domain": {
                "macro_auroc": 0.86,
                "per_class": {
                    "Pneumonia": {"auroc": 0.83, "auprc": 0.48},
                    "Atelectasis": {"auroc": 0.85, "auprc": 0.52},
                },
            },
            "cross_domain": {
                "macro_auroc": 0.74,  # Bigger degradation under transfer
                "per_class": {
                    "Pneumonia": {"auroc": 0.71, "auprc": 0.31},  # Gap: 0.12
                    "Atelectasis": {"auroc": 0.72, "auprc": 0.36},  # Gap: 0.13
                },
            },
        },
        "u_ignore": {
            "in_domain": {
                "macro_auroc": 0.84,
                "per_class": {
                    "Pneumonia": {"auroc": 0.81, "auprc": 0.44},
                    "Atelectasis": {"auroc": 0.83, "auprc": 0.49},
                },
            },
            "cross_domain": {
                "macro_auroc": 0.79,  # Narrowest generalization gap
                "per_class": {
                    "Pneumonia": {"auroc": 0.76, "auprc": 0.38},  # Gap: 0.05
                    "Atelectasis": {"auroc": 0.78, "auprc": 0.42},  # Gap: 0.05
                },
            },
        },
    }


class TestUncertaintyPolicyDiscrimination:
    """Verifies generalization gap comparisons across uncertainty policies."""

    def test_compare_discrimination_identifies_best_policy(self, mock_policy_eval_results):
        findings = ["Pneumonia", "Atelectasis"]
        report = compare_uncertainty_policy_discrimination(
            policy_results=mock_policy_eval_results,
            findings=findings,
            shift_direction="chexpert_to_nih",
        )

        assert report["shift_direction"] == "chexpert_to_nih"
        assert set(report["policies_evaluated"]) == {"u_zeros", "u_ones", "u_ignore"}

        # U-Ignore has smallest macro gap (0.05 vs U-Zeros 0.08 and U-Ones 0.12)
        assert report["narrowest_gap_policy"] == "u_ignore"

        # Check Pneumonia gaps
        pneu_gaps = {p: report["per_class"]["Pneumonia"][p]["gap_auroc"] for p in report["policies_evaluated"]}
        assert pytest.approx(pneu_gaps["u_zeros"], abs=1e-3) == 0.08
        assert pytest.approx(pneu_gaps["u_ones"], abs=1e-3) == 0.12
        assert pytest.approx(pneu_gaps["u_ignore"], abs=1e-3) == 0.05


class TestProbabilityOverconfidenceQuantification:
    """Verifies over-prediction ratio (OPR) and overconfidence quantification."""

    def test_overconfidence_and_opr(self):
        np.random.seed(42)
        n_samples = 100
        # True prevalence ~ 10% (0.10)
        y_true = np.zeros((n_samples, 2))
        y_true[:10, 0] = 1.0  # Class 0: 10% prevalence
        y_true[:15, 1] = 1.0  # Class 1: 15% prevalence

        # U-Zeros: well-calibrated around prevalence (mean ~0.10)
        preds_zeros = np.clip(np.random.beta(1, 9, (n_samples, 2)), 0.01, 0.99)

        # U-Ones: heavily inflated predictions (mean ~0.35 -> OPR ~ 3.5x)
        preds_ones = np.clip(np.random.beta(4, 7, (n_samples, 2)), 0.01, 0.99)

        # U-Ignore: conservative predictions (mean ~0.12 -> OPR ~ 1.2x)
        preds_ignore = np.clip(np.random.beta(1.2, 9, (n_samples, 2)), 0.01, 0.99)

        policy_preds = {
            "u_zeros": preds_zeros,
            "u_ones": preds_ones,
            "u_ignore": preds_ignore,
        }

        report = quantify_probability_overconfidence(
            y_true=y_true,
            policy_predictions=policy_preds,
            findings=["Pneumonia", "Atelectasis"],
        )

        pneu = report["per_class"]["Pneumonia"]
        assert pneu["observed_prevalence"] == 0.10
        # U-Ones should exhibit dramatically higher OPR than U-Zeros and U-Ignore
        assert pneu["policies"]["u_ones"]["over_prediction_ratio"] > pneu["policies"]["u_zeros"]["over_prediction_ratio"]
        assert pneu["policies"]["u_ones"]["over_prediction_ratio"] > pneu["policies"]["u_ignore"]["over_prediction_ratio"]
        assert report["macro_metrics"]["u_ones"]["macro_over_prediction_ratio"] > 2.0


class TestThresholdTransferCollapse:
    """Verifies threshold transfer drop, threshold drift, and specificity collapse."""

    def test_threshold_transfer_specificity_collapse(self):
        np.random.seed(42)
        n_src = 80
        n_tgt = 80

        src_true = np.zeros((n_src, 1))
        src_true[:20, 0] = 1.0  # 25% pos

        tgt_true = np.zeros((n_tgt, 1))
        tgt_true[:10, 0] = 1.0  # 12.5% pos (prevalence shift)

        # Source predictions
        src_preds_zeros = np.where(src_true == 1, np.random.uniform(0.6, 0.9, (n_src, 1)), np.random.uniform(0.1, 0.4, (n_src, 1)))
        src_preds_ones = np.where(src_true == 1, np.random.uniform(0.7, 0.95, (n_src, 1)), np.random.uniform(0.4, 0.7, (n_src, 1)))  # higher baseline

        # Target predictions
        tgt_preds_zeros = np.where(tgt_true == 1, np.random.uniform(0.5, 0.8, (n_tgt, 1)), np.random.uniform(0.1, 0.4, (n_tgt, 1)))
        tgt_preds_ones = np.where(tgt_true == 1, np.random.uniform(0.7, 0.95, (n_tgt, 1)), np.random.uniform(0.45, 0.75, (n_tgt, 1)))  # highly shifted up

        src_by_p = {"u_zeros": src_preds_zeros, "u_ones": src_preds_ones}
        tgt_by_p = {"u_zeros": tgt_preds_zeros, "u_ones": tgt_preds_ones}

        collapse_report = quantify_threshold_transfer_collapse(
            source_val_true=src_true,
            source_val_preds_by_policy=src_by_p,
            target_true=tgt_true,
            target_preds_by_policy=tgt_by_p,
            findings=["Pneumonia"],
            threshold_criterion="youden",
        )

        pneu = collapse_report["per_class"]["Pneumonia"]
        assert "source_threshold" in pneu["u_zeros"]
        assert "transferred_specificity" in pneu["u_zeros"]
        assert "specificity_collapse" in pneu["u_zeros"]
        assert "specificity_collapse" in pneu["u_ones"]


class TestUncertaintyAblationSuite:
    """Verifies end-to-end suite orchestrator and report generation."""

    def test_run_suite_and_save_artifacts(self, tmp_path: Path, mock_policy_eval_results):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, (30, 2)).astype(float)
        preds = {
            "u_zeros": np.random.uniform(0.1, 0.9, (30, 2)),
            "u_ones": np.random.uniform(0.3, 0.95, (30, 2)),
            "u_ignore": np.random.uniform(0.1, 0.85, (30, 2)),
        }

        results = run_uncertainty_policy_ablation_suite(
            policy_eval_results=mock_policy_eval_results,
            target_true=y_true,
            target_predictions_by_policy=preds,
            source_val_true=y_true,
            source_val_predictions_by_policy=preds,
            findings=["Pneumonia", "Atelectasis"],
            output_dir=tmp_path,
            shift_direction="chexpert_to_nih",
        )

        assert "table_markdown" in results
        md = results["table_markdown"]
        assert "CheXpert Uncertainty Policy Robustness Ablation" in md
        assert "Cross-Dataset Discrimination" in md
        assert "Probability Overconfidence" in md
        assert "Threshold Transfer Stability" in md

        # Verify disk outputs
        json_file = tmp_path / "uncertainty_ablation_chexpert_to_nih.json"
        md_file = tmp_path / "uncertainty_ablation_chexpert_to_nih.md"
        assert json_file.is_file()
        assert md_file.is_file()

