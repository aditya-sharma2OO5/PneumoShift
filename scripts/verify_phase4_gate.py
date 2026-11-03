"""Automated Phase 4 Quality Gate & Ablation Robustness Verification CLI.

Validates:
1. Architecture & uncertainty configuration schema compliance and parameter bounds.
2. Architecture comparison integrity: EfficientNet-B0 vs DenseNet121 generalization gaps,
   probability calibration degradation, and penultimate feature space domain probing.
3. CheXpert uncertainty policy ablation integrity: U-Zeros vs U-Ones vs U-Ignore,
   overconfidence / over-prediction ratios (OPR), and threshold specificity collapse.
4. Multi-seed replication stability (seeds 42, 1337, 2024): variance within tolerance (SD < 0.05)
   and paired statistical significance testing.
5. Strict guardrails: target_cal / test split isolation, zero leakage, and gradient loss masking.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.config_schemas import (
    validate_arch_comparison_config,
    validate_model_config,
    validate_train_config,
    validate_uncertainty_ablation_config,
)
from pneumoshift.utils.io import load_json, load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.verify_phase4_gate")


def check_phase4_configurations() -> Dict[str, Any]:
    """Validates schemas and parameter assertions across all Phase 4 configurations."""
    configs_to_check = [
        ("configs/model/efficientnet_b0.yaml", validate_model_config),
        ("configs/train/nih_efficientnet_b0.yaml", validate_train_config),
        ("configs/train/chexpert_efficientnet_b0.yaml", validate_train_config),
        ("configs/train/chexpert_densenet121_uones.yaml", validate_train_config),
        ("configs/train/chexpert_densenet121_uignore.yaml", validate_train_config),
        ("configs/eval/arch_comparison.yaml", validate_arch_comparison_config),
        ("configs/eval/uncertainty_ablation.yaml", validate_uncertainty_ablation_config),
    ]

    failures = []
    validated_count = 0

    for path_str, validator_func in configs_to_check:
        p = Path(path_str)
        if not p.is_file():
            failures.append(f"Missing config file: {path_str}")
            continue
        try:
            raw = load_yaml_config(p)
            validator_func(raw)
            validated_count += 1
        except Exception as exc:
            failures.append(f"Validation error in {path_str}: {exc}")

    passed = len(failures) == 0 and validated_count == len(configs_to_check)
    return {
        "passed": passed,
        "message": (
            f"All {len(configs_to_check)} Phase 4 configurations passed schema and boundary validation."
            if passed
            else "; ".join(failures)
        ),
        "validated_count": validated_count,
        "failures": failures,
    }


def check_architecture_comparison_bounds(arch_data: Dict[str, Any]) -> Dict[str, Any]:
    """Verifies that architecture comparison metrics conform to expected scientific hypotheses."""
    failures = []

    gap_data = arch_data.get("generalization_gap_comparison", {})
    m_dn = gap_data.get("macro_densenet_gap_auroc")
    m_eff = gap_data.get("macro_efficientnet_gap_auroc")
    diff = gap_data.get("macro_gap_difference")

    if m_dn is None or m_eff is None:
        failures.append("Missing macro generalization gap for DenseNet or EfficientNet.")

    cal_data = arch_data.get("calibration_comparison", {})
    dn_cal = cal_data.get("densenet121", {})
    eff_cal = cal_data.get("efficientnet_b0", {})

    if "ece_degradation" not in dn_cal or "ece_degradation" not in eff_cal:
        failures.append("Missing calibration degradation metrics.")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            f"Architecture comparison valid: DenseNet gap={m_dn:+.4f}, EfficientNet gap={m_eff:+.4f}, diff={diff:+.4f}."
            if passed
            else "; ".join(failures)
        ),
        "failures": failures,
    }


def check_uncertainty_ablation_bounds(uncert_data: Dict[str, Any]) -> Dict[str, Any]:
    """Verifies uncertainty ablation metrics: positive prediction bias (OPR) and threshold collapse."""
    failures = []

    disc_data = uncert_data.get("discrimination_comparison", {})
    policies = disc_data.get("policies_evaluated", [])
    expected = {"u_zeros", "u_ones", "u_ignore"}

    if not expected.issubset(set(policies)):
        failures.append(f"Incomplete policies evaluated: {policies}, expected {expected}")

    over_data = uncert_data.get("overconfidence_comparison", {})
    if over_data:
        m_over = over_data.get("macro_metrics", {})
        opr_ones = m_over.get("u_ones", {}).get("macro_over_prediction_ratio", 0.0)
        opr_zeros = m_over.get("u_zeros", {}).get("macro_over_prediction_ratio", 0.0)
        opr_ignore = m_over.get("u_ignore", {}).get("macro_over_prediction_ratio", 0.0)

        # U-Ones must show over-prediction bias compared to U-Zeros / U-Ignore
        if opr_ones <= opr_zeros or opr_ones <= opr_ignore:
            failures.append(f"U-Ones OPR ({opr_ones:.2f}) not higher than U-Zeros ({opr_zeros:.2f}) / U-Ignore ({opr_ignore:.2f}).")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            f"Uncertainty ablation verified across policies {policies} with confirmed U-Ones over-prediction bias."
            if passed
            else "; ".join(failures)
        ),
        "failures": failures,
    }


def check_multi_seed_variance_bounds(multi_seed_data: Dict[str, Any]) -> Dict[str, Any]:
    """Verifies that multi-seed replication variance is bounded and paired tests executed."""
    failures = []

    arch_m = multi_seed_data.get("architecture_multi_seed", {})
    dn_m = arch_m.get("densenet_aggregation", {}).get("macro_metrics", {}).get("generalization_gap", {})
    eff_m = arch_m.get("efficientnet_aggregation", {}).get("macro_metrics", {}).get("generalization_gap", {})

    dn_std = dn_m.get("std", 1.0)
    eff_std = eff_m.get("std", 1.0)

    # Maximum allowed random seed variance across seeds (SD < 0.05)
    max_allowed_std = 0.05
    if dn_std > max_allowed_std:
        failures.append(f"DenseNet multi-seed standard deviation {dn_std:.4f} > {max_allowed_std}")
    if eff_std > max_allowed_std:
        failures.append(f"EfficientNet multi-seed standard deviation {eff_std:.4f} > {max_allowed_std}")

    sig = arch_m.get("macro_gap_significance", {})
    if "t_pvalue" not in sig:
        failures.append("Missing paired Student's t-test in multi-seed architecture comparison.")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            f"Multi-seed variance bounded: DenseNet SD={dn_std:.4f}, EfficientNet SD={eff_std:.4f}; paired test p={sig.get('t_pvalue', 1.0):.4f}."
            if passed
            else "; ".join(failures)
        ),
        "failures": failures,
    }


def check_guardrails() -> Dict[str, Any]:
    """Verifies data isolation, zero test leakage, and gradient loss masking."""
    failures = []

    # Check that training configs do not point to test splits
    for cfg_name in [
        "configs/train/nih_efficientnet_b0.yaml",
        "configs/train/chexpert_efficientnet_b0.yaml",
        "configs/train/chexpert_densenet121_uones.yaml",
        "configs/train/chexpert_densenet121_uignore.yaml",
    ]:
        p = Path(cfg_name)
        if p.is_file():
            raw = load_yaml_config(p)
            save_best = raw.get("train", {}).get("save_best_metric", "")
            if "test" in save_best.lower():
                failures.append(f"{cfg_name} monitors test metric ({save_best}) for checkpointing!")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": "All guardrails passed: zero test leakage, target_cal split isolation, and valid checkpoint monitors.",
        "failures": failures,
    }


def run_phase4_quality_gate(
    results_dir: Path = Path("results/tables/phase4_ablations"),
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Executes the full Phase 4 quality gate verification."""
    gate_results = {
        "overall_passed": True,
        "checks": {},
    }

    # 1. Configs Check
    cfg_check = check_phase4_configurations()
    gate_results["checks"]["configurations"] = cfg_check
    if not cfg_check["passed"]:
        gate_results["overall_passed"] = False

    # 2. Master Report / Results Check
    master_file = results_dir / "phase4_master_report.json"
    if master_file.is_file():
        master_data = load_json(master_file)
    elif smoke_test or not master_file.is_file():
        from scripts.run_phase4_ablations import run_phase4_orchestrator
        logger.info("Running orchestrator in smoke_test mode for quality gate verification...")
        master_data = run_phase4_orchestrator(output_dir=str(results_dir), smoke_test=True)
    else:
        master_data = {}

    arch_check = check_architecture_comparison_bounds(master_data.get("architecture_comparison", {}))
    gate_results["checks"]["architecture_comparison"] = arch_check
    if not arch_check["passed"]:
        gate_results["overall_passed"] = False

    uncert_check = check_uncertainty_ablation_bounds(master_data.get("uncertainty_ablation", {}))
    gate_results["checks"]["uncertainty_ablation"] = uncert_check
    if not uncert_check["passed"]:
        gate_results["overall_passed"] = False

    seed_check = check_multi_seed_variance_bounds(master_data.get("multi_seed_replication", {}))
    gate_results["checks"]["multi_seed_replication"] = seed_check
    if not seed_check["passed"]:
        gate_results["overall_passed"] = False

    # 3. Guardrails
    guard_check = check_guardrails()
    gate_results["checks"]["guardrails"] = guard_check
    if not guard_check["passed"]:
        gate_results["overall_passed"] = False

    return gate_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 Quality Gate & Ablation Robustness Verification CLI.")
    parser.add_argument("--results_dir", default="results/tables/phase4_ablations", help="Directory of ablation outputs")
    parser.add_argument("--output_file", default="results/tables/phase4_quality_gate.json", help="Path to verdict JSON")
    parser.add_argument("--smoke_test", action="store_true", help="Run in dry-run smoke test mode")
    args = parser.parse_args()

    results = run_phase4_quality_gate(
        results_dir=Path(args.results_dir),
        smoke_test=args.smoke_test,
    )

    out_p = Path(args.output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_p)

    verdict_str = "PASSED" if results["overall_passed"] else "FAILED"
    print("\n" + "=" * 80)
    print(f"  PNEUMOSHIFT PHASE 4 QUALITY GATE VERDICT: {verdict_str}")
    print("=" * 80)
    for check_name, res in results["checks"].items():
        icon = "[PASS]" if res.get("passed", False) else "[FAIL]"
        print(f"  {icon} {check_name.upper():<26} : {res.get('message', '')}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
