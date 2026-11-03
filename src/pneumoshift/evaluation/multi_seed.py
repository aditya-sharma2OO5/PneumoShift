"""Multi-seed replication suite and statistical significance engine.

Aggregates evaluations across fixed random seeds (42, 1337, 2024), computing:
  - Mean ± standard deviation, standard error, and Student's t 95% confidence intervals.
  - Paired hypothesis testing (paired Student's t-test and Wilcoxon signed-rank test).
  - Architecture comparisons (DenseNet121 vs EfficientNet-B0) across seeds.
  - Uncertainty policy ablations (U-Zeros vs U-Ones vs U-Ignore) across seeds.
  - Publication-ready multi-seed Markdown tables.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import scipy.stats as stats

from pneumoshift.data.harmonize_labels import HEADLINE_CLASS, SHARED_CLASSES
from pneumoshift.evaluation.cross_domain import _extract_metric_value
from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.multi_seed")

DEFAULT_SEEDS = [42, 1337, 2024]


def compute_sample_ci_95(values: Sequence[float]) -> Dict[str, float]:
    """Computes mean, standard deviation, SEM, and 95% confidence interval.

    Uses Student's t-distribution with df = N - 1.

    Args:
        values: Sequence of numeric measurements across seeds.

    Returns:
        Dict with 'mean', 'std', 'sem', 'ci_lower', 'ci_upper', and 'n_seeds'.
    """
    clean_vals = [float(v) for v in values if v is not None and not np.isnan(v)]
    n = len(clean_vals)

    if n == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "n_seeds": 0,
        }

    mean_val = float(np.mean(clean_vals))
    if n == 1:
        return {
            "mean": mean_val,
            "std": 0.0,
            "sem": 0.0,
            "ci_lower": mean_val,
            "ci_upper": mean_val,
            "n_seeds": 1,
        }

    std_val = float(np.std(clean_vals, ddof=1))
    sem_val = float(std_val / np.sqrt(n))
    t_crit = float(stats.t.ppf(0.975, df=n - 1))
    ci_lower = float(mean_val - t_crit * sem_val)
    ci_upper = float(mean_val + t_crit * sem_val)

    return {
        "mean": mean_val,
        "std": std_val,
        "sem": sem_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_seeds": n,
    }


def paired_seed_significance_test(
    values_a: Sequence[float],
    values_b: Sequence[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Executes paired hypothesis testing between two configurations matched across seeds.

    H0: Mean difference μ_d = 0 (μ_A == μ_B).
    H1: Mean difference μ_d != 0 (two-sided).

    Computes both paired Student's t-test and Wilcoxon signed-rank test.

    Args:
        values_a: Sequence of measurements for Configuration A across seeds.
        values_b: Sequence of measurements for Configuration B across matching seeds.
        alpha: Significance threshold (default: 0.05).

    Returns:
        Structured dict with mean difference, t-statistic, t-pvalue, Wilcoxon stats,
        and statistical significance flags.
    """
    a_arr = np.array(values_a, dtype=float)
    b_arr = np.array(values_b, dtype=float)

    if len(a_arr) != len(b_arr):
        raise ValueError(f"Mismatched paired sample lengths: {len(a_arr)} vs {len(b_arr)}")

    diffs = a_arr - b_arr
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0

    # Paired Student's t-test
    if np.allclose(diffs, diffs[0]):
        # Exactly identical differences: zero variance
        t_stat = 0.0 if np.isclose(diffs[0], 0.0) else float("inf")
        t_pval = 1.0 if np.isclose(diffs[0], 0.0) else 0.0
    else:
        try:
            t_res = stats.ttest_rel(a_arr, b_arr)
            t_stat = float(t_res.statistic)
            t_pval = float(t_res.pvalue)
        except Exception:
            t_stat = float("nan")
            t_pval = float("nan")

    # Wilcoxon signed-rank test
    try:
        if np.all(diffs == 0):
            w_stat = 0.0
            w_pval = 1.0
        else:
            w_res = stats.wilcoxon(a_arr, b_arr, zero_method="wilcox")
            w_stat = float(w_res.statistic)
            w_pval = float(w_res.pvalue)
    except Exception:
        w_stat = float("nan")
        w_pval = float("nan")

    is_sig_t = bool(t_pval < alpha) if not np.isnan(t_pval) else False
    is_sig_w = bool(w_pval < alpha) if not np.isnan(w_pval) else False

    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "wilcoxon_statistic": w_stat,
        "wilcoxon_pvalue": w_pval,
        "is_significant_ttest": is_sig_t,
        "is_significant_wilcoxon": is_sig_w,
        "n_pairs": len(diffs),
    }


def aggregate_seed_metrics(
    seed_results: Dict[Union[int, str], Dict[str, Any]],
    findings: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Aggregates per-finding and macro performance across multiple seeds.

    Args:
        seed_results: Dict mapping seed (e.g. 42, 1337, 2024) to evaluation dict.
        findings: Target pathology classes (default: SHARED_CLASSES).

    Returns:
        Structured dict with mean ± std and 95% CIs for in-domain, cross-domain,
        and generalization gaps.
    """
    classes = findings or SHARED_CLASSES
    seeds = sorted(list(seed_results.keys()), key=lambda s: str(s))

    per_class_agg: Dict[str, Dict[str, Any]] = {}
    macro_in_auroc_list = []
    macro_cross_auroc_list = []
    macro_gap_list = []
    macro_ece_list = []

    for f in classes:
        in_aucs, cross_aucs, gaps = [], [], []

        for s in seeds:
            s_dict = seed_results[s]
            in_auc = _extract_metric_value(s_dict.get("in_domain", s_dict), f, "auroc")
            cross_auc = _extract_metric_value(s_dict.get("cross_domain", s_dict), f, "auroc")

            if in_auc is not None:
                in_aucs.append(in_auc)
            if cross_auc is not None:
                cross_aucs.append(cross_auc)
            if in_auc is not None and cross_auc is not None:
                gaps.append(in_auc - cross_auc)

        per_class_agg[f] = {
            "in_domain_auroc": compute_sample_ci_95(in_aucs),
            "cross_domain_auroc": compute_sample_ci_95(cross_aucs),
            "generalization_gap": compute_sample_ci_95(gaps),
            "raw_in_domain_aurocs": in_aucs,
            "raw_cross_domain_aurocs": cross_aucs,
            "raw_generalization_gaps": gaps,
        }

    # Macro aggregation
    for s in seeds:
        s_dict = seed_results[s]
        in_d = s_dict.get("in_domain", s_dict)
        cross_d = s_dict.get("cross_domain", s_dict)

        m_in = in_d.get("macro_auroc", in_d.get("val_macro_auroc"))
        m_cross = cross_d.get("macro_auroc", cross_d.get("val_macro_auroc"))
        m_ece = cross_d.get("ece_uniform", cross_d.get("ece"))

        if m_in is not None:
            macro_in_auroc_list.append(float(m_in))
        if m_cross is not None:
            macro_cross_auroc_list.append(float(m_cross))
        if m_in is not None and m_cross is not None:
            macro_gap_list.append(float(m_in - m_cross))
        if m_ece is not None:
            macro_ece_list.append(float(m_ece))

    return {
        "seeds_evaluated": seeds,
        "n_seeds": len(seeds),
        "per_class": per_class_agg,
        "macro_metrics": {
            "in_domain_auroc": compute_sample_ci_95(macro_in_auroc_list),
            "cross_domain_auroc": compute_sample_ci_95(macro_cross_auroc_list),
            "generalization_gap": compute_sample_ci_95(macro_gap_list),
            "ece_uniform": compute_sample_ci_95(macro_ece_list),
            "raw_macro_in_aurocs": macro_in_auroc_list,
            "raw_macro_cross_aurocs": macro_cross_auroc_list,
            "raw_macro_gaps": macro_gap_list,
            "raw_macro_eces": macro_ece_list,
        },
    }


def compare_multi_seed_architectures(
    densenet_seed_results: Dict[Union[int, str], Dict[str, Any]],
    efficientnet_seed_results: Dict[Union[int, str], Dict[str, Any]],
    findings: Optional[List[str]] = None,
    shift_direction: str = "nih_to_chexpert",
) -> Dict[str, Any]:
    """Compares DenseNet121 vs EfficientNet-B0 across multi-seed replications with paired hypothesis testing.

    Evaluates whether differences in generalization gaps (Δ_DN vs Δ_EFF) are
    statistically significant beyond stochastic random seed variance.
    """
    classes = findings or SHARED_CLASSES

    dn_agg = aggregate_seed_metrics(densenet_seed_results, findings=classes)
    eff_agg = aggregate_seed_metrics(efficientnet_seed_results, findings=classes)

    paired_findings: Dict[str, Dict[str, Any]] = {}
    for f in classes:
        dn_gaps = dn_agg["per_class"][f]["raw_generalization_gaps"]
        eff_gaps = eff_agg["per_class"][f]["raw_generalization_gaps"]

        if len(dn_gaps) == len(eff_gaps) and len(dn_gaps) >= 2:
            sig_test = paired_seed_significance_test(dn_gaps, eff_gaps)
        else:
            sig_test = {}

        paired_findings[f] = {
            "densenet_gap": dn_agg["per_class"][f]["generalization_gap"],
            "efficientnet_gap": eff_agg["per_class"][f]["generalization_gap"],
            "significance_test": sig_test,
        }

    # Macro gap comparison
    dn_m_gaps = dn_agg["macro_metrics"]["raw_macro_gaps"]
    eff_m_gaps = eff_agg["macro_metrics"]["raw_macro_gaps"]

    if len(dn_m_gaps) == len(eff_m_gaps) and len(dn_m_gaps) >= 2:
        macro_sig = paired_seed_significance_test(dn_m_gaps, eff_m_gaps)
    else:
        macro_sig = {}

    return {
        "shift_direction": shift_direction,
        "densenet_aggregation": dn_agg,
        "efficientnet_aggregation": eff_agg,
        "per_class_comparison": paired_findings,
        "macro_gap_significance": macro_sig,
    }


def compare_multi_seed_uncertainty_policies(
    policy_seed_results: Dict[str, Dict[Union[int, str], Dict[str, Any]]],
    baseline_policy: str = "u_zeros",
    findings: Optional[List[str]] = None,
    shift_direction: str = "chexpert_to_nih",
) -> Dict[str, Any]:
    """Compares CheXpert uncertainty policies across multi-seed replications.

    Executes paired significance tests between baseline_policy (U-Zeros) and
    alternatives (U-Ones and U-Ignore).
    """
    classes = findings or SHARED_CLASSES
    policies = list(policy_seed_results.keys())

    policy_aggs = {
        p: aggregate_seed_metrics(policy_seed_results[p], findings=classes)
        for p in policies
    }

    paired_comparisons: Dict[str, Any] = {}
    base_agg = policy_aggs.get(baseline_policy)

    if base_agg:
        base_gaps = base_agg["macro_metrics"]["raw_macro_gaps"]

        for p in policies:
            if p == baseline_policy:
                continue
            p_gaps = policy_aggs[p]["macro_metrics"]["raw_macro_gaps"]
            if len(base_gaps) == len(p_gaps) and len(base_gaps) >= 2:
                sig = paired_seed_significance_test(p_gaps, base_gaps)
            else:
                sig = {}

            paired_comparisons[f"{p}_vs_{baseline_policy}"] = {
                "tested_policy": p,
                "baseline_policy": baseline_policy,
                "gap_significance": sig,
            }

    return {
        "shift_direction": shift_direction,
        "policy_aggregations": policy_aggs,
        "paired_comparisons_to_baseline": paired_comparisons,
    }


def generate_multi_seed_tables(
    arch_comparison: Optional[Dict[str, Any]] = None,
    policy_comparison: Optional[Dict[str, Any]] = None,
    shift_direction: str = "nih_to_chexpert",
) -> str:
    """Formats publication Markdown tables for multi-seed replication experiments."""
    lines = [
        f"## Multi-Seed Statistical Replication & Significance Analysis ({shift_direction.upper()})",
        "",
    ]

    if arch_comparison:
        lines.extend([
            "### 1. Architecture Robustness across Seeds (Seeds: 42, 1337, 2024)",
            "",
            "| Finding | DenseNet121 Gap (Mean ± SD) | DenseNet 95% CI | EfficientNet-B0 Gap (Mean ± SD) | EfficientNet 95% CI | Mean Diff (Δ_DN - Δ_EFF) | Paired t-test p-value | Wilcoxon p-value | Significant (p<0.05) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        per_c = arch_comparison.get("per_class_comparison", {})
        for f, data in per_c.items():
            dn_gap = data.get("densenet_gap", {})
            eff_gap = data.get("efficientnet_gap", {})
            sig = data.get("significance_test", {})

            str_dn = f"{dn_gap.get('mean', 0.0):.4f} ± {dn_gap.get('std', 0.0):.4f}"
            str_dn_ci = f"[{dn_gap.get('ci_lower', 0.0):.4f}, {dn_gap.get('ci_upper', 0.0):.4f}]"

            str_eff = f"{eff_gap.get('mean', 0.0):.4f} ± {eff_gap.get('std', 0.0):.4f}"
            str_eff_ci = f"[{eff_gap.get('ci_lower', 0.0):.4f}, {eff_gap.get('ci_upper', 0.0):.4f}]"

            mean_diff = f"{sig.get('mean_diff', 0.0):+.4f}" if "mean_diff" in sig else "—"
            t_p = f"{sig.get('t_pvalue', 1.0):.4f}" if "t_pvalue" in sig else "—"
            w_p = f"{sig.get('wilcoxon_pvalue', 1.0):.4f}" if "wilcoxon_pvalue" in sig else "—"
            is_sig = "✓" if sig.get("is_significant_ttest") or sig.get("is_significant_wilcoxon") else " "

            bold_f = f"**{f}**" if f == HEADLINE_CLASS else f
            lines.append(
                f"| {bold_f} | {str_dn} | {str_dn_ci} | {str_eff} | {str_eff_ci} | {mean_diff} | {t_p} | {w_p} | {is_sig} |"
            )

        # Macro row
        dn_macro = arch_comparison.get("densenet_aggregation", {}).get("macro_metrics", {}).get("generalization_gap", {})
        eff_macro = arch_comparison.get("efficientnet_aggregation", {}).get("macro_metrics", {}).get("generalization_gap", {})
        m_sig = arch_comparison.get("macro_gap_significance", {})

        m_dn_str = f"{dn_macro.get('mean', 0.0):.4f} ± {dn_macro.get('std', 0.0):.4f}"
        m_eff_str = f"{eff_macro.get('mean', 0.0):.4f} ± {eff_macro.get('std', 0.0):.4f}"
        m_diff = f"{m_sig.get('mean_diff', 0.0):+.4f}" if "mean_diff" in m_sig else "—"
        m_t_p = f"{m_sig.get('t_pvalue', 1.0):.4f}" if "t_pvalue" in m_sig else "—"
        m_w_p = f"{m_sig.get('wilcoxon_pvalue', 1.0):.4f}" if "wilcoxon_pvalue" in m_sig else "—"
        m_sig_flag = "✓" if m_sig.get("is_significant_ttest") or m_sig.get("is_significant_wilcoxon") else " "

        lines.append(
            f"| **Macro Average** | **{m_dn_str}** | — | **{m_eff_str}** | — | **{m_diff}** | **{m_t_p}** | **{m_w_p}** | **{m_sig_flag}** |"
        )
        lines.append("")

    if policy_comparison:
        lines.extend([
            "### 2. Uncertainty Policy Replication & Significance vs. U-Zeros",
            "",
            "| Policy | Macro In-Domain AUROC | Macro Cross-Domain AUROC | Macro Gap (Mean ± SD) | Paired t-test vs U-Zeros | Significant (p<0.05) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])
        aggs = policy_comparison.get("policy_aggregations", {})
        pairs = policy_comparison.get("paired_comparisons_to_baseline", {})

        for p, agg in aggs.items():
            in_m = agg.get("macro_metrics", {}).get("in_domain_auroc", {})
            cr_m = agg.get("macro_metrics", {}).get("cross_domain_auroc", {})
            gap_m = agg.get("macro_metrics", {}).get("generalization_gap", {})

            str_in = f"{in_m.get('mean', 0.0):.4f} ± {in_m.get('std', 0.0):.4f}"
            str_cr = f"{cr_m.get('mean', 0.0):.4f} ± {cr_m.get('std', 0.0):.4f}"
            str_gap = f"{gap_m.get('mean', 0.0):.4f} ± {gap_m.get('std', 0.0):.4f}"

            if p == "u_zeros":
                pair_str = "Baseline"
                sig_flag = "—"
            else:
                pair_key = f"{p}_vs_u_zeros"
                sig_info = pairs.get(pair_key, {}).get("gap_significance", {})
                pval = sig_info.get("t_pvalue")
                pair_str = f"p={pval:.4f}" if pval is not None else "—"
                sig_flag = "✓" if sig_info.get("is_significant_ttest") else " "

            lines.append(f"| **{p.upper()}** | {str_in} | {str_cr} | {str_gap} | {pair_str} | {sig_flag} |")
        lines.append("")

    return "\n".join(lines)


def run_multi_seed_replication_suite(
    densenet_seed_results: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    efficientnet_seed_results: Optional[Dict[Union[int, str], Dict[str, Any]]] = None,
    policy_seed_results: Optional[Dict[str, Dict[Union[int, str], Dict[str, Any]]]] = None,
    findings: Optional[List[str]] = None,
    shift_direction: str = "nih_to_chexpert",
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Orchestrates comprehensive multi-seed replication and statistical hypothesis testing suite.

    Args:
        densenet_seed_results: Optional dict mapping seeds to DenseNet evaluations.
        efficientnet_seed_results: Optional dict mapping seeds to EfficientNet evaluations.
        policy_seed_results: Optional dict mapping policy names to seed evaluations.
        findings: Target pathology classes.
        shift_direction: Direction of shift.
        output_dir: Destination path for saving results.

    Returns:
        Structured dict containing all aggregations, hypothesis tests, and markdown table.
    """
    logger.info("Running multi-seed replication suite for %s...", shift_direction)

    arch_comparison = None
    if densenet_seed_results and efficientnet_seed_results:
        arch_comparison = compare_multi_seed_architectures(
            densenet_seed_results=densenet_seed_results,
            efficientnet_seed_results=efficientnet_seed_results,
            findings=findings,
            shift_direction=shift_direction,
        )

    policy_comparison = None
    if policy_seed_results:
        policy_comparison = compare_multi_seed_uncertainty_policies(
            policy_seed_results=policy_seed_results,
            findings=findings,
            shift_direction=shift_direction,
        )

    table_md = generate_multi_seed_tables(
        arch_comparison=arch_comparison,
        policy_comparison=policy_comparison,
        shift_direction=shift_direction,
    )

    results = {
        "shift_direction": shift_direction,
        "architecture_multi_seed": arch_comparison,
        "policy_multi_seed": policy_comparison,
        "table_markdown": table_md,
    }

    if output_dir:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        save_json(results, out_p / f"multi_seed_replication_{shift_direction}.json")
        with open(out_p / f"multi_seed_replication_{shift_direction}.md", "w", encoding="utf-8") as f:
            f.write(table_md)
        logger.info("Saved multi-seed replication results to %s", out_p)

    return results

