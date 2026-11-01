"""Published literature benchmark targets and automated comparison utilities."""

from typing import Any, Dict, List, Optional, Tuple
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.evaluation.benchmarks")

# ---------------------------------------------------------------------------
# Published Literature Benchmarks (AUROC)
# ---------------------------------------------------------------------------

PAN_2019_NIH: Dict[str, float] = {
    "Atelectasis": 0.833,
    "Cardiomegaly": 0.920,
    "Consolidation": 0.804,
    "Edema": 0.910,
    "Pleural Effusion": 0.892,
    "Pneumonia": 0.734,
    "Pneumothorax": 0.898,
    "No Finding": 0.900,
}

IRVIN_2019_CHEXPERT_UZEROS: Dict[str, float] = {
    "Atelectasis": 0.811,
    "Cardiomegaly": 0.840,
    "Consolidation": 0.932,
    "Edema": 0.929,
    "Pleural Effusion": 0.931,
}

ARASTEH_2023_NIH: Dict[str, Dict[str, float]] = {
    "Atelectasis": {"mean": 0.747, "std": 0.004},
    "Cardiomegaly": {"mean": 0.875, "std": 0.005},
    "Consolidation": {"mean": 0.728, "std": 0.005},
    "Pleural Effusion": {"mean": 0.815, "std": 0.003},
    "Pneumonia": {"mean": 0.688, "std": 0.011},
    "Pneumothorax": {"mean": 0.844, "std": 0.004},
    "No Finding": {"mean": 0.722, "std": 0.003},
}

ARASTEH_2023_CHEXPERT: Dict[str, Dict[str, float]] = {
    "Atelectasis": {"mean": 0.684, "std": 0.004},
    "Cardiomegaly": {"mean": 0.867, "std": 0.003},
    "Consolidation": {"mean": 0.744, "std": 0.005},
    "Pleural Effusion": {"mean": 0.873, "std": 0.002},
    "Pneumonia": {"mean": 0.764, "std": 0.008},
    "Pneumothorax": {"mean": 0.855, "std": 0.003},
    "No Finding": {"mean": 0.872, "std": 0.003},
}

LITERATURE_BENCHMARKS: Dict[str, Any] = {
    "pan_2019_nih": PAN_2019_NIH,
    "irvin_2019_chexpert": IRVIN_2019_CHEXPERT_UZEROS,
    "arasteh_2023_nih": ARASTEH_2023_NIH,
    "arasteh_2023_chexpert": ARASTEH_2023_CHEXPERT,
    # Aliases
    "nih": PAN_2019_NIH,
    "chexpert": ARASTEH_2023_CHEXPERT,
}


def _resolve_benchmark_target(entry: Any) -> float:
    """Extracts target mean AUROC whether entry is float or dict with mean."""
    if isinstance(entry, dict) and "mean" in entry:
        return float(entry["mean"])
    return float(entry)


def compare_against_benchmarks(
    evaluated_metrics: Dict[str, float],
    benchmark_name: str = "pan_2019_nih",
    headline_class: str = "Pneumonia",
    sanity_bounds: Tuple[float, float] = (0.65, 0.85),
) -> Dict[str, Any]:
    """Compares evaluated model metrics against published literature benchmarks.

    Args:
        evaluated_metrics: Dict mapping metric names to values (e.g. ``auroc_pneumonia``).
        benchmark_name: Name of literature benchmark (e.g. ``pan_2019_nih``).
        headline_class: Finding targeted as primary headline finding.
        sanity_bounds: Expected min and max sanity interval for headline finding AUROC.

    Returns:
        Structured dict with comparisons, sanity status, and formatted table.
    """
    bench_data = LITERATURE_BENCHMARKS.get(benchmark_name.lower())
    if bench_data is None:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. Available: {list(LITERATURE_BENCHMARKS.keys())}"
        )

    comparisons: Dict[str, Dict[str, Any]] = {}
    table_rows: List[str] = [
        "| Finding | Evaluated AUROC | Literature Benchmark | Difference | Status |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]

    for finding, ref_val in bench_data.items():
        ref_auroc = _resolve_benchmark_target(ref_val)
        col_key = finding.lower().replace(" ", "_")

        # Try various metric keys: auroc_<col>, val_pneumonia_auroc, or direct finding
        eval_auroc = (
            evaluated_metrics.get(f"auroc_{col_key}")
            or evaluated_metrics.get(f"val_{col_key}_auroc")
            or evaluated_metrics.get(f"test_{col_key}_auroc")
            or evaluated_metrics.get(finding)
        )

        if eval_auroc is not None:
            diff = float(eval_auroc - ref_auroc)
            status = "Aligned" if abs(diff) <= 0.06 else ("Higher" if diff > 0 else "Lower")
            comparisons[finding] = {
                "evaluated": float(eval_auroc),
                "benchmark": ref_auroc,
                "difference": diff,
                "status": status,
            }
            table_rows.append(
                f"| **{finding}** | {eval_auroc:.4f} | {ref_auroc:.4f} | {diff:+.4f} | {status} |"
            )
        else:
            comparisons[finding] = {
                "evaluated": None,
                "benchmark": ref_auroc,
                "difference": None,
                "status": "Not Evaluated",
            }
            table_rows.append(
                f"| **{finding}** | — | {ref_auroc:.4f} | — | Not Evaluated |"
            )

    # Sanity checks on headline target
    headline_key = headline_class.lower().replace(" ", "_")
    headline_eval = (
        evaluated_metrics.get(f"auroc_{headline_key}")
        or evaluated_metrics.get(f"val_{headline_key}_auroc")
        or evaluated_metrics.get(f"test_{headline_key}_auroc")
        or evaluated_metrics.get(headline_class)
    )

    sanity_passed = True
    sanity_msg = "Headline finding within expected sanity bounds."
    if headline_eval is not None:
        min_bound, max_bound = sanity_bounds
        if headline_eval < min_bound or headline_eval > max_bound:
            sanity_passed = False
            sanity_msg = (
                f"Headline AUROC {headline_eval:.4f} outside sanity bounds "
                f"[{min_bound:.2f}, {max_bound:.2f}]."
            )
            logger.warning(sanity_msg)
        else:
            logger.info("Sanity check passed: %s AUROC = %.4f", headline_class, headline_eval)
    else:
        sanity_msg = f"Headline class '{headline_class}' not present in evaluated metrics."

    formatted_table = "\n".join(table_rows)

    return {
        "benchmark_name": benchmark_name,
        "comparisons": comparisons,
        "headline_sanity_passed": sanity_passed,
        "headline_sanity_message": sanity_msg,
        "table_markdown": formatted_table,
    }

