#!/usr/bin/env python3
"""CLI entrypoint for pre-specified NIH -> CheXpert transfer analysis (ANALYSIS_PLAN_TRANSFER.md v2.2.3).

Examples:
  Dry run on NIH-only stored predictions (exercises pipeline end to end without CheXpert):
    python scripts/analyze_transfer.py --dry-run --out-dir outputs/dry_run

  Production run on verified oldckpt_real_v1 outputs:
    python scripts/analyze_transfer.py --preds-dir results/predictions/oldckpt_real_v1 --out-dir outputs/transfer_oldckpt_real_v1
"""

import argparse
import sys
from pathlib import Path

# Allow execution directly from repo root without editable install
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pneumoshift.analysis.pipeline import format_dry_run_note, run_transfer_analysis
from pneumoshift.analysis.provenance import (
    EXPECTED_PLAN_RUN_ID,
    load_nih_dry_run_bundle,
    load_real_run_bundle,
)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run pre-specified NIH -> CheXpert transfer analysis.")
    ap.add_argument("--dry-run", action="store_true", help="Run NIH-only stand-in pseudo-contrast (no CheXpert touched).")
    ap.add_argument("--preds-dir", type=Path, default=None, help="Directory holding stamped prediction .npz and gate JSONs.")
    ap.add_argument("--nih-splits-dir", type=Path, default=REPO_ROOT / "data" / "splits" / "nih", help="NIH split CSV directory (for dry-run subgroups).")
    ap.add_argument("--plan", type=Path, default=REPO_ROOT / "ANALYSIS_PLAN_TRANSFER.md", help="Path to ANALYSIS_PLAN_TRANSFER.md.")
    ap.add_argument("--manifest", type=Path, default=REPO_ROOT / "configs" / "checkpoint_manifest.json", help="Path to checkpoint_manifest.json.")
    ap.add_argument("--ref", type=Path, default=REPO_ROOT / "configs" / "nih_test_reference_logits.npz", help="Path to reference logits bundle.")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output directory for JSON and CSV tables.")
    ap.add_argument("--run-id", type=str, default=EXPECTED_PLAN_RUN_ID, help="Required execution run_id for production.")
    ap.add_argument("--expected-commit", type=str, default=None, help="Optional expected git_commit in NIH_GATE_PASSED.json.")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Exact bootstrap iterations (plan default: 2000).")
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap random seed (plan default: 42).")
    ap.add_argument("--platt-splits", type=int, default=20, help="Target-10%% Platt patient splits (plan default: 20).")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = args.out_dir
    if out_dir is None:
        out_dir = (REPO_ROOT / "outputs" / "dry_run") if args.dry_run else (REPO_ROOT / "outputs" / f"transfer_{args.run_id}")

    # Safety guard: never write DRY_RUN outputs into docs/
    if args.dry_run and "docs" in out_dir.resolve().parts:
        raise ValueError("Refusing to write DRY_RUN outputs inside docs/.")

    if args.dry_run:
        preds_dir = args.preds_dir or (REPO_ROOT / "results" / "predictions")
        bundle = load_nih_dry_run_bundle(
            preds_dir=preds_dir,
            nih_splits_dir=args.nih_splits_dir,
            plan_path=args.plan,
            manifest_path=args.manifest,
        )
    else:
        if args.preds_dir is None:
            raise ValueError("--preds-dir is required unless --dry-run is specified.")
        bundle = load_real_run_bundle(
            preds_dir=args.preds_dir,
            plan_path=args.plan,
            manifest_path=args.manifest,
            ref_bundle_path=args.ref,
            expected_run_id=args.run_id,
            expected_git_commit=args.expected_commit,
        )

    result = run_transfer_analysis(
        bundle=bundle,
        out_dir=out_dir,
        script_path=Path(__file__).resolve(),
        n_bootstrap=args.bootstrap,
        bootstrap_seed=args.seed,
        n_target_platt_splits=args.platt_splits,
    )

    prov = result["provenance"]
    print(f"=== Transfer Analysis Complete [{format_dry_run_note(prov['dry_run'])}] ===")
    print(f"Run ID        : {prov['run_id']}")
    print(f"Plan SHA-256  : {prov['analysis_plan_sha256']}")
    print(f"Script SHA-256: {prov['analysis_script_sha256']}")
    print(f"Results JSON  : {result['results_json_path']}")
    for row in result["primary_endpoints_6test_bh_family"]:
        print(
            f"  [{prov['mode']}] {row['architecture']:<15} | {row['endpoint_id']:<28} | "
            f"delta={row['delta_mean']:+.5f} [{row['delta_ci_lower']:+.5f}, {row['delta_ci_upper']:+.5f}] "
            f"seed_sd={row['delta_between_seed_sd']:.5f} p={row['p_value_two_sided']:.4g} q_BH={row['bh_q_value']:.4g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
