"""Master One-Click End-to-End Reproduction CLI Orchestrator for PneumoShift.

Coordinates and verifies the complete 7-phase empirical research pipeline:
  Step 1/7: Data split verification & leakage audit (prepare_splits.py)
  Step 2/7: Baseline model training / checkpoint verification (train.py)
  Step 3/7: Cross-domain generalization & calibration evaluation (evaluate_cross_domain.py)
  Step 4/7: Architecture inductive bias & uncertainty policy ablations (run_phase4_ablations.py)
  Step 5/7: Explainability suite, faithfulness & shortcut audit (run_phase5_xai.py)
  Step 6/7: Master results aggregation & statistical meta-analysis (aggregate_results.py)
  Step 7/7: Publication figure generation & interactive HTML gallery (generate_figures.py & generate_gallery.py)

Usage:
  # Rapid zero-data CI verification mode (<60s)
  py -3.12 scripts/reproduce_all.py --smoke_test

  # Subsampled cohort execution on single GPU (~10k images)
  py -3.12 scripts/reproduce_all.py --subsample --seed 42

  # Full-scale multi-GPU production reproduction (complete cohorts)
  py -3.12 scripts/reproduce_all.py --full --seed 42
"""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pneumoshift.utils.io import save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.reproduce_all")


STEP_DEFINITIONS = [
    {
        "step_num": 1,
        "key": "splits",
        "name": "Data Splits & Leakage Audit",
        "script": "scripts/prepare_splits.py",
        "description": "Patient-partitioned split generation and zero-leakage audit",
        "artifacts": [
            "data/synthetic/splits/synthetic_train.csv",
            "results/tables/audit_synthetic/data_audit_report.json",
        ],
    },
    {
        "step_num": 2,
        "key": "train",
        "name": "Baseline Model Training",
        "script": "scripts/train.py",
        "description": "Source domain model training and checkpoint verification",
        "artifacts": [
            "results/runs",
        ],
    },
    {
        "step_num": 3,
        "key": "cross_domain",
        "name": "Cross-Domain Generalization & Calibration",
        "script": "scripts/evaluate_cross_domain.py",
        "description": "Cross-domain generalization gap, ECE, temperature scaling, threshold transfer",
        "artifacts": [
            "results/tables/cross_domain/nih_to_chexpert/cross_domain_summary.json",
            "results/tables/cross_domain/nih_to_chexpert/unified_report.md",
        ],
    },
    {
        "step_num": 4,
        "key": "ablations",
        "name": "Architecture & Uncertainty Ablations",
        "script": "scripts/run_phase4_ablations.py",
        "description": "DenseNet vs EfficientNet comparison and CheXpert uncertainty policy ablation",
        "artifacts": [
            "results/tables/phase4_ablations/phase4_master_report.json",
            "results/tables/phase4_ablations/phase4_master_report.md",
        ],
    },
    {
        "step_num": 5,
        "key": "xai",
        "name": "Explainability Suite & Shortcut Audit",
        "script": "scripts/run_phase5_xai.py",
        "description": "Grad-CAM generation, faithfulness dynamics, Adebayo sanity, bounding box hit rates",
        "artifacts": [
            "results/tables/xai/phase5_master_report.json",
            "results/figures/xai/overlays/sample_pneumonia_gradcam.png",
        ],
    },
    {
        "step_num": 6,
        "key": "aggregation",
        "name": "Master Results Aggregation & Meta-Analysis",
        "script": "scripts/aggregate_results.py",
        "description": "Meta-analysis, FDR hypothesis testing (H1-H5), export of Master Tables 1-6",
        "artifacts": [
            "results/tables/master/table1_cross_domain_discrimination.csv",
            "results/tables/master/table6_hypothesis_testing_summary.csv",
            "results/tables/master/phase6_master_report.json",
        ],
    },
    {
        "step_num": 7,
        "key": "figures_and_gallery",
        "name": "Publication Figures & HTML Dashboard",
        "script": "scripts/generate_figures.py",
        "secondary_script": "scripts/generate_gallery.py",
        "description": "Publication Figures 1-6 (PDF, SVG, PNG 300 DPI) and standalone HTML gallery",
        "artifacts": [
            "results/figures/fig1_study_overview.png",
            "results/figures/fig2_roc_pr_curves.pdf",
            "reports/gallery.html",
        ],
    },
]


def build_step_command(
    step_info: Dict[str, Any],
    mode: str = "smoke_test",
    output_dir: Path = Path("results"),
    seed: int = 42,
    device: Optional[str] = None,
    python_exe: Optional[str] = None,
) -> List[List[str]]:
    """Constructs the exact executable command(s) for a pipeline step."""
    py = python_exe or sys.executable
    step_num = step_info["step_num"]
    commands: List[List[str]] = []

    if step_num == 1:
        # Data Splits
        cmd = [py, step_info["script"], "--seed", str(seed)]
        if mode == "smoke_test":
            cmd.extend(["--smoke_test", "--dataset", "synthetic", "--audit"])
        elif mode == "subsample":
            cmd.extend(["--dataset", "all", "--audit"])
        else:
            cmd.extend(["--dataset", "all", "--audit"])
        commands.append(cmd)

    elif step_num == 2:
        # Model Training
        cmd = [py, step_info["script"], "--seed", str(seed)]
        if mode == "smoke_test":
            cmd.extend(["--smoke_test", "--epochs", "1", "--architecture", "densenet121"])
        elif mode == "subsample":
            cmd.extend(["--epochs", "5", "--architecture", "densenet121"])
        else:
            cmd.extend(["--epochs", "20", "--architecture", "densenet121"])
        commands.append(cmd)

    elif step_num == 3:
        # Cross-Domain Evaluation
        cmd = [py, step_info["script"]]
        if mode == "smoke_test":
            cmd.extend(["--smoke_test", "--no_bootstrap"])
        elif mode == "subsample":
            cmd.extend(["--n_resamples", "100"])
        else:
            cmd.extend(["--n_resamples", "1000"])
        commands.append(cmd)

    elif step_num == 4:
        # Phase 4 Ablations
        out_p4 = str(output_dir / "tables" / "phase4_ablations")
        cmd = [py, step_info["script"], "--output_dir", out_p4]
        if mode == "smoke_test":
            cmd.append("--smoke_test")
        commands.append(cmd)

    elif step_num == 5:
        # Phase 5 XAI
        out_xai = str(output_dir / "tables" / "xai")
        fig_xai = str(output_dir / "figures" / "xai")
        cmd = [py, step_info["script"], "--output_dir", out_xai, "--figures_dir", fig_xai]
        if mode == "smoke_test":
            cmd.append("--smoke_test")
        commands.append(cmd)

    elif step_num == 6:
        # Phase 6 Aggregation
        out_master = str(output_dir / "tables" / "master")
        cmd = [py, step_info["script"], "--output_dir", out_master]
        if mode == "smoke_test":
            cmd.append("--smoke_test")
        elif mode == "subsample":
            cmd.extend(["--n_resamples", "200"])
        else:
            cmd.extend(["--n_resamples", "1000"])
        commands.append(cmd)

    elif step_num == 7:
        # Phase 7 Figures & Gallery (two consecutive commands)
        fig_out = str(output_dir / "figures")
        tbl_master = str(output_dir / "tables" / "master")
        cmd_figs = [py, step_info["script"], "--output_dir", fig_out, "--tables_dir", tbl_master]
        if mode == "smoke_test":
            cmd_figs.append("--smoke_test")
        commands.append(cmd_figs)

        gallery_out = "reports/gallery.html"
        cmd_gallery = [
            py,
            step_info["secondary_script"],
            "--output_file",
            gallery_out,
            "--tables_dir",
            tbl_master,
            "--figures_dir",
            fig_out,
        ]
        if mode == "smoke_test":
            cmd_gallery.append("--smoke_test")
        commands.append(cmd_gallery)

    return commands


def execute_subcommand(
    cmd: List[str],
    cwd: Optional[Path] = None,
    timeout: Optional[int] = None,
    verbose: bool = False,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, str, str, float]:
    """Executes a single CLI command in a separate subprocess with timing."""
    start_time = time.perf_counter()
    proc_env = os.environ.copy()
    if env:
        proc_env.update(env)

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd or PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=proc_env,
        )
        elapsed = time.perf_counter() - start_time
        return proc.returncode, proc.stdout, proc.stderr, elapsed
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - start_time
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = f"Command timed out after {timeout} seconds: {' '.join(cmd)}"
        return -1, stdout, stderr, elapsed
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        return -1, "", str(exc), elapsed


def format_summary_table(step_results: List[Dict[str, Any]], total_elapsed: float) -> str:
    """Formats a clean, publication-grade ASCII summary table."""
    lines = [
        "",
        "=" * 92,
        f"  {'PNEUMOSHIFT MASTER REPRODUCTION SUMMARY':^88}",
        "=" * 92,
        f"  {'Step':<6} | {'Pipeline Phase':<38} | {'Status':<10} | {'Duration':<10} | {'Verification'}",
        "-" * 92,
    ]

    for res in step_results:
        s_num = f"#{res['step_num']}"
        s_name = res["name"]
        status = res["status"]
        duration = f"{res['elapsed']:.1f}s"
        status_disp = f"[{status}]"
        
        # Check artifacts status
        all_artifacts_exist = res.get("artifacts_found", True)
        note = "OK - All artifacts present" if all_artifacts_exist else "Warning - Partial artifacts"
        if status == "SKIPPED":
            note = "Bypassed by user flag"
        elif status == "FAILED":
            note = res.get("error_snippet", "Execution failed")[:26]

        lines.append(f"  {s_num:<6} | {s_name:<38} | {status_disp:<10} | {duration:<10} | {note}")

    lines.extend([
        "=" * 92,
        f"  Total Master Execution Time: {total_elapsed:.1f} seconds ({total_elapsed / 60.0:.2f} minutes)",
        "=" * 92,
        "",
    ])
    return "\n".join(lines)


def run_master_reproduction(
    mode: str = "smoke_test",
    output_dir: Union[str, Path] = "results",
    seed: int = 42,
    device: Optional[str] = None,
    skip_steps: Optional[List[Union[int, str]]] = None,
    only_step: Optional[Union[int, str]] = None,
    continue_on_error: bool = False,
    verbose: bool = False,
    python_exe: Optional[str] = None,
) -> Dict[str, Any]:
    """Executes the master reproduction pipeline across all 7 phases.

    Args:
        mode: Execution scale mode ('smoke_test', 'subsample', 'full').
        output_dir: Target root directory for results and reports.
        seed: Master random seed.
        device: Computational device override (e.g. 'cpu', 'cuda').
        skip_steps: List of step indices (1-7) or step keys to bypass.
        only_step: Optional single step index or key to run exclusively.
        continue_on_error: Whether to continue pipeline execution after a step failure.
        verbose: If True, streams subprocess output directly to stdout.
        python_exe: Optional python interpreter binary path.

    Returns:
        Consolidated reproduction summary dictionary.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)

    py_exe = python_exe or sys.executable
    t0_master = time.perf_counter()

    # Parse skip / only step constraints
    skip_set = set()
    if skip_steps:
        for s in skip_steps:
            skip_set.add(str(s).strip().lower())

    only_filter: Optional[str] = None
    if only_step is not None:
        only_filter = str(only_step).strip().lower()

    print("\n" + "=" * 92)
    print(f"  {'PNEUMOSHIFT: MASTER ONE-CLICK REPRODUCTION PIPELINE':^88}")
    print("=" * 92)
    print(f"  Execution Mode       : {mode.upper()}")
    print(f"  Master Random Seed   : {seed}")
    print(f"  Output Directory     : {out_path.resolve()}")
    print(f"  Python Interpreter   : {py_exe}")
    print(f"  Device Configuration : {device or 'Auto-detect'}")
    print(f"  Steps to Execute     : {7 - len(skip_set) if not only_filter else 1} of 7")
    print("=" * 92 + "\n")

    step_results: List[Dict[str, Any]] = []
    overall_success = True

    for step_info in STEP_DEFINITIONS:
        s_num = step_info["step_num"]
        s_key = step_info["key"]
        s_name = step_info["name"]

        # Check skip / only conditions
        if only_filter and str(s_num) != only_filter and s_key != only_filter:
            continue

        if str(s_num) in skip_set or s_key in skip_set:
            logger.info("Skipping Step %d/7: %s (bypassed by user)", s_num, s_name)
            step_results.append({
                "step_num": s_num,
                "key": s_key,
                "name": s_name,
                "status": "SKIPPED",
                "elapsed": 0.0,
                "artifacts_found": True,
                "commands": [],
            })
            continue

        print(f"--> [Step {s_num}/7] Initiating: {s_name}...")
        cmds = build_step_command(
            step_info=step_info,
            mode=mode,
            output_dir=out_path,
            seed=seed,
            device=device,
            python_exe=py_exe,
        )

        step_elapsed = 0.0
        step_failed = False
        error_msg = ""

        for cmd_idx, cmd in enumerate(cmds, 1):
            cmd_display = " ".join(cmd[1:])
            logger.info("Executing subcommand [%d/%d]: py %s", cmd_idx, len(cmds), cmd_display)
            
            retcode, stdout, stderr, cmd_dur = execute_subcommand(
                cmd=cmd,
                cwd=PROJECT_ROOT,
                verbose=verbose,
            )
            step_elapsed += cmd_dur

            if verbose and stdout:
                print(stdout)

            if retcode != 0:
                step_failed = True
                overall_success = False
                error_msg = stderr.strip() if stderr else f"Exited with code {retcode}"
                logger.error("Step %d (%s) subcommand failed with returncode %d:\n%s", s_num, s_name, retcode, error_msg)
                if not verbose and stdout:
                    print(stdout[-500:])
                print(f"  [ERROR] {error_msg[-300:]}")
                break

        # Check expected artifacts
        artifacts_found = True
        missing_artifacts = []
        for art_rel in step_info.get("artifacts", []):
            art_p = PROJECT_ROOT / art_rel
            if not art_p.exists():
                artifacts_found = False
                missing_artifacts.append(art_rel)

        status_str = "FAILED" if step_failed else "PASS"
        if step_failed:
            print(f"  FAILED in {step_elapsed:.1f}s")
        else:
            print(f"  PASSED in {step_elapsed:.1f}s")

        step_results.append({
            "step_num": s_num,
            "key": s_key,
            "name": s_name,
            "status": status_str,
            "elapsed": step_elapsed,
            "artifacts_found": artifacts_found,
            "missing_artifacts": missing_artifacts,
            "commands": cmds,
            "error_snippet": error_msg[:120] if step_failed else None,
        })

        if step_failed and not continue_on_error:
            logger.error("Halting reproduction pipeline due to failure in Step %d.", s_num)
            break

    total_master_time = time.perf_counter() - t0_master

    # Print summary table
    summary_text = format_summary_table(step_results, total_master_time)
    print(summary_text)

    # Save reproduction manifest
    manifest: Dict[str, Any] = {
        "title": "PneumoShift Master Reproduction Manifest",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "seed": seed,
        "device": device or "auto",
        "python_exe": py_exe,
        "total_elapsed_seconds": round(total_master_time, 2),
        "overall_status": "SUCCESS" if overall_success else "FAILURE",
        "steps_executed": len(step_results),
        "step_results": step_results,
    }

    manifest_path = out_path / "reproduction_summary.json"
    save_json(manifest, manifest_path)
    logger.info("Saved reproduction summary manifest to %s", manifest_path)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PneumoShift Master One-Click End-to-End Reproduction CLI."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--smoke_test",
        action="store_true",
        help="Rapid zero-data CI verification mode (<60s).",
    )
    group.add_argument(
        "--subsample",
        action="store_true",
        help="Subsampled cohort reproduction (~10k-20k images per domain).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Full-scale multi-GPU production reproduction (complete cohorts).",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        help="Root output directory for artifacts (default: results).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master random seed for deterministic reproduction (default: 42).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override: 'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--skip_steps",
        default=None,
        help="Comma-separated list of step numbers or step keys to skip (e.g. '2,train').",
    )
    parser.add_argument(
        "--only_step",
        default=None,
        help="Execute only a single step number or key (e.g. '6' or 'aggregation').",
    )
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        help="Continue executing remaining steps even if an intermediate step fails.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Stream full stdout and stderr from subprocesses.",
    )
    args = parser.parse_args()

    mode = "smoke_test"
    if args.full:
        mode = "full"
    elif args.subsample:
        mode = "subsample"
    elif args.smoke_test:
        mode = "smoke_test"

    skip_list = [s.strip() for s in args.skip_steps.split(",")] if args.skip_steps else None

    result = run_master_reproduction(
        mode=mode,
        output_dir=args.output_dir,
        seed=args.seed,
        device=args.device,
        skip_steps=skip_list,
        only_step=args.only_step,
        continue_on_error=args.continue_on_error,
        verbose=args.verbose,
    )

    if result.get("overall_status") != "SUCCESS":
        sys.exit(1)


if __name__ == "__main__":
    main()

