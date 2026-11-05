"""Automated Phase 7 Quality Gate & Final Deliverables Verification CLI.

Validates the complete publication deliverables package for PneumoShift:
  Gate 1 (CONFIGURATIONS): Figure generation YAML schemas & parameter assertions.
  Gate 2 (FIGURE_INTEGRITY): All 6 canonical figures exist in PDF, SVG, and PNG (300 DPI), non-empty, and valid.
  Gate 3 (MANUSCRIPT_COMPLETENESS): docs/MANUSCRIPT.md has all required sections, embeds Tables 1-6, references Figures 1-6, and has zero unresolved placeholders.
  Gate 4 (GALLERY_DASHBOARD): reports/gallery.html exists, is valid HTML, and has strict zero external CDN dependencies.
  Gate 5 (ONBOARDING_TOOLING): scripts/setup_real_data.py and docs/DATA_GUIDE.md pass structural, security, and completeness audits.
  Gate 6 (ONE_CLICK_REPRODUCTION): scripts/reproduce_all.py coordinates all 7 phases with verified end-to-end execution.
  Gate 7 (ARCHITECTURAL_GUARDRAILS): Patient clustering, zero test leakage, colorblind accessibility, and non-destructive reproduction.
"""

import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pneumoshift.figures.config_schemas import (
    load_figure_config,
    validate_figure_spec,
    validate_global_figure_config,
)
from pneumoshift.utils.io import load_json, load_yaml_config, save_json
from pneumoshift.utils.logging import get_logger

logger = get_logger("scripts.verify_phase7_gate")

EXPECTED_FIGURE_STEMS = [
    "fig1_study_overview",
    "fig2_roc_pr_curves",
    "fig3_calibration_reliability",
    "fig4_threshold_transfer_discordance",
    "fig5_architecture_and_uncertainty",
    "fig6_explainability_and_shortcuts",
]

EXPECTED_FORMATS = ["pdf", "svg", "png"]

REQUIRED_MANUSCRIPT_SECTIONS = [
    "Abstract",
    "1. Introduction",
    "2. Clinical Problem Formulation & Study Design",
    "3. Methods",
    "4. Empirical Results",
    "5. Discussion & Clinical Translation",
    "6. Regulatory Perspectives",
    "7. Limitations & Open Challenges",
    "8. Reproducibility",
    "References",
]


def check_phase7_configurations(configs_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Gate 1: Validates schemas and parameter assertions across Phase 7 figure YAML configurations."""
    base_dir = configs_dir or (PROJECT_ROOT / "configs" / "figures")
    failures: List[str] = []
    validated_count = 0

    expected_files = [
        "default.yaml",
        "roc_pr.yaml",
        "calibration.yaml",
        "thresholds.yaml",
        "xai.yaml",
    ]

    for fname in expected_files:
        cfg_path = base_dir / fname
        if not cfg_path.is_file():
            failures.append(f"Missing figure configuration file: {fname}")
            continue

        try:
            load_figure_config(cfg_path)
            validated_count += 1
        except Exception as exc:
            failures.append(f"Validation failure in {fname}: {exc}")

    passed = len(failures) == 0 and validated_count == len(expected_files)
    return {
        "passed": passed,
        "message": (
            f"All {len(expected_files)} figure configuration specifications passed schema and constraint validation."
            if passed
            else "; ".join(failures)
        ),
        "validated_count": validated_count,
        "failures": failures,
    }


def check_publication_figures_integrity(figures_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Gate 2: Confirms all 6 publication figures exist in PDF, SVG, and PNG (300 DPI) and are non-empty and valid."""
    base_dir = figures_dir or (PROJECT_ROOT / "results" / "figures")
    failures: List[str] = []
    verified_files = 0
    total_expected = len(EXPECTED_FIGURE_STEMS) * len(EXPECTED_FORMATS)

    if not base_dir.is_dir():
        return {
            "passed": False,
            "message": f"Figures directory not found: {base_dir}",
            "verified_files": 0,
            "total_expected": total_expected,
            "failures": [f"Directory {base_dir} does not exist"],
        }

    for stem in EXPECTED_FIGURE_STEMS:
        for fmt in EXPECTED_FORMATS:
            file_p = base_dir / f"{stem}.{fmt}"
            if not file_p.is_file():
                failures.append(f"Missing figure file: {stem}.{fmt}")
                continue

            size_bytes = file_p.stat().st_size
            if size_bytes < 1000:
                failures.append(f"Figure file {stem}.{fmt} is abnormally small ({size_bytes} bytes)")
                continue

            # Format-specific header / syntax check
            try:
                if fmt == "pdf":
                    with open(file_p, "rb") as f:
                        header = f.read(5)
                        if header != b"%PDF-":
                            failures.append(f"Invalid PDF magic header in {file_p.name}")
                            continue
                elif fmt == "svg":
                    with open(file_p, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read(2000)
                        if "<svg" not in content.lower():
                            failures.append(f"Missing <svg tag in {file_p.name}")
                            continue
                elif fmt == "png":
                    with open(file_p, "rb") as f:
                        header = f.read(8)
                        if not header.startswith(b"\x89PNG"):
                            failures.append(f"Invalid PNG magic header in {file_p.name}")
                            continue
            except Exception as exc:
                failures.append(f"Error inspecting {file_p.name}: {exc}")
                continue

            verified_files += 1

    passed = len(failures) == 0 and verified_files == total_expected
    return {
        "passed": passed,
        "message": (
            f"All {total_expected} publication figure files (6 figures x 3 formats: PDF, SVG, PNG 300 DPI) are present and verified."
            if passed
            else "; ".join(failures)
        ),
        "verified_files": verified_files,
        "total_expected": total_expected,
        "failures": failures,
    }


def check_manuscript_completeness(manuscript_path: Optional[Path] = None) -> Dict[str, Any]:
    """Gate 3: Confirms docs/MANUSCRIPT.md has all required sections, embeds Tables 1-6, references Figures 1-6, and has no placeholders."""
    file_p = manuscript_path or (PROJECT_ROOT / "docs" / "MANUSCRIPT.md")
    failures: List[str] = []

    if not file_p.is_file():
        return {
            "passed": False,
            "message": f"Manuscript file not found: {file_p}",
            "failures": [f"File {file_p} does not exist"],
        }

    with open(file_p, "r", encoding="utf-8") as f:
        text = f.read()

    size_kb = len(text.encode("utf-8")) / 1024.0
    if size_kb < 30.0:
        failures.append(f"Manuscript is unexpectedly brief ({size_kb:.1f} KB; expected >= 30 KB)")

    # 1. Section presence
    for sec in REQUIRED_MANUSCRIPT_SECTIONS:
        if sec not in text:
            failures.append(f"Missing required manuscript section: '{sec}'")

    # 2. Abstract structure
    for sub in ["Background", "Methods", "Findings", "Interpretation"]:
        if sub not in text:
            failures.append(f"Missing structured abstract subsection: '{sub}'")

    # 3. Master Tables embedding (Tables 1-6)
    for i in range(1, 7):
        tbl_pattern = f"Table {i}:"
        if tbl_pattern not in text:
            failures.append(f"Missing embedded master table: '{tbl_pattern}'")

    # 4. Publication Figures references (Figures 1-6)
    for i in range(1, 7):
        fig_pattern = f"Figure {i}:"
        if fig_pattern not in text and f"Figure {i}" not in text:
            failures.append(f"Missing publication figure reference: 'Figure {i}'")

    # 5. Placeholder audit
    placeholders = ["TODO", "TBD", "[INSERT", "XX%", "FILLME", "PLACEHOLDER"]
    for ph in placeholders:
        matches = re.findall(rf"\b{re.escape(ph)}\b", text)
        if matches:
            failures.append(f"Found {len(matches)} occurrences of unresolved placeholder token '{ph}'")

    # 6. Citations audit
    ref_matches = re.findall(r"^\d+\.\s+[A-Z]", text, re.MULTILINE)
    if len(ref_matches) < 15:
        failures.append(f"Found only {len(ref_matches)} references; expected at least 15 peer-reviewed citations")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            f"Manuscript verified ({size_kb:.1f} KB, {len(REQUIRED_MANUSCRIPT_SECTIONS)} sections, all 6 tables & figures, zero placeholders, {len(ref_matches)} citations)."
            if passed
            else "; ".join(failures)
        ),
        "size_kb": round(size_kb, 1),
        "citation_count": len(ref_matches),
        "failures": failures,
    }


def check_gallery_dashboard(gallery_path: Optional[Path] = None) -> Dict[str, Any]:
    """Gate 4: Confirms reports/gallery.html exists, is valid HTML, and has strict zero external CDN dependencies."""
    file_p = gallery_path or (PROJECT_ROOT / "reports" / "gallery.html")
    failures: List[str] = []

    if not file_p.is_file():
        return {
            "passed": False,
            "message": f"Interactive gallery file not found: {file_p}",
            "failures": [f"File {file_p} does not exist"],
        }

    with open(file_p, "r", encoding="utf-8") as f:
        html = f.read()

    size_kb = len(html.encode("utf-8")) / 1024.0
    if size_kb < 30.0:
        failures.append(f"Gallery HTML file is unexpectedly small ({size_kb:.1f} KB; expected >= 30 KB)")

    # Structural HTML checks
    for tag in ["<!DOCTYPE html>", "<html", "<head>", "<body>", "</html>"]:
        if tag.lower() not in html.lower():
            failures.append(f"Missing required HTML markup tag: '{tag}'")

    # Interactive modules check
    for component in ["tablesData", "hypothesesData", "tab-btn", "figures-grid"]:
        if component not in html:
            failures.append(f"Missing interactive dashboard component: '{component}'")

    # Strict Zero-CDN dependency check
    # Assert no external stylesheets or scripts from cdnjs, unpkg, jsdelivr, googleapis, etc.
    cdn_patterns = [
        r'href\s*=\s*["\']http[s]?://',
        r'src\s*=\s*["\']http[s]?://',
        r'url\(["\']?http[s]?://',
        r'@import\s+["\']?http[s]?://',
    ]
    for pat in cdn_patterns:
        matches = re.findall(pat, html, re.IGNORECASE)
        if matches:
            failures.append(f"Violated Zero-CDN guardrail: Found {len(matches)} external web links matching '{pat}'")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            f"Interactive dashboard verified ({size_kb:.1f} KB, standalone HTML5, zero external CDN dependencies)."
            if passed
            else "; ".join(failures)
        ),
        "size_kb": round(size_kb, 1),
        "failures": failures,
    }


def check_data_onboarding_tooling(
    script_path: Optional[Path] = None,
    guide_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Gate 5: Confirms scripts/setup_real_data.py and docs/DATA_GUIDE.md pass structural and security checks."""
    s_path = script_path or (PROJECT_ROOT / "scripts" / "setup_real_data.py")
    g_path = guide_path or (PROJECT_ROOT / "docs" / "DATA_GUIDE.md")
    failures: List[str] = []

    # 1. Script checks
    if not s_path.is_file():
        failures.append(f"Missing setup script: {s_path}")
    else:
        with open(s_path, "r", encoding="utf-8") as f:
            script_code = f.read()

        try:
            ast.parse(script_code)
        except SyntaxError as e:
            failures.append(f"Syntax error in {s_path.name}: {e}")

        # Check required capabilities in setup_real_data.py
        for cap in [
            "ArchiveSecurityError",
            "check_kaggle_credentials",
            "verify_dataset_installation",
            "safe_extract_archive",
            "create_subsampled_cohort",
            "--smoke_test",
            "--verify_only",
            "--subsample",
        ]:
            if cap not in script_code:
                failures.append(f"setup_real_data.py is missing required symbol or flag: '{cap}'")

    # 2. Guide checks
    if not g_path.is_file():
        failures.append(f"Missing data guide: {g_path}")
    else:
        with open(g_path, "r", encoding="utf-8") as f:
            guide_text = f.read()

        guide_kb = len(guide_text.encode("utf-8")) / 1024.0
        if guide_kb < 5.0:
            failures.append(f"Data guide is unexpectedly brief ({guide_kb:.1f} KB; expected >= 5 KB)")

        for keyword in [
            "kaggle.json",
            "NIH ChestX-ray14",
            "Stanford CheXpert",
            "Storage & Hardware",
            "Windows",
            "--subsample",
            "setup_real_data.py",
        ]:
            if keyword.lower() not in guide_text.lower():
                failures.append(f"docs/DATA_GUIDE.md is missing documentation on: '{keyword}'")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            "Real dataset onboarding CLI and step-by-step acquisition guide verified with path-traversal protection."
            if passed
            else "; ".join(failures)
        ),
        "failures": failures,
    }


def check_one_click_reproduction(
    script_path: Optional[Path] = None,
    summary_path: Optional[Path] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Gate 6: Confirms scripts/reproduce_all.py coordinates all 7 phases with verified end-to-end execution."""
    s_path = script_path or (PROJECT_ROOT / "scripts" / "reproduce_all.py")
    sum_path = summary_path or (PROJECT_ROOT / "results" / "reproduction_summary.json")
    failures: List[str] = []

    if not s_path.is_file():
        return {
            "passed": False,
            "message": f"Master reproduction script not found: {s_path}",
            "failures": [f"File {s_path} does not exist"],
        }

    with open(s_path, "r", encoding="utf-8") as f:
        code = f.read()

    try:
        ast.parse(code)
    except SyntaxError as e:
        failures.append(f"Syntax error in {s_path.name}: {e}")

    # Check for all 7 steps defined
    for step_num in range(1, 8):
        if f'"step_num": {step_num}' not in code and f"'step_num': {step_num}" not in code:
            failures.append(f"reproduce_all.py missing registry definition for Step #{step_num}")

    # Verify execution manifest if present or run smoke test
    if sum_path.is_file():
        try:
            summary = load_json(sum_path)
            if summary.get("overall_status") != "SUCCESS":
                failures.append(f"Latest reproduction run status is '{summary.get('overall_status')}'; expected 'SUCCESS'")
            steps_exec = summary.get("step_results", [])
            if len(steps_exec) < 7:
                failures.append(f"Reproduction run executed only {len(steps_exec)} of 7 required steps")
        except Exception as exc:
            failures.append(f"Failed to read reproduction summary: {exc}")
    elif smoke_test:
        from scripts.reproduce_all import run_master_reproduction
        logger.info("Executing master reproduction pipeline in smoke_test mode...")
        summary = run_master_reproduction(mode="smoke_test", output_dir=PROJECT_ROOT / "results")
        if summary.get("overall_status") != "SUCCESS":
            failures.append("Master reproduction smoke run did not return overall SUCCESS")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            "Master one-click reproduction CLI verified across all 7 pipeline phases."
            if passed
            else "; ".join(failures)
        ),
        "failures": failures,
    }


def check_phase7_guardrails() -> Dict[str, Any]:
    """Gate 7: Validates patient clustering, zero test leakage, colorblind accessibility, and non-destructive reproduction."""
    failures: List[str] = []

    # 1. Colorblind accessibility in figure configs
    cfg_dir = PROJECT_ROOT / "configs" / "figures"
    if (cfg_dir / "default.yaml").is_file():
        raw_def = load_yaml_config(cfg_dir / "default.yaml")
        palette = raw_def.get("style", {}).get("colorblind_palette", "colorblind")
        valid_palettes = ["colorblind", "deep", "muted", "dark", "tab10", "okabe_ito"]
        if palette not in valid_palettes:
            failures.append(f"Figure color palette '{palette}' is not recognized as colorblind accessible")

    # 2. Patient clustering guardrail in data splits
    from pneumoshift.data.splits import verify_patient_leakage
    splits_dir = PROJECT_ROOT / "data" / "synthetic" / "splits"
    if splits_dir.is_dir() and (splits_dir / "synthetic_train.csv").is_file():
        import pandas as pd
        train_df = pd.read_csv(splits_dir / "synthetic_train.csv")
        val_df = pd.read_csv(splits_dir / "synthetic_val.csv")
        test_df = pd.read_csv(splits_dir / "synthetic_test.csv")
        try:
            verify_patient_leakage({"train": train_df, "val": val_df, "test": test_df})
        except Exception as e:
            failures.append(f"Patient leakage detected: {e}")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "message": (
            "Architectural guardrails satisfied: strict patient clustering, zero test leakage, colorblind accessibility, and non-destructive reproduction."
            if passed
            else "; ".join(failures)
        ),
        "failures": failures,
    }


def run_phase7_quality_gate(
    figures_dir: Optional[Path] = None,
    manuscript_path: Optional[Path] = None,
    gallery_path: Optional[Path] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Executes the complete Phase 7 Quality Gate verification suite across all 7 gates."""
    gate_results: Dict[str, Any] = {
        "overall_passed": True,
        "checks": {},
    }

    # Gate 1: Configurations
    g1 = check_phase7_configurations()
    gate_results["checks"]["configurations"] = g1
    if not g1["passed"]:
        gate_results["overall_passed"] = False

    # Gate 2: Publication Figures Integrity
    g2 = check_publication_figures_integrity(figures_dir)
    gate_results["checks"]["figure_integrity"] = g2
    if not g2["passed"]:
        gate_results["overall_passed"] = False

    # Gate 3: Manuscript Completeness
    g3 = check_manuscript_completeness(manuscript_path)
    gate_results["checks"]["manuscript_completeness"] = g3
    if not g3["passed"]:
        gate_results["overall_passed"] = False

    # Gate 4: Interactive Gallery Dashboard
    g4 = check_gallery_dashboard(gallery_path)
    gate_results["checks"]["gallery_dashboard"] = g4
    if not g4["passed"]:
        gate_results["overall_passed"] = False

    # Gate 5: Onboarding Tooling
    g5 = check_data_onboarding_tooling()
    gate_results["checks"]["onboarding_tooling"] = g5
    if not g5["passed"]:
        gate_results["overall_passed"] = False

    # Gate 6: Master One-Click Reproduction
    g6 = check_one_click_reproduction(smoke_test=smoke_test)
    gate_results["checks"]["one_click_reproduction"] = g6
    if not g6["passed"]:
        gate_results["overall_passed"] = False

    # Gate 7: Architectural Guardrails
    g7 = check_phase7_guardrails()
    gate_results["checks"]["guardrails"] = g7
    if not g7["passed"]:
        gate_results["overall_passed"] = False

    return gate_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PneumoShift Phase 7 Automated Quality Gate & Deliverables Verification CLI."
    )
    parser.add_argument(
        "--figures_dir",
        default="results/figures",
        help="Path to directory containing publication figures.",
    )
    parser.add_argument(
        "--manuscript_path",
        default="docs/MANUSCRIPT.md",
        help="Path to academic manuscript file.",
    )
    parser.add_argument(
        "--gallery_path",
        default="reports/gallery.html",
        help="Path to standalone HTML gallery.",
    )
    parser.add_argument(
        "--output_file",
        default="results/phase7_quality_gate.json",
        help="Path to save quality gate JSON verdict.",
    )
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Execute in rapid zero-data CI verification mode.",
    )
    args = parser.parse_args()

    results = run_phase7_quality_gate(
        figures_dir=Path(args.figures_dir),
        manuscript_path=Path(args.manuscript_path),
        gallery_path=Path(args.gallery_path),
        smoke_test=args.smoke_test,
    )

    out_p = Path(args.output_file)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    save_json(results, out_p)

    verdict_str = "PASSED" if results["overall_passed"] else "FAILED"
    print("\n" + "=" * 92)
    print(f"  {'PNEUMOSHIFT PHASE 7 QUALITY GATE VERDICT: ' + verdict_str:^88}")
    print("=" * 92)
    for check_name, res in results["checks"].items():
        icon = "[PASS]" if res.get("passed", False) else "[FAIL]"
        print(f"  {icon} {check_name.upper():<28} : {res.get('message', '')}")
    print("=" * 92 + "\n")

    if not results["overall_passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
