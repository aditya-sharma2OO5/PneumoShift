"""Unit tests for Phase 7 camera-ready academic research manuscript integrity."""

from pathlib import Path
import re
import pytest


def test_manuscript_file_exists():
    p = Path("docs/MANUSCRIPT.md")
    assert p.is_file(), "docs/MANUSCRIPT.md must exist"
    assert p.stat().st_size > 20 * 1024, "Manuscript must be comprehensive (>20 KB)"


def test_manuscript_required_sections():
    p = Path("docs/MANUSCRIPT.md")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    required_headers = [
        "## Abstract",
        "### Background",
        "### Methods",
        "### Findings",
        "### Interpretation",
        "## 1. Introduction",
        "## 2. Clinical Problem Formulation & Study Design",
        "## 3. Methods",
        "## 4. Empirical Results",
        "## 5. Discussion & Clinical Translation",
        "## 6. Regulatory Perspectives",
        "## 7. Limitations & Open Challenges",
        "## 8. Reproducibility & Code Availability",
        "## References",
    ]

    for header in required_headers:
        assert header in content, f"Missing required manuscript section: '{header}'"


def test_manuscript_embeds_all_tables():
    p = Path("docs/MANUSCRIPT.md")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    expected_tables = [
        "Table 1: Primary Cross-Domain Discrimination",
        "Table 2: 4-Regime Probability Calibration",
        "Table 3: Source-Optimal Decision Threshold Transfer",
        "Table 4: Architecture & CheXpert Uncertainty Policy Ablations",
        "Table 5: Explainability Saliency Audit",
        "Table 6: Consolidated Hypothesis Testing",
    ]

    for t in expected_tables:
        assert t in content, f"Missing table in manuscript: '{t}'"


def test_manuscript_references_all_figures():
    p = Path("docs/MANUSCRIPT.md")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    for fig_num in range(1, 7):
        assert f"Figure {fig_num}" in content, f"Missing reference to Figure {fig_num}"
        assert f"fig{fig_num}_" in content, f"Missing figure file link to fig{fig_num}"


def test_manuscript_zero_unfinished_placeholders():
    p = Path("docs/MANUSCRIPT.md")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    forbidden_tokens = ["TODO", "TBD", "[INSERT", "XXX", "FIXME"]
    for token in forbidden_tokens:
        matches = re.findall(rf"\b{re.escape(token)}\b", content)
        assert len(matches) == 0, f"Detected unfinished placeholder '{token}' in manuscript"


def test_manuscript_references_count():
    p = Path("docs/MANUSCRIPT.md")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify at least 15 formal references
    ref_matches = re.findall(r"^\d+\.\s+[A-Z]", content, flags=re.MULTILINE)
    assert len(ref_matches) >= 15, f"Expected at least 15 references, found {len(ref_matches)}"

