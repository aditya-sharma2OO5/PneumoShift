"""Unit tests for Phase 7 camera-ready academic research manuscript integrity."""

from pathlib import Path
import re
import pytest


def _has_stamped_cross_domain_predictions() -> bool:
    pred_dir = Path("results/predictions")
    if not pred_dir.is_dir():
        return False
    cross_files = list(pred_dir.glob("*cross*.npz"))
    if not cross_files:
        return False
    for p in cross_files:
        try:
            import numpy as np
            with np.load(p) as data:
                if str(data.get("data_source", "")) == "real_inference":
                    return True
        except Exception:
            continue
    return False


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
    if not _has_stamped_cross_domain_predictions():
        pytest.skip("No stamped cross-domain predictions exist: cross-domain tables quarantined/pending real inference")
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
    if not _has_stamped_cross_domain_predictions():
        pytest.skip("No stamped cross-domain predictions exist: cross-domain figures quarantined/pending real inference")
    p = Path("docs/MANUSCRIPT.md")
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    for fig_num in range(1, 7):
        assert f"Figure {fig_num}" in content, f"Missing reference to Figure {fig_num}"
        assert f"fig{fig_num}_" in content, f"Missing figure file link to fig{fig_num}"


def test_manuscript_contains_no_quarantined_numbers():
    """Asserts that docs/MANUSCRIPT.md contains zero numeric values or tokens from the quarantined set."""
    p = Path("docs/MANUSCRIPT.md")
    assert p.is_file(), "docs/MANUSCRIPT.md must exist"
    with open(p, "r", encoding="utf-8") as f:
        text = f.read()

    quarantined_tokens = [
        "82.5",     # Pointing game in-domain hit rate
        "38.0",     # Pointing game cross-domain hit rate
        "288.6",    # Peripheral shortcut surge
        "+288",     # Peripheral shortcut surge
        "0.661",    # DenseNet121 transferred AUROC
        "0.700",    # EfficientNet-B0 transferred AUROC
        "0.658",    # DenseNet121 CheXpert->NIH AUROC
        "0.692",    # EfficientNet-B0 CheXpert->NIH AUROC
        "0.752",    # DenseNet121 CheXpert in-domain synthetic AUROC
        "0.761",    # EfficientNet-B0 CheXpert in-domain synthetic AUROC
        "0.138",    # DenseNet cross-domain uncalibrated ECE
        "0.124",    # DenseNet source temp scaled ECE
        "0.058",    # DenseNet oracle recalibrated ECE
        "0.126",    # EfficientNet cross-domain uncalibrated ECE
        "0.115",    # EfficientNet source temp scaled ECE
        "0.052",    # EfficientNet oracle recalibrated ECE
        "0.075",    # Old cross-domain ECE
        "0.063",    # Old source temp scaled ECE
        "14.6%",    # DenseNet calibration recovery %
        "48.2%",    # DenseNet oracle recovery %
        "13.6%",    # EfficientNet calibration recovery %
        "52.4%",    # EfficientNet oracle recovery %
        "49.0%",    # Target specificity collapse
        "51.2%",    # Target specificity collapse
        "0.394",    # Transferred Youden F1
        "0.342",    # Transferred Sens-90 F1
        "0.462",    # Oracle target F1
        "0.418",    # Transferred Youden F1 (EffNet)
        "0.368",    # Transferred Sens-90 F1 (EffNet)
        "0.488",    # Oracle target F1 (EffNet)
        "0.985",    # DenseNet penultimate domain probe AUROC
        "0.942",    # Linear probe AUROC
        "2.76",     # In-domain faithfulness FDR ratio
        "1.20",     # Cross-domain faithfulness FDR ratio
        "-56.5%",   # FDR degradation %
        "2.94",     # EfficientNet in-domain FDR ratio
        "1.41",     # EfficientNet cross-domain FDR ratio
        "2.45",     # Uncertainty ablation over-prediction ratio
        "26.4%",    # Uncertainty ablation specificity collapse
    ]

    violations = []
    for token in quarantined_tokens:
        if token in text:
            violations.append(token)

    assert len(violations) == 0, (
        f"Manuscript contains quarantined numeric tokens: {violations}. "
        "All cross-domain metrics must be replaced with '[PENDING: real CheXpert inference]'."
    )


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

