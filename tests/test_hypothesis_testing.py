"""Unit tests for Phase 6 Statistical Hypothesis Testing and Multiple Comparisons Corrections."""

from pathlib import Path
import tempfile
import pytest

from pneumoshift.aggregation.collector import build_synthetic_manifest
from pneumoshift.aggregation.config_schemas import HypothesisTestRecord
from pneumoshift.aggregation.hypothesis_testing import (
    adjust_multiple_comparisons,
    benjamini_hochberg,
    bonferroni_correction,
    evaluate_study_hypotheses,
    format_hypothesis_markdown_table,
    holm_bonferroni,
    save_hypothesis_report,
)
from pneumoshift.utils.io import load_json


class TestMultipleComparisonsCorrections:
    """Verifies Benjamini-Hochberg, Bonferroni, and Holm corrections."""

    def test_benjamini_hochberg_basic(self):
        # 4 p-values
        raw_p = [0.01, 0.04, 0.03, 0.20]
        q_vals, sig = benjamini_hochberg(raw_p, alpha=0.05)
        assert len(q_vals) == 4
        assert len(sig) == 4
        # Smallest p-value (0.01) with m=4 has q = 0.01 * 4 / 1 = 0.04 <= 0.05
        assert q_vals[0] <= 0.05
        assert sig[0] is True
        # Largest p-value (0.20) is not significant at 0.05
        assert sig[3] is False

    def test_benjamini_hochberg_monotonicity(self):
        raw_p = [0.001, 0.004, 0.015, 0.020, 0.050, 0.120]
        q_vals, _ = benjamini_hochberg(raw_p, alpha=0.05)
        # Verify that ordered raw p-values map to monotonic q-values
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-9

    def test_empty_p_values(self):
        assert benjamini_hochberg([]) == ([], [])
        assert bonferroni_correction([]) == ([], [])
        assert holm_bonferroni([]) == ([], [])

    def test_bonferroni_correction(self):
        raw_p = [0.01, 0.02, 0.50]
        p_adj, sig = bonferroni_correction(raw_p, alpha=0.05)
        # m=3: 0.01 * 3 = 0.03 (sig), 0.02 * 3 = 0.06 (not sig), 0.50 * 3 = 1.0 (capped)
        assert pytest.approx(p_adj[0], rel=1e-5) == 0.03
        assert sig[0] is True
        assert pytest.approx(p_adj[1], rel=1e-5) == 0.06
        assert sig[1] is False
        assert p_adj[2] == 1.0

    def test_holm_bonferroni_step_down(self):
        raw_p = [0.005, 0.015, 0.040, 0.100]
        p_adj, sig = holm_bonferroni(raw_p, alpha=0.05)
        assert len(p_adj) == 4
        # Sorted: 0.005 * 4 = 0.02 (sig)
        assert p_adj[0] <= 0.05
        assert sig[0] is True
        # Verify monotonic
        for i in range(len(p_adj) - 1):
            assert p_adj[i] <= p_adj[i + 1] + 1e-9

    def test_adjust_multiple_comparisons_dispatcher(self):
        raw_p = [0.01, 0.03]
        q_bh, _ = adjust_multiple_comparisons(raw_p, method="benjamini_hochberg")
        p_bonf, _ = adjust_multiple_comparisons(raw_p, method="bonferroni")
        p_holm, _ = adjust_multiple_comparisons(raw_p, method="holm")

        assert len(q_bh) == 2
        assert len(p_bonf) == 2
        assert len(p_holm) == 2

        with pytest.raises(ValueError, match="Unknown multiple comparisons method"):
            adjust_multiple_comparisons(raw_p, method="unsupported_method")


class TestStudyHypothesesEvaluation:
    """Verifies formal evaluation of study hypotheses H1 through H5."""

    def test_evaluate_study_hypotheses_synthetic(self):
        manifest = build_synthetic_manifest()
        records = evaluate_study_hypotheses(manifest, alpha=0.05, fdr_method="benjamini_hochberg")

        assert len(records) == 5
        hyp_ids = [r.hypothesis_id for r in records]
        assert hyp_ids == ["H1", "H2", "H3", "H4", "H5"]

        for r in records:
            assert isinstance(r, HypothesisTestRecord)
            assert 0.0 <= r.raw_p_value <= 1.0
            assert 0.0 <= r.fdr_q_value <= 1.0
            # Under synthetic calibrated mode, all study hypotheses are supported
            assert r.is_significant is True

        # Verify H1 specifics
        h1 = records[0]
        assert h1.hypothesis_id == "H1"
        assert h1.effect_size > 0.05  # Generalization drop
        assert h1.test_type == "paired_bootstrap"

        # Verify H2 specifics
        h2 = records[1]
        assert h2.hypothesis_id == "H2"
        assert h2.effect_size > 0.03  # ECE escalation

    def test_format_hypothesis_markdown_table(self):
        manifest = build_synthetic_manifest()
        records = evaluate_study_hypotheses(manifest)
        md = format_hypothesis_markdown_table(records)

        assert "### Table 6: Consolidated Hypothesis Testing" in md
        assert "| **H1** |" in md
        assert "| **H5** |" in md
        assert "REJECT $H_0$ (PASS)" in md

    def test_save_hypothesis_report(self):
        manifest = build_synthetic_manifest()
        records = evaluate_study_hypotheses(manifest)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_json = Path(tmpdir) / "hypothesis_results.json"
            json_p, md_p = save_hypothesis_report(records, out_json)

            assert json_p.is_file()
            assert md_p.is_file()

            loaded = load_json(json_p)
            assert len(loaded) == 5
            assert loaded[0]["hypothesis_id"] == "H1"

            with open(md_p, "r", encoding="utf-8") as f:
                md_text = f.read()
            assert "Table 6" in md_text

