"""Unit tests for Phase 6 Master Publication Table Generation and Multi-Format Exporters."""

from pathlib import Path
import tempfile
import pandas as pd
import pytest

from pneumoshift.aggregation.collector import build_synthetic_manifest
from pneumoshift.aggregation.hypothesis_testing import evaluate_study_hypotheses
from pneumoshift.aggregation.tables import (
    build_table1_cross_domain_discrimination,
    build_table2_calibration_regimes,
    build_table3_threshold_transfer,
    build_table4_architecture_and_uncertainty,
    build_table5_explainability_audit,
    build_table6_hypothesis_testing,
    dataframe_to_latex,
    dataframe_to_markdown,
    export_all_master_tables,
    export_master_table,
)


@pytest.fixture
def sample_manifest():
    return build_synthetic_manifest()


class TestTableConverters:
    """Verifies LaTeX and Markdown conversion helpers."""

    def test_dataframe_to_markdown(self):
        df = pd.DataFrame({"Model": ["DenseNet", "EfficientNet"], "AUROC": [0.768, 0.774]})
        md = dataframe_to_markdown(df, title="Test Table")

        assert "### Test Table" in md
        assert "| Model | AUROC |" in md
        assert "| DenseNet | 0.768 |" in md

    def test_dataframe_to_latex(self):
        df = pd.DataFrame({
            "Setting": ["A_to_B", "C_to_D"],
            "Delta_%": [12.5, 24.8],
            "Range": ["0.5 +- 0.1", "0.8 +- 0.2"],
        })
        tex = dataframe_to_latex(df, caption="LaTeX Caption", label="tab:test")

        assert r"\begin{table}[htbp]" in tex
        assert r"\caption{LaTeX Caption}" in tex
        assert r"\label{tab:test}" in tex
        assert r"\toprule" in tex
        assert r"\midrule" in tex
        assert r"\bottomrule" in tex
        assert r"A\_to\_B" in tex
        assert r"Delta\_\%" in tex
        assert r"\pm " in tex


class TestTableBuilders:
    """Verifies that all 6 master tables are built with the required schema and columns."""

    def test_build_table1(self, sample_manifest):
        df = build_table1_cross_domain_discrimination(sample_manifest)
        assert len(df) == 8
        assert "Pneumonia AUROC [95% CI]" in df.columns
        assert "Generalization Gap (Delta AUROC)" in df.columns
        assert any(df["Architecture"] == "DenseNet121")
        assert any(df["Architecture"] == "EfficientNet-B0")

    def test_build_table2(self, sample_manifest):
        df = build_table2_calibration_regimes(sample_manifest)
        assert len(df) == 8
        assert "ECE (Equal-Width) [95% CI]" in df.columns
        assert "Calibration Recovery (%)" in df.columns
        assert any("Oracle Recalibrated" in r for r in df["Calibration Regime"])

    def test_build_table3(self, sample_manifest):
        df = build_table3_threshold_transfer(sample_manifest)
        assert len(df) == 6
        assert "Operating Threshold (tau)" in df.columns
        assert "Target F1 Score" in df.columns
        assert any("Source Youden J" in p for p in df["Threshold Policy"])

    def test_build_table4(self, sample_manifest):
        df = build_table4_architecture_and_uncertainty(sample_manifest)
        assert len(df) == 6
        assert "Uncertainty Policy" in df.columns
        assert "Over-Prediction Ratio" in df.columns
        assert any("U-Ones" in p for p in df["Uncertainty Policy"])

    def test_build_table5(self, sample_manifest):
        df = build_table5_explainability_audit(sample_manifest)
        assert len(df) == 4
        assert "Faithfulness Ratio (FDR)" in df.columns
        assert "Peripheral Shortcut Ratio (%)" in df.columns
        assert any("NIH In-Domain" in d for d in df["Domain Setting"])

    def test_build_table6(self, sample_manifest):
        df = build_table6_hypothesis_testing(sample_manifest)
        assert len(df) == 5
        assert "Hypothesis ID" in df.columns
        assert "FDR q-value" in df.columns
        assert list(df["Hypothesis ID"]) == ["H1", "H2", "H3", "H4", "H5"]


class TestTableExporters:
    """Verifies export of tables into CSV, JSON, Markdown, and LaTeX."""

    def test_export_single_master_table(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = export_master_table(df, title="Table A", base_name="table_a", output_dir=tmpdir)
            assert "csv" in paths and paths["csv"].is_file()
            assert "json" in paths and paths["json"].is_file()
            assert "md" in paths and paths["md"].is_file()
            assert "tex" in paths and paths["tex"].is_file()

            # Verify contents
            csv_text = paths["csv"].read_text()
            assert "A,B" in csv_text

            tex_text = paths["tex"].read_text()
            assert r"\toprule" in tex_text

    def test_export_all_master_tables(self, sample_manifest):
        with tempfile.TemporaryDirectory() as tmpdir:
            exported = export_all_master_tables(sample_manifest, output_dir=tmpdir)
            assert len(exported) == 6
            assert set(exported.keys()) == {"table1", "table2", "table3", "table4", "table5", "table6"}

            for tid, path_map in exported.items():
                for fmt in ["csv", "json", "md", "tex"]:
                    assert fmt in path_map
                    assert path_map[fmt].is_file()
                    assert path_map[fmt].stat().st_size > 0

