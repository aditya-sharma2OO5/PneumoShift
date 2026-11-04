"""Unit tests for Phase 6 Unified Results Aggregation CLI Orchestrator."""

from pathlib import Path
import tempfile
import pytest

from scripts.aggregate_results import main, run_aggregation_pipeline
from pneumoshift.utils.io import load_json


class TestAggregateResultsCLI:
    """Verifies end-to-end execution of the Phase 6 results aggregation CLI."""

    def test_run_aggregation_pipeline_smoke_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_aggregation_pipeline(
                output_dir=tmpdir,
                smoke_test=True,
                n_resamples=50,
                fdr_alpha=0.05,
            )

            assert res["phase"] == "Phase 6"
            assert res["smoke_test"] is True
            assert Path(res["manifest_path"]).is_file()

            # Verify hypothesis testing results
            hyp_p = Path(res["hypothesis_testing"]["json_path"])
            assert hyp_p.is_file()
            hyps = load_json(hyp_p)
            assert len(hyps) == 5

            # Verify exported tables
            tables = res["exported_tables"]
            assert len(tables) == 6
            for tid in ["table1", "table2", "table3", "table4", "table5", "table6"]:
                assert tid in tables
                for fmt in ["csv", "json", "md", "tex"]:
                    p = Path(tables[tid][fmt])
                    assert p.is_file()
                    assert p.stat().st_size > 0

            # Verify master report files
            assert (Path(tmpdir) / "phase6_master_report.json").is_file()
            assert (Path(tmpdir) / "phase6_master_report.md").is_file()

    def test_run_aggregation_pipeline_with_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            res = run_aggregation_pipeline(
                config_path="configs/aggregation/default.yaml",
                output_dir=tmpdir,
                smoke_test=True,
            )
            assert res["phase"] == "Phase 6"
            assert (Path(tmpdir) / "run_manifest.json").is_file()

    def test_cli_main_smoke_test(self, monkeypatch):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_args = [
                "aggregate_results.py",
                "--output_dir", str(tmpdir),
                "--smoke_test",
                "--n_resamples", "20",
            ]
            monkeypatch.setattr("sys.argv", test_args)
            main()

            assert (Path(tmpdir) / "run_manifest.json").is_file()
            assert (Path(tmpdir) / "table1_cross_domain_discrimination.csv").is_file()
            assert (Path(tmpdir) / "phase6_master_report.md").is_file()

