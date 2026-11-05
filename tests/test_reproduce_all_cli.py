"""Unit tests for Master One-Click Reproduction CLI (scripts/reproduce_all.py)."""

import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict
import pytest

from scripts.reproduce_all import (
    STEP_DEFINITIONS,
    build_step_command,
    execute_subcommand,
    format_summary_table,
    run_master_reproduction,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestReproductionStepDefinitions:
    """Validates structural integrity of the 7-phase step registry."""

    def test_all_seven_steps_registered(self) -> None:
        assert len(STEP_DEFINITIONS) == 7
        step_nums = [s["step_num"] for s in STEP_DEFINITIONS]
        assert step_nums == [1, 2, 3, 4, 5, 6, 7]

    def test_step_attributes_and_existing_scripts(self) -> None:
        required_keys = {"step_num", "key", "name", "script", "description", "artifacts"}
        for s in STEP_DEFINITIONS:
            assert required_keys.issubset(s.keys())
            script_path = PROJECT_ROOT / s["script"]
            assert script_path.is_file(), f"Script {s['script']} must exist on disk"

            if "secondary_script" in s:
                sec_path = PROJECT_ROOT / s["secondary_script"]
                assert sec_path.is_file(), f"Secondary script {s['secondary_script']} must exist on disk"


class TestCommandBuilder:
    """Tests command argument generation across operational modes."""

    def test_build_step_command_smoke_test(self, tmp_path: Path) -> None:
        for step in STEP_DEFINITIONS:
            cmds = build_step_command(
                step_info=step,
                mode="smoke_test",
                output_dir=tmp_path,
                seed=123,
                python_exe="python",
            )
            assert len(cmds) >= 1
            for cmd in cmds:
                assert cmd[0] == "python"
                assert any("smoke_test" in arg or "--audit" in arg for arg in cmd)

    def test_build_step_command_subsample_and_full(self, tmp_path: Path) -> None:
        # Step 2: Training in full mode should specify 20 epochs
        step2 = STEP_DEFINITIONS[1]
        cmds_full = build_step_command(step2, mode="full", output_dir=tmp_path, seed=42)
        assert ["--epochs", "20"] == cmds_full[0][cmds_full[0].index("--epochs") : cmds_full[0].index("--epochs") + 2]

        # Step 7: Figures and Gallery returns 2 subcommands
        step7 = STEP_DEFINITIONS[6]
        cmds_p7 = build_step_command(step7, mode="smoke_test", output_dir=tmp_path)
        assert len(cmds_p7) == 2
        assert "generate_figures.py" in cmds_p7[0][1]
        assert "generate_gallery.py" in cmds_p7[1][1]


class TestSubprocessExecutionAndFormatting:
    """Tests execution mechanics and summary table formatting."""

    def test_execute_subcommand_success(self) -> None:
        cmd = [sys.executable, "-c", "print('Antigravity Test Output')"]
        retcode, stdout, stderr, elapsed = execute_subcommand(cmd)
        assert retcode == 0
        assert "Antigravity Test Output" in stdout
        assert elapsed > 0.0

    def test_execute_subcommand_failure(self) -> None:
        cmd = [sys.executable, "-c", "import sys; sys.exit(42)"]
        retcode, stdout, stderr, elapsed = execute_subcommand(cmd)
        assert retcode == 42
        assert elapsed >= 0.0

    def test_format_summary_table(self) -> None:
        mock_results = [
            {
                "step_num": 1,
                "name": "Data Splits",
                "status": "PASS",
                "elapsed": 3.42,
                "artifacts_found": True,
            },
            {
                "step_num": 2,
                "name": "Model Training",
                "status": "SKIPPED",
                "elapsed": 0.0,
                "artifacts_found": True,
            },
            {
                "step_num": 3,
                "name": "Cross-Domain Evaluation",
                "status": "FAILED",
                "elapsed": 1.25,
                "artifacts_found": False,
                "error_snippet": "OutOfMemoryError: CUDA",
            },
        ]
        table_str = format_summary_table(mock_results, total_elapsed=4.67)
        assert "PNEUMOSHIFT MASTER REPRODUCTION SUMMARY" in table_str
        assert "#1" in table_str
        assert "[PASS]" in table_str
        assert "[SKIPPED]" in table_str
        assert "[FAILED]" in table_str
        assert "Total Master Execution Time: 4.7 seconds" in table_str


class TestMasterReproductionRunner:
    """Tests the high-level orchestrator logic."""

    def test_skip_all_steps(self, tmp_path: Path) -> None:
        result = run_master_reproduction(
            mode="smoke_test",
            output_dir=tmp_path,
            seed=42,
            skip_steps=[1, 2, 3, 4, 5, 6, 7],
        )
        assert result["overall_status"] == "SUCCESS"
        assert result["steps_executed"] == 7
        assert all(step["status"] == "SKIPPED" for step in result["step_results"])

        manifest_file = tmp_path / "reproduction_summary.json"
        assert manifest_file.is_file()
        with open(manifest_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["mode"] == "smoke_test"
        assert data["seed"] == 42

    def test_only_step_execution(self, tmp_path: Path) -> None:
        # Run Step 1 only in smoke test mode
        result = run_master_reproduction(
            mode="smoke_test",
            output_dir=tmp_path,
            seed=42,
            only_step=1,
        )
        assert result["overall_status"] == "SUCCESS"
        assert result["steps_executed"] == 1
        assert result["step_results"][0]["step_num"] == 1
        assert result["step_results"][0]["status"] == "PASS"

    def test_cli_help_flag(self) -> None:
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "reproduce_all.py"), "--help"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 0
        assert "PneumoShift Master One-Click End-to-End Reproduction CLI" in res.stdout

