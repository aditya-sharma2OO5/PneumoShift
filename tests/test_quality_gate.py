"""Unit tests for Phase 2 Quality Gate & Guardrails Verification CLI."""

from pathlib import Path
import pandas as pd
import pytest
import torch

from scripts.verify_phase2_gate import (
    check_checkpoint_payload,
    check_guardrails,
    check_sanity_bounds,
    check_splits_leakage,
    run_phase2_quality_gate,
)


class TestQualityGate:
    def test_check_splits_leakage_clean(self, tmp_path: Path):
        # Create disjoint synthetic splits
        train_df = pd.DataFrame({"patient_id": ["p1", "p2", "p3"], "Pneumonia": [1.0, 0.0, 0.0]})
        val_df = pd.DataFrame({"patient_id": ["p4", "p5"], "Pneumonia": [0.0, 1.0]})
        test_df = pd.DataFrame({"patient_id": ["p6", "p7"], "Pneumonia": [1.0, 0.0]})

        train_df.to_csv(tmp_path / "testds_train.csv", index=False)
        val_df.to_csv(tmp_path / "testds_val.csv", index=False)
        test_df.to_csv(tmp_path / "testds_test.csv", index=False)

        res = check_splits_leakage(tmp_path, "testds")
        assert res["passed"] is True
        assert "Zero patient leakage verified" in res["message"]

    def test_check_splits_leakage_detected(self, tmp_path: Path):
        # Create overlapping splits (leakage)
        train_df = pd.DataFrame({"patient_id": ["p1", "p2", "LEAK_PATIENT"], "Pneumonia": [1.0, 0.0, 0.0]})
        val_df = pd.DataFrame({"patient_id": ["p4", "p5"], "Pneumonia": [0.0, 1.0]})
        test_df = pd.DataFrame({"patient_id": ["LEAK_PATIENT", "p7"], "Pneumonia": [1.0, 0.0]})

        train_df.to_csv(tmp_path / "bad_train.csv", index=False)
        val_df.to_csv(tmp_path / "bad_val.csv", index=False)
        test_df.to_csv(tmp_path / "bad_test.csv", index=False)

        res = check_splits_leakage(tmp_path, "bad")
        assert res["passed"] is False
        assert "leakage detected" in res["message"].lower()

    def test_check_checkpoint_payload_valid(self, tmp_path: Path):
        ckpt_path = tmp_path / "checkpoint_best.pt"
        torch.save(
            {
                "epoch": 5,
                "model_state_dict": {},
                "optimizer_state_dict": {},
                "metrics": {"val_macro_auroc": 0.82},
                "config": {"seed": 42},
                "git_hash": "a1b2c3d4",
            },
            ckpt_path,
        )

        res = check_checkpoint_payload(ckpt_path)
        assert res["passed"] is True
        assert res["details"]["epoch"] == 5
        assert res["details"]["git_hash"] == "a1b2c3d4"

    def test_check_checkpoint_payload_missing_keys(self, tmp_path: Path):
        ckpt_path = tmp_path / "incomplete.pt"
        torch.save({"epoch": 1, "model_state_dict": {}}, ckpt_path)

        res = check_checkpoint_payload(ckpt_path)
        assert res["passed"] is False
        assert "missing mandatory keys" in res["message"].lower()

    def test_check_sanity_bounds(self):
        # Valid metrics
        valid_metrics = {
            "val_pneumonia_auroc": 0.735,
            "val_atelectasis_auroc": 0.820,
            "val_cardiomegaly_auroc": 0.910,
        }
        res = check_sanity_bounds(valid_metrics)
        assert res["passed"] is True

        # Invalid pneumonia metric (too low)
        invalid_metrics = {
            "val_pneumonia_auroc": 0.510,
        }
        res_fail = check_sanity_bounds(invalid_metrics)
        assert res_fail["passed"] is False
        assert len(res_fail["failures"]) > 0

    def test_check_guardrails(self):
        # Clean config: validation only
        clean_cfg = {"train": {"save_best_metric": "val_macro_auroc"}}
        assert check_guardrails(clean_cfg)["passed"] is True

        # Peeking violation config
        peeking_cfg = {"train": {"save_best_metric": "test_pneumonia_auroc"}}
        assert check_guardrails(peeking_cfg)["passed"] is False

    def test_run_phase2_quality_gate_smoke(self, tmp_path: Path):
        res = run_phase2_quality_gate(
            nih_dir=tmp_path / "nih",
            chexpert_dir=tmp_path / "chexpert",
            splits_dir=tmp_path / "splits",
            smoke_test=True,
        )
        assert "overall_passed" in res
        assert "checks" in res
        assert "patient_leakage" in res["checks"]
        assert "sanity_bounds" in res["checks"]
        assert "guardrails" in res["checks"]
        assert res["overall_passed"] is True

