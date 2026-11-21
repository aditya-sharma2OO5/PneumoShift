"""Unit tests for empirical prediction provenance stamping and loader integrity guards."""

import io
from pathlib import Path
import numpy as np
import pytest

from pneumoshift.data.provenance import (
    get_valid_checkpoint_shas,
    verify_prediction_stamp,
)


def test_all_results_predictions_are_stamped_real_inference():
    """All .npz prediction files in results/predictions/ must carry valid stamps."""
    pred_dir = Path("results/predictions")
    assert pred_dir.is_dir(), f"Prediction directory {pred_dir} must exist"

    npz_files = list(pred_dir.glob("*.npz"))
    assert len(npz_files) >= 12, f"Expected at least 12 NIH prediction files, found {len(npz_files)}"

    valid_shas = get_valid_checkpoint_shas()
    required_meta = [
        "generated_by",
        "git_commit",
        "run_id",
        "checkpoint_sha256",
        "config_hash",
        "dataset_name",
        "row_order_hash",
        "timestamp",
        "data_source",
    ]

    for p in npz_files:
        with np.load(p, allow_pickle=True) as data:
            # 1. verify_prediction_stamp helper must pass
            assert verify_prediction_stamp(data, filepath=p) is True

            # 2. explicit checks on metadata fields
            assert str(data["data_source"]) == "real_inference", f"{p.name} data_source is not 'real_inference'"
            ckpt_sha = str(data["checkpoint_sha256"]).strip()
            assert ckpt_sha in valid_shas, f"{p.name} checkpoint SHA {ckpt_sha} not authorized"

            for k in required_meta:
                assert k in data, f"{p.name} missing required provenance metadata key '{k}'"


def test_loader_guard_rejects_quarantine_path():
    """Loader guard must strictly reject any file located in results/quarantine/."""
    dummy_data = {
        "y_true": np.array([0, 1]),
        "data_source": np.array("real_inference"),
        "checkpoint_sha256": np.array("6e77c6a6cc3e10cb5150924dd81ff74c5eb44359466ae2d3f56519c875d23a59"),
    }
    with pytest.raises(PermissionError, match="quarantined"):
        verify_prediction_stamp(dummy_data, filepath="results/quarantine/old_cross_domain/dummy.npz")


def test_loader_guard_rejects_missing_or_synthetic_data_source():
    """Loader guard must reject files with missing or non-'real_inference' data_source."""
    unstamped_data = {
        "y_true": np.array([0, 1]),
        "checkpoint_sha256": np.array("6e77c6a6cc3e10cb5150924dd81ff74c5eb44359466ae2d3f56519c875d23a59"),
    }
    with pytest.raises(ValueError, match="lacks 'data_source'"):
        verify_prediction_stamp(unstamped_data, filepath="results/predictions/unstamped.npz")

    synthetic_data = {
        "y_true": np.array([0, 1]),
        "data_source": np.array("synthetic"),
        "checkpoint_sha256": np.array("6e77c6a6cc3e10cb5150924dd81ff74c5eb44359466ae2d3f56519c875d23a59"),
    }
    with pytest.raises(ValueError, match="Only 'real_inference' data is permitted"):
        verify_prediction_stamp(synthetic_data, filepath="results/predictions/synthetic.npz")


def test_loader_guard_rejects_unauthorized_checkpoint_sha():
    """Loader guard must reject files whose checkpoint SHA is not in the authorized manifest."""
    bogus_ckpt_data = {
        "y_true": np.array([0, 1]),
        "data_source": np.array("real_inference"),
        "checkpoint_sha256": np.array("deadbeef" * 8),
    }
    with pytest.raises(ValueError, match="does not match any authorized model"):
        verify_prediction_stamp(bogus_ckpt_data, filepath="results/predictions/bogus.npz")
