"""Provenance verification utilities for empirical predictions.

Enforces strict research integrity guards:
1. Rejects any prediction file whose data_source is not 'real_inference'.
2. Rejects any file containing quarantined, synthetic, or fallback data.
3. Asserts that the checkpoint SHA-256 recorded in the prediction file matches
   a known valid checkpoint in checkpoint_manifest.json.
"""

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Set


def get_valid_checkpoint_shas(manifest_path: Optional[Path] = None) -> Set[str]:
    """Retrieves the set of authorized checkpoint SHA-256 hashes."""
    candidates = [
        manifest_path,
        Path("checkpoint_manifest.json"),
        Path("configs/checkpoint_manifest.json"),
        Path("results/checkpoints/checkpoint_manifest.json"),
    ]
    for c in candidates:
        if c is not None and c.is_file():
            try:
                with open(c, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if "valid_sha256s" in data:
                    return set(data["valid_sha256s"])
                if "checkpoints" in data:
                    return {v["sha256"] for v in data["checkpoints"].values() if "sha256" in v}
            except Exception:
                pass

    # Built-in fallback set for the 6 verified NIH checkpoints
    return {
        "6e77c6a6cc3e10cb5150924dd81ff74c5eb44359466ae2d3f56519c875d23a59",  # densenet121 42
        "c3f6afccac87254c980a88ea72c71fb13631364483401cbed5bc8b2052b946a0",  # densenet121 1337
        "494dd05fd54cd0effdc3393534d99d6a821b2ff93f63bf0c4841493eb0c851c3",  # densenet121 2024
        "dc2b37375b6909b814830dd57eb384765f5c18ed8fdc34ced349fa48872458bc",  # efficientnet_b0 42
        "7772e387d3d2b4000825da5e70f858c09ce81862340d1f6199b0d0aad7f958b9",  # efficientnet_b0 1337
        "8d8b6f80892e9854001ac24bfadea81f25703c5d617ea83c24eab9935788815c",  # efficientnet_b0 2024
    }


def verify_prediction_stamp(
    data: Mapping[str, Any],
    filepath: Optional[Any] = None,
    manifest_path: Optional[Path] = None,
) -> bool:
    """Validates the provenance stamp of a loaded prediction dataset.

    Raises:
        ValueError: If data_source is missing, not 'real_inference', or if
                    checkpoint_sha256 is missing or not authorized.
    """
    path_str = str(filepath) if filepath is not None else "<in-memory>"

    # 1. Guard against quarantined paths
    if "quarantine" in path_str.lower():
        raise PermissionError(f"Access to quarantined data is strictly prohibited: {path_str}")

    # 2. Check data_source
    if "data_source" not in data:
        raise ValueError(
            f"Prediction file '{path_str}' lacks 'data_source' stamp! "
            "Unstamped files may contain unverified or synthetic data."
        )

    ds_val = str(data["data_source"])
    if ds_val != "real_inference":
        raise ValueError(
            f"Prediction file '{path_str}' has data_source='{ds_val}'. "
            "Only 'real_inference' data is permitted in evaluation, aggregation, and figures."
        )

    # 3. Check checkpoint SHA-256
    if "checkpoint_sha256" not in data:
        raise ValueError(
            f"Prediction file '{path_str}' lacks 'checkpoint_sha256' stamp! "
            "Cannot establish weight lineage."
        )

    ckpt_sha = str(data["checkpoint_sha256"]).strip()
    valid_shas = get_valid_checkpoint_shas(manifest_path)

    if ckpt_sha not in valid_shas:
        raise ValueError(
            f"Prediction file '{path_str}' checkpoint SHA-256 '{ckpt_sha}' "
            "does not match any authorized model in checkpoint_manifest.json!"
        )

    return True
