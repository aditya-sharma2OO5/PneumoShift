"""File I/O, YAML parsing, hashing, and JSON helpers."""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict
import yaml


def load_yaml_config(config_path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file into a dictionary."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_json(data: Any, file_path: str | Path, indent: int = 2) -> None:
    """Saves data to a JSON file, creating parent directories if needed."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: str | Path) -> Any:
    """Loads data from a JSON file."""
    path = Path(file_path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_file_sha256(file_path: str | Path) -> str:
    """Computes SHA-256 hash of a file for integrity and reproducibility auditing."""
    path = Path(file_path)
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            sha256.update(chunk)
    return sha256.hexdigest()

