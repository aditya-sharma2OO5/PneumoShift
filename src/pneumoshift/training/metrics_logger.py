"""Per-epoch metric logging to CSV and JSON for reproducibility auditing."""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.training.metrics_logger")


class EpochMetricsLogger:
    """Appends per-epoch metrics to a CSV file and writes a cumulative JSON snapshot.

    Columns are discovered dynamically from the first call to :meth:`log` so
    any combination of per-class AUROCs, AUPRCs, loss values, and custom
    scalars is supported without hard-coding column names.

    Files written
    ~~~~~~~~~~~~~
    * ``epoch_metrics.csv``  — one row per epoch, append-only.
    * ``epoch_metrics.json`` — full history list, rewritten each epoch for
      programmatic downstream consumption (e.g. plotting notebooks).
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.output_dir / "epoch_metrics.csv"
        self.json_path = self.output_dir / "epoch_metrics.json"
        self._history: List[Dict[str, Any]] = []
        self._csv_columns: Optional[List[str]] = None

    def log(self, epoch: int, metrics: Dict[str, float]) -> None:
        """Persist one epoch's metrics to CSV (append) and JSON (rewrite).

        Args:
            epoch: 1-indexed epoch number.
            metrics: Flat dict of metric name → value.  Must include at
                least ``train_loss`` and ``loss`` (validation loss).
        """
        row: Dict[str, Any] = {"epoch": epoch, **metrics}
        self._history.append(row)

        # ---- CSV (append) ----
        if self._csv_columns is None:
            # First call — discover columns and write header
            self._csv_columns = list(row.keys())
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                writer.writeheader()
                writer.writerow(row)
        else:
            # Subsequent calls — add any new columns that appeared
            new_keys = [k for k in row.keys() if k not in self._csv_columns]
            if new_keys:
                self._csv_columns.extend(new_keys)
                # Rewrite entire CSV with updated header
                self._rewrite_csv()
            else:
                with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self._csv_columns)
                    writer.writerow(row)

        # ---- JSON (rewrite) ----
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self._history, f, indent=2, default=str)

        logger.debug("Logged metrics for epoch %d to %s", epoch, self.output_dir)

    def _rewrite_csv(self) -> None:
        """Rewrite the full CSV when new metric columns appear mid-training."""
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_columns, extrasaction="ignore")
            writer.writeheader()
            for row in self._history:
                writer.writerow(row)

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Returns accumulated metric rows (read-only copy)."""
        return list(self._history)

