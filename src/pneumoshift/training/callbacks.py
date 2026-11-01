"""Training callbacks for early stopping and checkpoint management."""

from pathlib import Path
from typing import Any, Dict, Optional
import subprocess
import torch
import torch.nn as nn
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.training.callbacks")


def get_git_hash() -> str:
    """Attempts to retrieve the current git commit hash."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown_git_hash"


class EarlyStopping:
    """Early stops training when monitored metric stops improving."""

    def __init__(self, patience: int = 6, min_delta: float = 1e-4, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return True

        improved = (
            (current_score > self.best_score + self.min_delta)
            if self.mode == "max"
            else (current_score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                logger.info("Early stopping triggered after %d epochs without improvement.", self.counter)
            return False


class ModelCheckpoint:
    """Saves model weights, optimizer state, configuration, and metadata.

    Supports a primary monitor metric (default ``val_macro_auroc``) and an
    optional secondary monitor (e.g. ``val_pneumonia_auroc``) so that the best
    checkpoint for the headline finding is also preserved independently.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        monitor: str = "val_macro_auroc",
        secondary_monitor: Optional[str] = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.secondary_monitor = secondary_monitor
        self.best_score: float = -float("inf")
        self.best_secondary_score: float = -float("inf")

    def _build_state(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "git_hash": get_git_hash(),
        }

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        config: Dict[str, Any],
        is_best: bool = False,
    ) -> Path:
        state = self._build_state(model, optimizer, epoch, metrics, config)

        latest_path = self.checkpoint_dir / "checkpoint_latest.pt"
        torch.save(state, latest_path)

        saved_path = latest_path

        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pt"
            torch.save(state, best_path)
            logger.info(
                "Saved new best checkpoint at epoch %d with %s: %.4f",
                epoch,
                self.monitor,
                metrics.get(self.monitor, 0.0),
            )
            saved_path = best_path

        # Secondary monitor (e.g. val_pneumonia_auroc)
        if self.secondary_monitor and self.secondary_monitor in metrics:
            sec_score = metrics[self.secondary_monitor]
            if sec_score > self.best_secondary_score:
                self.best_secondary_score = sec_score
                sec_path = self.checkpoint_dir / "checkpoint_best_pneumonia.pt"
                torch.save(state, sec_path)
                logger.info(
                    "Saved new best secondary checkpoint at epoch %d with %s: %.4f",
                    epoch,
                    self.secondary_monitor,
                    sec_score,
                )

        return saved_path

