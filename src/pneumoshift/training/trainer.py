"""Training engine with mixed precision, gradient clipping, and two-stage fine-tuning."""

from pathlib import Path
import random
import time
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.training.callbacks import EarlyStopping, ModelCheckpoint, save_atomic_checkpoint
from pneumoshift.training.metrics_logger import EpochMetricsLogger
from pneumoshift.utils.logging import get_logger

logger = get_logger("pneumoshift.training.trainer")


class Trainer:
    """Orchestrates model training, validation, early stopping, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[Any] = None,
        device: Optional[torch.device] = None,
        config: Optional[Dict[str, Any]] = None,
        label_cols: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config or {}
        self.label_cols = label_cols or SHARED_CLASSES

        self.model.to(self.device)
        if isinstance(self.criterion, nn.Module):
            self.criterion.to(self.device)

        use_amp = self.config.get("train", {}).get("mixed_precision", True) and self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        checkpoint_dir = self.config.get("train", {}).get("checkpoint_dir", "results/runs/checkpoints")
        primary_monitor = self.config.get("train", {}).get("save_best_metric", "val_macro_auroc")
        secondary_monitor = self.config.get("train", {}).get("save_headline_metric", "val_pneumonia_auroc")
        self.checkpoint_cb = ModelCheckpoint(
            checkpoint_dir=checkpoint_dir,
            monitor=primary_monitor,
            secondary_monitor=secondary_monitor,
        )
        patience = self.config.get("train", {}).get("early_stopping_patience", 6)
        self.early_stopping = EarlyStopping(patience=patience, mode="max")

        run_output_dir = self.config.get("train", {}).get("run_output_dir", "results/runs")
        self.metrics_logger = EpochMetricsLogger(output_dir=run_output_dir)
        self.best_epoch: int = 1
        self.stopped_due_to_timeout: bool = False

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        clip_norm = self.config.get("train", {}).get("clip_grad_norm", 1.0)
        grad_accum_steps = self.config.get("train", {}).get("gradient_accumulation_steps", 1)

        self.optimizer.zero_grad()
        for step, batch in enumerate(self.train_loader):
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(self.device)

            amp_ctx = (
                torch.amp.autocast("cuda", enabled=self.scaler.is_enabled())
                if hasattr(torch, "amp") and hasattr(torch.amp, "autocast")
                else torch.cuda.amp.autocast(enabled=self.scaler.is_enabled())
            )
            with amp_ctx:
                logits = self.model(images)
                if masks is not None:
                    try:
                        loss = self.criterion(logits, targets, masks)
                    except TypeError:
                        loss = self.criterion(logits, targets)
                else:
                    try:
                        loss = self.criterion(logits, targets)
                    except TypeError:
                        loss = self.criterion(logits, targets, None)
                if grad_accum_steps > 1:
                    loss = loss / grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(self.train_loader):
                if clip_norm > 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clip_norm)

                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

            total_loss += loss.item() * grad_accum_steps

        return total_loss / max(len(self.train_loader), 1)

    @torch.no_grad()
    def evaluate(
        self,
        loader: torch.utils.data.DataLoader,
        uncertainty_mode: str = "uncertain_excluded",
    ) -> Dict[str, float]:
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0

        for batch in loader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            masks = batch.get("mask", None)
            if masks is not None:
                masks = masks.to(self.device)

            logits = self.model(images)
            if masks is not None:
                try:
                    loss = self.criterion(logits, targets, masks)
                except TypeError:
                    loss = self.criterion(logits, targets)
            else:
                try:
                    loss = self.criterion(logits, targets)
                except TypeError:
                    loss = self.criterion(logits, targets, None)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        if not all_preds:
            return {"loss": 0.0, "val_macro_auroc": 0.0, "val_pneumonia_auroc": 0.0}

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        # Calculate AUROCs and AUPRCs per class
        aurocs = []
        auprcs = []
        metrics: Dict[str, float] = {
            "loss": total_loss / max(len(loader), 1),
            "uncertainty_mode": uncertainty_mode,
        }

        for idx, col in enumerate(self.label_cols):
            # Process uncertain (-1) values according to uncertainty_mode
            if uncertainty_mode == "uncertain_as_negative":
                col_true = y_true[:, idx].copy()
                col_true[col_true == -1.0] = 0.0
                col_pred = y_pred[:, idx]
            else:  # "uncertain_excluded"
                valid_mask = y_true[:, idx] != -1.0
                col_true = y_true[valid_mask, idx]
                col_pred = y_pred[valid_mask, idx]

            col_key = col.lower().replace(" ", "_")

            if len(np.unique(col_true)) == 2:
                auc = float(roc_auc_score(col_true, col_pred))
                ap = float(average_precision_score(col_true, col_pred))
            else:
                auc = 0.5
                ap = 0.0
            aurocs.append(auc)
            auprcs.append(ap)
            metrics[f"auroc_{col_key}"] = auc
            metrics[f"auprc_{col_key}"] = ap

        metrics["val_macro_auroc"] = float(np.mean(aurocs))
        metrics["val_macro_auprc"] = float(np.mean(auprcs))
        pneu_idx = self.label_cols.index("Pneumonia") if "Pneumonia" in self.label_cols else 5
        metrics["val_pneumonia_auroc"] = aurocs[pneu_idx]
        metrics["val_pneumonia_auprc"] = auprcs[pneu_idx]

        return metrics

    def fit(
        self,
        epochs: int,
        start_epoch: int = 1,
        max_hours: Optional[float] = None,
        start_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        logger.info("Starting training on device: %s from epoch %d to %d.", self.device, start_epoch, epochs)
        history: Dict[str, Any] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_auroc": [],
            "val_macro_auprc": [],
            "val_pneumonia_auroc": [],
            "val_pneumonia_auprc": [],
            "best_epoch": self.best_epoch,
        }

        two_stage = self.config.get("train", {}).get("two_stage_finetune", False)
        freeze_epochs = self.config.get("train", {}).get("freeze_epochs", 3)
        backbone_lr = float(self.config.get("train", {}).get("backbone_lr", 0.0))

        if two_stage and hasattr(self.model, "freeze_backbone"):
            if start_epoch <= freeze_epochs:
                self.model.freeze_backbone()

        for epoch in range(start_epoch, epochs + 1):
            if max_hours is not None and start_time is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= max_hours:
                    logger.warning(
                        "Max hours limit reached before epoch %d (%.2f / %.2f hours). Exiting cleanly.",
                        epoch,
                        elapsed_hours,
                        max_hours,
                    )
                    self.stopped_due_to_timeout = True
                    break

            if two_stage and epoch == freeze_epochs + 1 and hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()
                # Restore backbone learning rate in the optimizer param group
                for pg in self.optimizer.param_groups:
                    if pg.get("name") == "backbone":
                        head_lr = float(self.config.get("train", {}).get("learning_rate", 1e-4))
                        pg["lr"] = backbone_lr if backbone_lr > 0 else head_lr * 0.1
                        logger.info(
                            "Set backbone LR to %.2e after unfreezing at epoch %d.",
                            pg["lr"],
                            epoch,
                        )

            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            val_macro = val_metrics.get("val_macro_auroc", 0.0)
            val_pneu = val_metrics.get("val_pneumonia_auroc", 0.0)
            val_loss = val_metrics.get("loss", 0.0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_auroc"].append(val_macro)
            history["val_macro_auprc"].append(val_metrics.get("val_macro_auprc", 0.0))
            history["val_pneumonia_auroc"].append(val_pneu)
            history["val_pneumonia_auprc"].append(val_metrics.get("val_pneumonia_auprc", 0.0))

            monitor_score = val_metrics.get(self.checkpoint_cb.monitor, val_macro)

            # Scheduler step
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(monitor_score)
                else:
                    self.lr_scheduler.step()

            # Checkpoint & Early stopping
            is_best = monitor_score > self.checkpoint_cb.best_score
            if is_best:
                self.checkpoint_cb.best_score = monitor_score
                self.best_epoch = epoch

            logger.info(
                "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Macro AUROC: %.4f | Pneumonia AUROC: %.4f | Best: %s",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_macro,
                val_pneu,
                "YES" if is_best else "NO",
            )

            # Log all metrics (train loss + full validation metrics + selected_as_best) to CSV/JSON
            epoch_row = {"train_loss": train_loss, **val_metrics, "selected_as_best": is_best}
            self.metrics_logger.log(epoch=epoch, metrics=epoch_row)

            # Capture extra states for atomic resume support
            extra_state = {
                "scheduler_state_dict": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                "scaler_state_dict": self.scaler.state_dict() if hasattr(self, "scaler") and self.scaler else None,
                "best_epoch": self.best_epoch,
                "best_score": self.checkpoint_cb.best_score,
                "best_secondary_score": self.checkpoint_cb.best_secondary_score,
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "rng_state_numpy": np.random.get_state(),
                "rng_state_python": random.getstate(),
            }

            self.checkpoint_cb.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=self.config,
                is_best=is_best,
                extra_state=extra_state,
            )

            # Atomically save last.pt for uninterrupted resume capability
            last_path = self.checkpoint_cb.checkpoint_dir / "last.pt"
            state = self.checkpoint_cb._build_state(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=self.config,
                extra_state=extra_state,
            )
            save_atomic_checkpoint(state, last_path)

            if max_hours is not None and start_time is not None:
                elapsed_hours = (time.time() - start_time) / 3600.0
                if elapsed_hours >= max_hours:
                    logger.warning(
                        "Max hours limit reached after epoch %d (%.2f / %.2f hours). Exiting cleanly.",
                        epoch,
                        elapsed_hours,
                        max_hours,
                    )
                    self.stopped_due_to_timeout = True
                    break

            self.early_stopping(monitor_score)
            if self.early_stopping.should_stop:
                logger.info("Stopping early at epoch %d.", epoch)
                break

        history["best_epoch"] = self.best_epoch
        history["stopped_due_to_timeout"] = self.stopped_due_to_timeout
        return history

