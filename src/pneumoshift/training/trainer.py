"""Training engine with mixed precision, gradient clipping, and two-stage fine-tuning."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from pneumoshift.data.harmonize_labels import SHARED_CLASSES
from pneumoshift.training.callbacks import EarlyStopping, ModelCheckpoint
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
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.config.get("train", {}).get("mixed_precision", True) and self.device.type == "cuda"
        )

        checkpoint_dir = self.config.get("train", {}).get("checkpoint_dir", "results/runs/checkpoints")
        self.checkpoint_cb = ModelCheckpoint(checkpoint_dir=checkpoint_dir)
        patience = self.config.get("train", {}).get("early_stopping_patience", 6)
        self.early_stopping = EarlyStopping(patience=patience, mode="max")

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

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(images)
                loss = self.criterion(logits, targets, masks)
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
    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
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
            loss = self.criterion(logits, targets, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        if not all_preds:
            return {"loss": 0.0, "val_macro_auroc": 0.0, "val_pneumonia_auroc": 0.0}

        y_pred = np.vstack(all_preds)
        y_true = np.vstack(all_targets)

        # Calculate AUROCs per class
        aurocs = []
        metrics: Dict[str, float] = {"loss": total_loss / max(len(loader), 1)}

        for idx, col in enumerate(self.label_cols):
            # Mask out any ignored values (-1)
            valid_mask = y_true[:, idx] != -1.0
            col_true = y_true[valid_mask, idx]
            col_pred = y_pred[valid_mask, idx]

            if len(np.unique(col_true)) == 2:
                auc = float(roc_auc_score(col_true, col_pred))
            else:
                auc = 0.5
            aurocs.append(auc)
            metrics[f"auroc_{col.lower().replace(' ', '_')}"] = auc

        metrics["val_macro_auroc"] = float(np.mean(aurocs))
        pneu_idx = self.label_cols.index("Pneumonia") if "Pneumonia" in self.label_cols else 5
        metrics["val_pneumonia_auroc"] = aurocs[pneu_idx]

        return metrics

    def fit(self, epochs: int) -> Dict[str, List[float]]:
        logger.info("Starting training on device: %s for %d epochs.", self.device, epochs)
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_macro_auroc": [],
            "val_pneumonia_auroc": [],
        }

        two_stage = self.config.get("train", {}).get("two_stage_finetune", False)
        freeze_epochs = self.config.get("train", {}).get("freeze_epochs", 3)

        if two_stage and hasattr(self.model, "freeze_backbone"):
            self.model.freeze_backbone()

        for epoch in range(1, epochs + 1):
            if two_stage and epoch == freeze_epochs + 1 and hasattr(self.model, "unfreeze_backbone"):
                self.model.unfreeze_backbone()

            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader)

            val_macro = val_metrics.get("val_macro_auroc", 0.0)
            val_pneu = val_metrics.get("val_pneumonia_auroc", 0.0)
            val_loss = val_metrics.get("loss", 0.0)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_macro_auroc"].append(val_macro)
            history["val_pneumonia_auroc"].append(val_pneu)

            logger.info(
                "Epoch %02d/%02d | Train Loss: %.4f | Val Loss: %.4f | Macro AUROC: %.4f | Pneumonia AUROC: %.4f",
                epoch,
                epochs,
                train_loss,
                val_loss,
                val_macro,
                val_pneu,
            )

            # Scheduler step
            if self.lr_scheduler:
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_macro)
                else:
                    self.lr_scheduler.step()

            # Checkpoint & Early stopping
            is_best = val_macro > self.checkpoint_cb.best_score
            if is_best:
                self.checkpoint_cb.best_score = val_macro

            self.checkpoint_cb.save(
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch,
                metrics=val_metrics,
                config=self.config,
                is_best=is_best,
            )

            self.early_stopping(val_macro)
            if self.early_stopping.should_stop:
                logger.info("Stopping early at epoch %d.", epoch)
                break

        return history

