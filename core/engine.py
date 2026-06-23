"""
Training engine for neural network models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
import time
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training session."""
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    validation_split: float = 0.2
    early_stopping_patience: int = 5
    l1_lambda: float = 0.0
    l2_lambda: float = 0.0


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float


class TrainingEngine:
    """Handles model training, validation, and metric collection."""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.metrics_history: List[TrainingMetrics] = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self._build_criterion()

    def _build_criterion(self):
        loss_map = {
            "cross_entropy": nn.CrossEntropyLoss(),
            "mse": nn.MSELoss(),
            "bce": nn.BCEWithLogitsLoss(),
            "mae": nn.L1Loss(),
            "huber": nn.SmoothL1Loss(),
        }
        self.criterion = loss_map.get(self.config.loss_function, nn.CrossEntropyLoss())

    def build_optimizer(self, model: nn.Module):
        opt_map = {
            "adam": optim.Adam,
            "sgd": optim.SGD,
            "adamw": optim.AdamW,
            "rmsprop": optim.RMSprop,
        }
        opt_class = opt_map.get(self.config.optimizer, optim.Adam)
        return opt_class(model.parameters(), lr=self.config.learning_rate)

    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """Train for one epoch, return (loss, accuracy)."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            if self.config.l1_lambda > 0:
                l1 = sum(p.abs().sum() for p in self.model.parameters())
                loss += self.config.l1_lambda * l1
            if self.config.l2_lambda > 0:
                l2 = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss += self.config.l2_lambda * l2
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        return total_loss / len(train_loader), correct / total if total > 0 else 0.0

    def validate(self, val_loader) -> Tuple[float, float]:
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
        return total_loss / len(val_loader), correct / total if total > 0 else 0.0

    def train(self, model: nn.Module, train_loader, val_loader) -> List[TrainingMetrics]:
        """Full training loop with early stopping."""
        self.model = model.to(self.device)
        self.optimizer = self.build_optimizer(model)
        self.metrics_history = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        for epoch in range(self.config.epochs):
            start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            epoch_time = time.time() - start

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss,
                val_loss=val_loss,
                train_accuracy=train_acc,
                val_accuracy=val_acc,
                learning_rate=self.optimizer.param_groups[0]["lr"],
                epoch_time=epoch_time,
            )
            self.metrics_history.append(metrics)

            logger.info(
                f"Epoch {epoch + 1}/{self.config.epochs} - "
                f"train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, "
                f"train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}"
            )

            if val_loss < self.best_val_loss - 1e-4:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return self.metrics_history

    def export_onnx(self, path: str, input_shape: tuple):
        """Export model to ONNX format."""
        self.model.eval()
        dummy = torch.randn(1, *input_shape).to(self.device)
        torch.onnx.export(
            self.model, dummy, path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"Model exported to ONNX: {path}")

    def export_torchscript(self, path: str):
        """Export model to TorchScript."""
        self.model.eval()
        script = torch.jit.script(self.model)
        script.save(path)
        logger.info(f"Model exported to TorchScript: {path}")
