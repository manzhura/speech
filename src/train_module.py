import torch
import torch.nn as nn
import lightning as L
import torchmetrics

from src.model import MelNet


class MelModul(L.LightningModule):
    def __init__(self, n_mels = 80, n_groups=1):
        super().__init__()
        self._model = MelNet(n_mels=n_mels, groups=n_groups)
        self._criterion = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.classification.Accuracy(task="binary")
        self.val_acc = torchmetrics.classification.Accuracy(task="binary")
        self.test_acc = torchmetrics.classification.Accuracy(task="binary")

        self.save_hyperparameters(ignore=["_model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y.long())

        preds = logits.argmax(dim=1)
        self.train_acc(preds, y.int())

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y.long())

        preds = logits.argmax(dim=1)
        self.val_acc(preds, y.int())

        self.log("valid_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("valid_acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        x, y = batch
        logits = self(x)
        loss = self._criterion(logits, y.long())

        preds = logits.argmax(dim=1)
        self.test_acc(preds, y.int())

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)