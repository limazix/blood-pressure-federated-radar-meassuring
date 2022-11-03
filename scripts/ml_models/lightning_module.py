#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, R2Score, SignalNoiseRatio


class LightningModule(pl.LightningModule):
    """Class used to group Pytorch modules into a Lightning module

    Parameters:
        model (nn.Module): Pytorch module instance
    """

    def __init__(self, model: nn.Module, loss, optimizer, lr, example_input_array) -> None:
        super(LightningModule, self).__init__()
        self.save_hyperparameters(ignore=["model", "loss"])
        self.model = model
        self.layers = model.layers
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.example_input_array = example_input_array
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score(num_outputs=self.model.output_size)
        self.test_snr = SignalNoiseRatio()
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score(num_outputs=self.model.output_size)
        self.val_snr = SignalNoiseRatio()


    def forward(self, X):
        """Method used to convert the in-phase (I) and quadrature (Q) radar signals to the correspondent blood pressure

        Parameters:
            X (array): Bi-dimensional array with I and Q values
        Returns:
            (array): Uni-dimensional array of blood pressures
        """
        return self.model(X)

    def configure_optimizers(self):
        optimizer = self.optimizer(params=list(self.parameters()), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, test_batch, batch_idx):
        self._evaluate(
            test_batch, [self.val_mse, self.val_r2, self.val_snr], "val"
        )

    def test_step(self, test_batch, batch_idx):
        self._evaluate(
            test_batch,
            [self.test_mse, self.test_r2, self.test_snr],
            "test",
        )

    def _evaluate(self, batch, metrics, stage=None):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        logs = {f"{stage}_loss": loss}
        if stage:
            for index, metric in enumerate(metrics):
                metric(out, y)
                metric_name = "mse"
                if index == 1:
                    metric_name = "r2"
                elif index == 2:
                    metric_name = "snr"
                logs[f"{stage}_{metric_name}"] = metric
            self.log_dict(logs, prog_bar=True, on_epoch=True, on_step=False if stage == "val" else True)
