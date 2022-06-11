#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

import pytorch_lightning as pl
from torchmetrics import MeanSquaredError, R2Score


class LightningModule(pl.LightningModule):
    """Class used to group Pytorch modules into a Lightning module

    Parameters:
        model (nn.Module): Pytorch module instance
    """

    def __init__(self, model: nn.Module, loss, optimizer, lr) -> None:
        super(LightningModule, self).__init__()
        self.model = model
        self.layers = model.layers
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.val_mse = MeanSquaredError()
        self.val_r2 = R2Score(num_outputs=self.model.output_size)
        self.test_mse = MeanSquaredError()
        self.test_r2 = R2Score(num_outputs=self.model.output_size)

    def forward(self, X):
        """Method used to convert the in-phase (I) and quadrature (Q) radar signals to the correspondent blood pressure

        Parameters:
            X (array): Bi-dimensional array with I and Q values
        Returns:
            (array): Uni-dimensional array of blood pressures
        """
        return self.model(X)

    def configure_optimizers(self):
        return self.optimizer(params=list(self.parameters()), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, test_batch, batch_idx):
        self._evaluate(test_batch, [self.val_mse, self.val_r2], "val")

    def test_step(self, test_batch, batch_idx):
        self._evaluate(test_batch, [self.test_mse, self.test_r2], "test")

    def _evaluate(self, batch, metrics, stage=None):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            for index, metric in enumerate(metrics):
                metric(y, out)
                metric_name = "mse" if index == 0 else "r2"
                self.log(f"{stage}_{metric_name}", metric, prog_bar=True)
