#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

import pytorch_lightning as pl
from torchmetrics import MeanSquaredError


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
        self.test_mse = MeanSquaredError()

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
        self._evaluate(test_batch, self.val_mse, "val")

    def test_step(self, test_batch, batch_idx):
        self._evaluate(test_batch, self.test_mse, "test")

    def _evaluate(self, batch, metric, stage=None):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        metric(y, out)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_mse", metric, prog_bar=True)
