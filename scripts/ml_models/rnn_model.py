#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD


import pytorch_lightning as pl
from torchmetrics import R2Score


class RNNModel(pl.LightningModule):
    """Class used to represent a simple RNN implementation in Pytorch for Radar

    Parameters:
        input_size (int): The size of the input data array
        hidden_size (int): Number of hidden layers
        output_size (int): The size of the output data array
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        lr: float = 0.001,
    ) -> None:
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.val_r2 = R2Score(num_outputs=output_size)
        self.test_r2 = R2Score(num_outputs=output_size)
        self.setup_layers(input_size, hidden_size, num_layers, output_size)

    def setup_layers(self, input_size, hidden_size, num_layers, output_size):
        conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=100)
        rnn = nn.RNN(
            input_size - 99,
            hidden_size,
            num_layers,
            batch_first=True,
            nonlinearity="relu",
        )
        fc = nn.Linear(hidden_size, output_size)
        self.layers = [conv, rnn, fc]

    def forward(self, X):
        """Method used to convert the in-phase (I) and quadrature (Q) radar signals to the correspondent blood pressure

        Parameters:
            X (array): Bi-dimensional array with I and Q values
        Returns:
            (array): Uni-dimensional array of blood pressures
        """
        h0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size))
        X = X.view(X.size(0), X.size(2), X.size(1))

        X = self.layers[0](X)

        out, _ = self.layers[1](X, h0)

        out = self.layers[2](out[:, -1, :])
        return out

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        self._evaluate(valid_batch, self.val_r2, "validation")

    def test_step(self, test_batch, batch_idx):
        self._evaluate(test_batch, self.test_r2, "test")

    def _evaluate(self, batch, metric, stage=None):
        x, y = batch
        out = self(x)
        loss = self.loss(out, y)
        metric(y, out)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_r2", metric, prog_bar=True)
