#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD

import pytorch_lightning as pl


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
        lr: float = 0.0001,
    ) -> None:
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers, batch_first=True, nonlinearity="relu"
        )
        self.fc = nn.Linear(hidden_size, output_size)
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()

    def forward(self, X):
        """Method used to convert the in-phase (I) and quadrature (Q) radar signals to the correspondent blood pressure

        Parameters:
            X (array): Bi-dimensional array with I and Q values
        Returns:
            (array): Uni-dimensional array of blood pressures
        """
        h0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size))
        input = torch.reshape(X, (-1,))
        out, _ = self.rnn(input, h0)
        out = self.fc(out[:, -1, :])
        return out

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        out = self.forward(x)
        loss = self.loss(out, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, valid_batch, batch_idx):
        self._evaluate(valid_batch, "validation")

    def test_step(self, test_batch, batch_idx):
        self._evaluate(test_batch, "test")

    def _evaluate(self, batch, stage=None):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
