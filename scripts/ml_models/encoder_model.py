#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unicodedata import bidirectional
import torch
from torch import nn
from torch.autograd import Variable


class EncoderModel(nn.Module):
    """Class used to represent a simple RNN implementation in Pytorch for Radar

    Parameters:
        input_size (int): The size of the input data array
        hidden_size (int): Number of hidden layers
        output_size (int): The size of the output data array
    """

    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, latent_dim: int
    ) -> None:
        super(EncoderModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.setup_layers(input_size, hidden_size, num_layers, latent_dim)

    def setup_layers(self, input_size, hidden_size, num_layers, latent_dim):
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Conv1d(in_channels=2, out_channels=1, kernel_size=100))
        self.layers.append(
            nn.GRU(
                input_size - 99,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=0,
                bidirectional=True,
            )
        )
        self.layers.append(
            nn.GRU(
                hidden_size * 2,
                latent_dim,
                num_layers,
                batch_first=True,
                dropout=0,
                bidirectional=False,
            )
        )

    def forward(self, X):
        X = X.view(X.size(0), X.size(2), X.size(1))

        output = self.layers[0](X.float())
        output, _ = self.layers[1](output)
        output, _ = self.layers[2](output)

        return output[:, -1, :]
