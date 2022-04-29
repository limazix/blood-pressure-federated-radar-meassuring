#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class DecoderModel(nn.Module):
    """Class used to represent a simple RNN implementation in Pytorch for Radar

    Parameters:
        input_size (int): The size of the input data array
        hidden_size (int): Number of hidden layers
        output_size (int): The size of the output data array
    """

    def __init__(
        self,
        hidden_size: int,
        latent_dim: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super(DecoderModel, self).__init__()
        self.hidden_size = hidden_size
        self.setup_layers(hidden_size, latent_dim, num_layers, output_size)

    def setup_layers(self, hidden_size, latent_dim, num_layers, output_size):
        self.layers = nn.ModuleList([])
        self.layers.append(
            nn.GRU(
                latent_dim, latent_dim, num_layers, batch_first=True, dropout=0, bidirectional=False
            )
        )
        self.layers.append(
            nn.GRU(
                latent_dim, hidden_size, num_layers, batch_first=True, dropout=0, bidirectional=True
            )
        )
        self.layers.append(nn.Linear(hidden_size * 2, output_size))

    def forward(self, X):

        output, _ = self.layers[0](X)
        output, _ = self.layers[1](output)

        return self.layers[2](output)
