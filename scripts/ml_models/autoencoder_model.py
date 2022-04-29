#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn

from .encoder_model import EncoderModel
from .decoder_model import DecoderModel


class AutoencoderModel(nn.Module):
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
        latent_dim: int,
        num_layers: int,
        output_size: int,
    ) -> None:
        super(AutoencoderModel, self).__init__()
        self.output_size = output_size
        self.setup_layers(input_size, hidden_size, latent_dim, num_layers, output_size)

    def setup_layers(
        self, input_size, hidden_size, latent_dim, num_layers, output_size
    ):
        self.layers = nn.ModuleList([])
        self.layers.append(
            EncoderModel(input_size, hidden_size, num_layers, latent_dim)
        )
        self.layers.append(
            DecoderModel(hidden_size, latent_dim, num_layers, output_size)
        )

    def forward(self, X):
        output = self.layers[0](X)
        output = self.layers[1](output)
        return output
