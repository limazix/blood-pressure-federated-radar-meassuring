#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch.optim import SGD
from torch import nn

from utils.configurator import config

from ml_models.autoencoder_model import AutoencoderModel
from ml_models.lightning_module import LightningModule


class ModelBuilder:
    def build(self, subject_dataset):
        input_sample, output_sample = subject_dataset[0]
        input_size = len(input_sample)
        hidden_size = int(input_size * 0.8)
        latent_dim = int(hidden_size * 0.5)
        num_layers = 2
        output_size = len(output_sample)

        model = AutoencoderModel(
            input_size, hidden_size, latent_dim, num_layers, output_size
        )

        return LightningModule(
            model=model,
            loss=nn.L1Loss(),
            optimizer=SGD,
            lr=float(config["setup"]["learn_rate"]),
        )
