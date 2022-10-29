#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import plotly.express as px
import pytorch_lightning as pl


class SamplerCallback(pl.Callback):
    def __init__(self, sample_data, every_n_epochs=5):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.sample_data = sample_data

    def on_test_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch % self.every_n_epochs == 0:

            X, Y = self.sample_data
            with torch.no_grad():
                pl_module.eval()
                results = pl_module(X)
                pl_module.train()

            for idx, y_hat in enumerate(results):
                trainer.logger.experiment.add_scalars(
                    f"epoch_{trainer.current_epoch}",
                    {"expected": Y[idx], "result": y_hat},
                    idx,
                )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch == 1:
            X, Y = self.sample_data
            trainer.logger.experiment.add_graph(
                pl_module, input_to_model=X
            )

        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)
