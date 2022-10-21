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

            with torch.no_grad():
                pl_module.eval()
                results = pl_module(self.sample_data)
                pl_module.train()

            for idx, out in enumerate(results):
                y = self.sample_data[idx]
                for pos, y_hat in enumerate(out):
                    trainer.logger.experiment.add_scalars(
                        f"epoch_{trainer.current_epoch}",
                        {"expected": y[pos], "result": y_hat},
                        pos,
                    )

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if trainer.current_epoch == 1:
            trainer.logger.experiment.add_graph(
                pl_module, input_to_model=(self.sample_data)
            )

        for name, params in pl_module.named_parameters():
            trainer.logger.experiment.add_histogram(name, params, trainer.current_epoch)
