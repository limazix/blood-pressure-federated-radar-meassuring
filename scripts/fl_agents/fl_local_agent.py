#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import flwr as fl
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    DeviceStatsMonitor,
)

from pl_bolts.callbacks import ModuleDataMonitor

from callbacks.sampler_callback import SamplerCallback
from utils.configurator import config as configurator


class FLLocalAgent(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, aid):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.aid = aid

    def set_trainer(self, current_round):
        self.trainer = pl.Trainer(
            logger=TensorBoardLogger(
                save_dir=".", sub_dir=f"round_{current_round}/{self.aid}", version=configurator["setup"]["version"]
            ),
            callbacks=[
                ModuleDataMonitor(log_every_n_steps=50),
                DeviceStatsMonitor(),
                ModelCheckpoint(monitor="val_loss"),
                LearningRateMonitor(logging_interval="step"),
                EarlyStopping(monitor="val_loss", mode="min", patience=15),
                SamplerCallback(sample_data=next(iter(self.train_loader))),
            ],
            max_epochs=int(configurator["setup"]["epochs"]),
            enable_progress_bar=False,
            gradient_clip_val=0.5,
        )

    def get_parameters(self, config={}):
        params = []
        for layer in self.model.layers:
            params = params + _get_parameters(layer)
        return params

    def set_parameters(self, parameters):
        start = 0
        for layer in self.model.layers:
            end = start + len(layer.state_dict().keys())
            _set_parameters(layer, parameters[start:end])
            start = end

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        self.set_trainer(current_round=config.server_round)

        self.trainer.fit(
            self.model,
            train_dataloaders=self.train_loader,
            val_dataloaders=self.val_loader,
        )

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        results = self.trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss_epoch"]
        mse = results[0]["test_mse"]
        r2 = results[0]["test_r2"]
        explained_variance = results[0]["test_explaimed_variance"]

        return (
            loss,
            len(self.test_loader.dataset),
            {
                "loss": loss,
                "mse": mse,
                "r2": r2,
                "explained_variance": explained_variance,
            },
        )


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
