#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import flwr as fl
import pytorch_lightning as pl


class FLLocalAgent(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self):
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

        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False, gradient_clip_val=0.5)
        trainer.fit(self.model, self.train_loader) #, self.val_loader)

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, len(self.test_loader.dataset), {"loss": loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
