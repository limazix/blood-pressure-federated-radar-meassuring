#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import OrderedDict
import torch
import flwr as fl


class FLLocalAgent(fl.client.NumPyClient):
    def __init__(self, trainer, train_loader, test_loader):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainer = trainer

    def get_parameters(self):
        params = []
        for layer in self.trainer.model.layers:
            params = params + _get_parameters(layer)
        return params

    def set_parameters(self, parameters):
        start = 0
        for layer in self.trainer.model.layers:
            end = start + len(layer.state_dict().keys())
            _set_parameters(layer, parameters[start:end])
            start = end

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.trainer.train(
            epochs=int(config["setup"]["epochs"]), train_loader=self.train_loader
        )
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, mse = self.trainer.test(self.test_loader)
        return loss, len(self.test_loader.dataset), {"loss": loss, "mse": mse}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
