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
        rnn_params = _get_parameters(self.model.rnn)
        fc_params = _get_parameters(self.model.fc)
        return rnn_params + fc_params

    def set_parameters(self, parameters):
        _set_parameters(self.model.rnn, parameters[:-1])
        _set_parameters(self.model.fc, parameters[-1])

    def fit(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(max_epochs=1, progress_bar_refresh_rate=0)
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(), 55000, {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer(progress_bar_refresh_rate=0)
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, 10000, {"loss": loss}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
