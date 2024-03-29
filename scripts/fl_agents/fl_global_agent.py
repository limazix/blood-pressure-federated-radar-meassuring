#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flwr as fl

from utils.configurator import config
from strategy.aggregate_custom_metrics import AggregateCustomMetricStrategy

from builders.data_builder import DataBuilder
from builders.data_loader_builder import DataLoaderBuilder
from builders.model_builder import ModelBuilder

from .fl_local_agent import FLLocalAgent


def get_local_agent(subject_id):
    data_builder = DataBuilder()
    radar, bp = data_builder.get_data(subject_id)

    data_loader_builder = DataLoaderBuilder()
    (
        subject_dataset,
        train_loader,
        val_loader,
        test_loader,
    ) = data_loader_builder.build_loaders(radar, bp)

    model_builder = ModelBuilder()
    model = model_builder.build(subject_dataset)

    return FLLocalAgent(model, train_loader, val_loader, test_loader, aid=subject_id)

def get_experiment_config(server_round):
    return {"server_round": server_round}

def run_simulation() -> None:

    subjects_ids = [
        "GDN00{}".format(str(i + 1).zfill(2))
        for i in range(int(config["subjects"]["num_subjects"]))
    ]

    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.2,
        fraction_evaluate=0.1,
        on_fit_config_fn=get_experiment_config,
    )

    fl.simulation.start_simulation(
        client_fn=get_local_agent,
        clients_ids=subjects_ids,
        client_resources={"num_cpus": 4},
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        ray_init_args={"include_dashboard": False},
    )
