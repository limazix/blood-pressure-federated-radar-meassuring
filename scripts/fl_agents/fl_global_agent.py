#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flwr as fl

from utils.configurator import config
from strategy.aggregate_custom_metrics import AggregateCustomMetricStrategy


def run_global_agent() -> None:
    # Define strategy
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="{}:{}".format(
            config["server"]["hostname"], config["server"]["port"]
        ),
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        grpc_max_message_length=int(config["server"]["grpc"]),
    )
