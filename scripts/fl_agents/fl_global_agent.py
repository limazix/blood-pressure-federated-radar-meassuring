#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flwr as fl

from utils.configurator import config
from strategy.aggregate_custom_metrics import AggregateCustomMetricStrategy

def run_global_agent() -> None:
    # Define strategy
    strategy = AggregateCustomMetricStrategy(
        fraction_fit=0.5,
        fraction_eval=0.5,
    )

    # Start Flower server for three rounds of federated learning
    fl.server.start_server(
        server_address="{}:{}".format(
            config["server"]["hostname"], config["server"]["port"]
        ),
        config={"num_rounds": 3},
        strategy=strategy,
    )
