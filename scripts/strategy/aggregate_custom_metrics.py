#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flwr as fl
from torch.utils.tensorboard import SummaryWriter


class AggregateCustomMetricStrategy(fl.server.strategy.FedAvg):
    def aggregate_evaluate(
        self,
        rnd: int,
        results,
        failures,
    ):
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None

        writer = SummaryWriter("lightning_logs")
        # Weigh accuracy of each client by number of examples used
        mse = [r.metrics["mse"] * r.num_examples for _, r in results]
        r2 = [r.metrics["r2"] * r.num_examples for _, r in results]
        explained_variance = [
            r.metrics["explained_variance"] * r.num_examples for _, r in results
        ]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        mse_aggregated = sum(mse) / sum(examples)
        r2_aggregated = sum(r2) / sum(examples)
        explained_variance_agg = sum(explained_variance) / sum(examples)

        writer.add_scalar("mse_agg", scalar_value=mse_aggregated, global_step=rnd)
        writer.add_scalar("r2_agg", scalar_value=r2_aggregated, global_step=rnd)
        writer.add_scalar(
            "explained_variance_agg",
            scalar_value=explained_variance_agg,
            global_step=rnd,
        )

        writer.close()

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)

