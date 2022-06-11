#!/usr/bin/env python
# -*- coding: utf-8 -*-

import flwr as fl


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

        # Weigh accuracy of each client by number of examples used
        mse = [r.metrics["mse"] * r.num_examples for _, r in results]
        r2 = [r.metrics["r2"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Aggregate and print custom metric
        mse_aggregated = sum(mse) / sum(examples)
        r2_aggregated = sum(r2) / sum(examples)

        print(f"Round {rnd} mse aggregated from client results: {mse_aggregated}")
        print(f"Round {rnd} R2 aggregated from client results: {r2_aggregated}")

        # Call aggregate_evaluate from base class (FedAvg)
        return super().aggregate_evaluate(rnd, results, failures)
