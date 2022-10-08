#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

import flwr as fl
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data_models.data_builder import DataBuilder

from data_transforms.data_loader_builder import DataLoaderBuilder

from ml_models.model_builder import ModelBuilder

from fl_agents.fl_local_agent import FLLocalAgent
from fl_agents.fl_global_agent import run_global_agent

from utils.configurator import config


def run_lightning():
    data_builder = DataBuilder()
    radar, bp = data_builder.get_data()

    data_loader_builder = DataLoaderBuilder()
    dataset, train_loader, val_loader, test_loader = data_loader_builder.build_loaders(
        radar, bp
    )

    model_builder = ModelBuilder()
    model = model_builder.build(dataset)

    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        max_epochs=int(config["setup"]["epochs"]),
        enable_progress_bar=True,
        gradient_clip_val=0.5,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)


def run_local_agent(subject_id):
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

    agent = FLLocalAgent(model, train_loader, val_loader, test_loader)
    fl.client.start_numpy_client(
        server_address="{}:{}".format(
            config["server"]["hostname"], config["server"]["port"]
        ),
        client=agent,
        grpc_max_message_length=int(config["server"]["grpc"]),
    )


@click.command()
@click.option("--is-federated", is_flag=True)
@click.option("--is-global", is_flag=True)
@click.option("--subject-id", default=None)
def main(is_federated, is_global, subject_id):
    if is_federated:
        if not is_global:
            subject_id = os.environ.get("SUBJECT_ID", subject_id)
            run_local_agent(subject_id)
        else:
            run_global_agent()
    else:
        run_lightning()


if __name__ == "__main__":
    main()
