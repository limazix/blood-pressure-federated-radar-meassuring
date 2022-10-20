#!/usr/bin/env python
# -*- coding: utf-8 -*-

import click

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichModelSummary, DeviceStatsMonitor

from pl_bolts.callbacks import ModuleDataMonitor

from builders.data_builder import DataBuilder
from builders.data_loader_builder import DataLoaderBuilder
from builders.model_builder import ModelBuilder

from fl_agents.fl_global_agent import run_simulation

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
        logger=TensorBoardLogger(
            save_dir=".", sub_dir="centralized", version=config["setup"]["version"]
        ),
        callbacks=[
            ModuleDataMonitor(log_every_n_steps=50),
            DeviceStatsMonitor(),
            ModelCheckpoint(monitor="val_loss"),
            RichModelSummary(),
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor="val_loss", mode="min", patience=15),
        ],
        max_epochs=int(config["setup"]["epochs"]),
        enable_progress_bar=True,
        gradient_clip_val=0.5,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)


@click.command()
@click.option("--is-federated", is_flag=True)
def main(is_federated=None):
    if is_federated:
        run_simulation()
    else:
        run_lightning()


if __name__ == "__main__":
    main()
