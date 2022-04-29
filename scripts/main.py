#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import click

import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torch.optim import Adam
from torch import nn

import flwr as fl
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data_models.subject import Subject
from data_models.subject_dataset import SubjectDataset

from data_transforms.to_tensor import ToTensor
from data_transforms.normalize import Normalize

from ml_models.lightning_module import LightningModule
from ml_models.autoencoder_model import AutoencoderModel

from fl_agents.fl_local_agent import FLLocalAgent
from fl_agents.fl_global_agent import run_global_agent

from utils.configurator import config
from utils.validator import validate_directory_path


def build_dataloader(dataset, start_index, end_index):
    _dataset = copy.deepcopy(dataset)
    _dataset.prune(start_index, end_index)
    return DataLoader(
        dataset=_dataset,
        shuffle=False,
        num_workers=int(config["dataloader"]["num_workers"]),
        batch_size=int(config["dataloader"]["batch_size"]),
    )


def build_model(subject_dataset):
    input_sample, output_sample = subject_dataset[0]
    input_size = len(input_sample)
    hidden_size = int(input_size * 0.8)
    latent_dim = int(hidden_size * 0.5)
    num_layers = 2
    output_size = len(output_sample)

    model = AutoencoderModel(
        input_size, hidden_size, latent_dim, num_layers, output_size
    )

    return LightningModule(
        model=model,
        loss=nn.L1Loss(),
        optimizer=Adam,
        lr=float(config["setup"]["learn_rate"]),
    )


def get_subject_data(subject_id):
    subject_data_dir = os.path.abspath(
        os.path.normpath(os.path.join(config["setup"]["datadir"], subject_id))
    )

    if os.path.isfile(subject_data_dir):
        return None

    subject = Subject(code=subject_id)
    subject.setup(data_dir=subject_data_dir)
    return subject.get_all_data()


def get_data(subject_id=None):
    radar = None
    bp = None
    if subject_id is not None:
        radar, bp = get_subject_data(subject_id)
    else:
        data_dir = os.path.abspath(os.path.normpath(config["setup"]["datadir"]))
        validate_directory_path(data_dir)
        for subject_dir in os.listdir(data_dir):
            data = get_subject_data(subject_dir)
            if data is not None:
                radar = (
                    np.concatenate([radar, data[0]], axis=1)
                    if radar is not None
                    else data[0]
                )
                bp = np.concatenate([bp, data[1]]) if bp is not None else data[1]

    # replace nan with mean
    radar = np.nan_to_num(radar, nan=np.nanmean(radar, axis=0))
    bp = np.nan_to_num(bp, nan=np.nanmean(bp, axis=0))

    return radar.T, bp


def build_loaders(radar, bp):
    radar_mean, radar_std = np.mean(radar, axis=0), np.std(radar, axis=0)
    bp_mean, bp_std = np.mean(bp, axis=0), np.std(bp, axis=0)

    dataset = SubjectDataset(
        radar=radar,
        radar_sr=int(config["dataset"]["radar_sr"]),
        bp=bp,
        bp_sr=int(config["dataset"]["bp_sr"]),
        window_size=float(config["dataset"]["window_size"]),
        overlap=float(config["dataset"]["overlap"]),
        transform=Compose([ToTensor(), Normalize(mean=radar_mean, std=radar_std)]),
        target_transform=Compose([ToTensor(), Normalize(mean=bp_mean, std=bp_std)]),
    )
    data_size = len(dataset)

    end_train_index = int(data_size * float(config["setup"]["train_size"]))
    end_val_index = end_train_index + int(
        data_size * float(config["setup"]["val_size"])
    )

    train_loader = build_dataloader(dataset, start_index=0, end_index=end_train_index)
    val_loader = build_dataloader(
        dataset, start_index=end_train_index, end_index=end_val_index
    )
    test_loader = build_dataloader(
        dataset, start_index=end_val_index, end_index=data_size
    )
    return dataset, train_loader, val_loader, test_loader


def run_lightning():
    radar, bp = get_data()
    dataset, train_loader, val_loader, test_loader = build_loaders(radar, bp)
    model = build_model(dataset)

    trainer = pl.Trainer(
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=3)],
        max_epochs=int(config["setup"]["epochs"]),
        enable_progress_bar=True,
        gradient_clip_val=0.5,
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(model, test_loader)


def run_local_agent(subject_id):
    radar, bp = get_data(subject_id)
    subject_dataset, train_loader, val_loader, test_loader = build_loaders(radar, bp)
    model = build_model(subject_dataset)

    agent = FLLocalAgent(model, train_loader, val_loader, test_loader)
    fl.client.start_numpy_client(
        "{}:{}".format(config["server"]["hostname"], config["server"]["port"]),
        client=agent,
    )


@click.command()
@click.option("--is-federated", is_flag=True)
@click.option("--is-global", is_flag=True)
@click.option("--subject-id")
def main(is_federated, is_global, subject_id):
    if is_federated:
        if not is_global:
            run_local_agent(subject_id)
        else:
            run_global_agent()
    else:
        run_lightning()


if __name__ == "__main__":
    main()
