#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click
import copy

from torch.utils.data import DataLoader
import pytorch_lightning as pl
import flwr as fl

from data_models.subject import Subject
from data_models.subject_dataset import SubjectDataset

from ml_models.rnn_model import RNNModel
from fl_agents.fl_local_agent import FLLocalAgent

from utils.validator import validate_directory_path


def build_dataloader(dataset, start_index, end_index, batch_size):
    _dataset = copy.deepcopy(dataset)
    _dataset.prune(start_index, end_index)
    return DataLoader(
        dataset=_dataset, batch_size=batch_size, shuffle=False, num_workers=8
    )


def build_model(subject_dataset):
    input_sample, output_sample = subject_dataset[0]
    input_size = len(input_sample)
    hidden_size = int(input_size * 0.8)
    output_size = len(output_sample)
    return RNNModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=4,
        output_size=output_size,
        lr=0.001,
    )


@click.command()
@click.option("--data-dir")
@click.option("--subject-id")
@click.option(
    "--data-split",
    default="70:15:15",
    help="It defines the data split for train:test:validation (ex: 70:15:15)",
)
@click.option("--port")
def main(data_dir, subject_id, data_split: str, port):
    """Method used to run a single flower agent"""
    subject_data_dir = os.path.normpath(os.path.join(data_dir, subject_id))
    validate_directory_path(subject_data_dir)

    subject = Subject(code=subject_id)
    subject.setup(data_dir=subject_data_dir)
    radar, bp = subject.get_all_data()

    subject_dataset = SubjectDataset(
        radar=radar, radar_sr=2000, bp=bp, bp_sr=200, window_size=1, overlap=0.3
    )
    data_size = len(subject_dataset)

    train_size, val_size, test_size = data_split.split(":")
    end_train_index = int((data_size * int(train_size)) / 100)
    end_val_index = int((data_size * (int(train_size) + int(val_size))) / 100)

    train_loader = build_dataloader(
        subject_dataset, start_index=0, end_index=end_train_index, batch_size=32
    )
    val_loader = build_dataloader(
        subject_dataset,
        start_index=end_train_index,
        end_index=end_val_index,
        batch_size=32,
    )
    test_loader = build_dataloader(
        subject_dataset,
        start_index=end_val_index,
        end_index=len(subject_dataset),
        batch_size=32,
    )

    model = build_model(subject_dataset)

    agent = FLLocalAgent(model, train_loader, val_loader, test_loader)
    fl.client.start_numpy_client("localhost:8080", client=agent)


if __name__ == "__main__":
    main()
