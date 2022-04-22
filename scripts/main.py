#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import copy
import click

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import flwr as fl

from data_models.subject import Subject
from data_models.subject_dataset import SubjectDataset

from data_transforms.to_tensor import ToTensor
from data_transforms.butter_transform import ButterTransform
from data_transforms.arctan_demodulation import ArctanDemodulation

from ml_models.rnn_model import RNNModel
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
    output_size = len(output_sample)
    return RNNModel(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=4,
        output_size=output_size,
        lr=float(config["setup"]["learn_rate"]),
    )


def run_local(subject_id):
    subject_data_dir = os.path.abspath(
        os.path.normpath(os.path.join(config["setup"]["datadir"], subject_id))
    )
    validate_directory_path(subject_data_dir)

    subject = Subject(code=subject_id)
    subject.setup(data_dir=subject_data_dir)
    radar, bp = subject.get_all_data()

    subject_dataset = SubjectDataset(
        radar=radar,
        radar_sr=int(config["dataset"]["radar_sr"]),
        bp=bp,
        bp_sr=int(config["dataset"]["bp_sr"]),
        window_size=float(config["dataset"]["window_size"]),
        overlap=float(config["dataset"]["overlap"]),
        transform=Compose([ButterTransform(), ArctanDemodulation(), ToTensor()]),
        target_transform=Compose([ToTensor()]),
    )
    data_size = len(subject_dataset)

    end_train_index = int(data_size * float(config["setup"]["train_size"]))

    train_loader = build_dataloader(
        subject_dataset, start_index=0, end_index=end_train_index
    )
    test_loader = build_dataloader(
        subject_dataset, start_index=end_train_index, end_index=len(subject_dataset)
    )

    model = build_model(subject_dataset)

    agent = FLLocalAgent(model, train_loader, test_loader)
    fl.client.start_numpy_client(
        "{}:{}".format(config["server"]["hostname"], config["server"]["port"]),
        client=agent,
    )


@click.command()
@click.option("--is-global", is_flag=True)
@click.option("--subject-id")
def main(is_global, subject_id):
    if not is_global:
        run_local(subject_id)
    else:
        run_global_agent()


if __name__ == "__main__":
    main()
