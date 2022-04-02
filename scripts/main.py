#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click
import pandas as pd

from torch.utils.data import DataLoader

from data_models.subject import Subject
from data_models.suject_dataset import SubjectDataset

from ml_models.rnn_model import RNNModel
from fl_agents.fl_local_agent import FLLocalAgent

from utils.validator import validate_directory_path
from utils.logger import logger


def setup_subjects(data_dir):
    """Method used to create all subject instancies based on the input data

    Parameters:
        data_dir (str): Data directory absolute path

    Returns:
        (list): A list instancy with all available subjects
    """
    logger.info("[Steup Subjects]")
    subjects = []
    for subject_code in os.listdir(data_dir):
        subject_root_path = os.path.join(data_dir, subject_code)
        if os.path.isdir(subject_root_path):
            subject = Subject(code=subject_code)
            subject.setup(subject_root_path)
            subjects.append(subject)
    return subjects


def split_train_test(data: pd.DataFrame, train_size, batch_size):
    """Method used to create the train and test data loaders

    Parameters:
        data (DataFrame): DataFrame instance with alldata
        train_size (float): Value between 0-100 to define the tain data size

    Returns:
        DataLoader: Train Dataloader instance
        DataLoader: Test Dataloader instance
    """
    data_size = len(data)
    train_max_pos = int((data_size * train_size) / 100)
    data_train, data_test = data.iloc[:train_max_pos], data.iloc[train_max_pos:]
    train_dataset, test_dataset = SubjectDataset(data_train), SubjectDataset(data_test)
    return DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=False
    ), DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def setup_local_agents(subjects, train_size, batch_size):
    """Method used to load all local agents

    Parameters:
        subjects (list): A list instance with all subjects data

    Returns:
        list: A list instance with all local agents set
    """
    logger.info("[Start] Setup Local Agents")
    model = RNNModel(input_size=2000, hidden_size=1000, output_size=200)
    agents = []
    for subject in subjects:
        train_dataloader, test_dataloader = split_train_test(
            subject.get_all_data(), train_size, batch_size
        )
        agent = FLLocalAgent(model)
        agents.append(agent)
    return agents


@click.command()
@click.option("--data-dir", help="path to the data directory")
@click.option("--train-size", default=80, help="data train size (ex: 80)")
@click.option("--batch-size", default=64)
def run(data_dir, train_size, batch_size):
    """Method used to run the application from cli

    Parameters:
        data_dir (str): Data directory path
        train_size (float): Percentage of the data used for training
        batch_size (int): Size of pytorch batch
    """
    logger.info("[Start]")

    data_dir = os.path.abspath(data_dir)
    validate_directory_path(data_dir)

    subjects = setup_subjects(data_dir)
    local_agents = setup_local_agents(subjects, train_size, batch_size)


if __name__ == "__main__":
    run()
