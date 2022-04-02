#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

from data_models.subject import Subject
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


def setup_local_agents(subjects):
    """Method used to load all local agents

    Parameters:
        subjects (list): A list instance with all subjects data

    Returns:
        list: A list instance with all local agents set
    """
    logger.info("[Start] Setup Local Agents")
    model = RNNModel(input_size=2000, hidden_size=2, output_size=200)
    agents = []
    for subject in subjects:
        agent = FLLocalAgent(model)
        agents.append(agent)
    return agents


@click.command()
@click.option("--data-dir", help="path to the data directory")
@click.option("--train-size", default=80, help="data train size (ex: 80)")
def run(data_dir, train_size):
    """Method used to run the application from cli

    Parameters:
        data_dir (str): Data directory path
    """
    logger.info("[Start]")

    data_dir = os.path.abspath(data_dir)
    validate_directory_path(data_dir)

    subjects = setup_subjects(data_dir)
    local_agents = setup_local_agents(subjects)


if __name__ == "__main__":
    run()
