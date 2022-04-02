#!/usr/bin/env python
# -*- coding: utf-8 -*-

from email.policy import default
import os
import click

from data_models.subject import Subject
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


@click.command()
@click.option("--data-dir", help="path to the data directory")
@click.option("--train-size", default=80, help="data train size (ex: 80)")
def run(data_dir):
    """Method used to run the application from cli

    Parameters:
        data_dir (str): Data directory path
    """
    logger.info("[Start]")

    data_dir = os.path.abspath(data_dir)
    validate_directory_path(data_dir)

    subjects = setup_subjects(data_dir)


if __name__ == "__main__":
    run()
