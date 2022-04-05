#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import click

from torch.utils.data import DataLoader

from data_models.subject import Subject
from scripts.data_models.subject_dataset import SubjectDataset


def setup_subjects(data_dir):
    """Method used to create all subject instancies based on the input data

    Parameters:
        data_dir (str): Data directory absolute path

    Returns:
        (list): A list instancy with all available subjects
    """
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


@click.group()
@click.option("--data-dir")
@click.pass_context
def cli(ctx, data_dir):
    """Method used to group all commands for a single subject run"""
    ctx.ensure_object(dict)
    ctx.obj["data-dir"] = data_dir


@cli.command()
@click.option("--subject-id")
@click.pass_context
def single(ctx, subject_id):
    print(ctx, subject_id)


if __name__ == "__main__":
    cli()
