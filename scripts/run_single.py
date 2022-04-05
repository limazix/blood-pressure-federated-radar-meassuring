#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

from torch.utils.data import DataLoader

from data_models.subject import Subject
from data_models.subject_dataset import SubjectDataset

from utils.validator import validate_directory_path


def build_dataloader(dataset, start_index, end_index, batch_size):
    return DataLoader(
        dataset=dataset[start_index:end_index], batch_size=batch_size, shuffle=False
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
        radar=radar, radar_sr=2000, bp=bp, bp_sr=200, window_size=3, overlap=1
    )
    data_size = len(subject_dataset)

    train_size, test_size, val_size = data_split.split(":")
    end_train_index = int((data_size * int(train_size)) / 100)
    end_test_index = int((data_size * (int(train_size) + int(test_size))) / 100)

    train_loader = build_dataloader(
        subject_dataset, start_index=0, end_index=end_train_index, batch_size=32
    )
    test_loader = build_dataloader(
        subject_dataset,
        start_index=end_train_index,
        end_index=end_test_index,
        batch_size=32,
    )
    val_loader = build_dataloader(
        subject_dataset,
        start_index=end_test_index,
        end_index=len(subject_dataset),
        batch_size=32,
    )


if __name__ == "__main__":
    main()
