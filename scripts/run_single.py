#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

from data_models.subject import Subject
from data_models.subject_dataset import SubjectDataset

from utils.validator import validate_directory_path


@click.command()
@click.option("--data-dir")
@click.option("--subject-id")
@click.option("--port")
def main(data_dir, subject_id, port):
    """Method used to run a single flower agent"""
    subject_data_dir = os.path.normpath(os.path.join(data_dir, subject_id))
    validate_directory_path(subject_data_dir)

    subject = Subject(code=subject_id)
    subject.setup(data_dir=subject_data_dir)
    radar, bp = subject.get_all_data()
    
    subject_dataset = SubjectDataset(radar=radar, radar_sr=2000, bp=bp, bp_sr=200, window_size=3, overlap=1)
    print(len(subject_dataset))


if __name__ == "__main__":
    main()
