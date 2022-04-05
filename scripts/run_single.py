#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

from utils.validator import validate_directory_path


@click.command()
@click.option("--data-dir")
@click.option("--subject-id")
@click.option("--port")
def main(data_dir, subject_id, port):
    subject_data_dir = os.path.normpath(os.path.join(data_dir, subject_id))
    validate_directory_path(subject_data_dir)


if __name__ == "__main__":
    main()
