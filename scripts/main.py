#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import click

from utils.validator import validate_directory_path


@click.command()
@click.option('--data-dir', help="path to the data directory")
def run(data_dir):
    """Method used to run the application from cli
    
    Parameters:
        data_dir (str): Data directory path
    """
    data_dir = os.path.abspath(data_dir)
    validate_directory_path(data_dir)
    print(data_dir)


if __name__ == "__main__":
    run()
