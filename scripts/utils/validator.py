#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def validate_directory_path(directory_path: str):
    """Method used to validate a given directory path

    Parameters:
        directory_path (str): Absolute directory path
    """
    if not os.path.isdir(directory_path):
        raise Exception("The data root path should be a valid directory")
