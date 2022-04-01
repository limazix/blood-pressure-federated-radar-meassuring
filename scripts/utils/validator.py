#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


def validate_directory_path(directory_path: str):
    """Method used to validate a given directory path

    Parameters:
        directory_path (str): Absolute directory path
    """
    if not os.path.isdir(directory_path):
        raise Exception("The given path should be a valid directory")


def validate_file_path(file_path: str):
    """Method used to valdate a file path

    Parameters:
        file_path (str): Absolute path to a file
    """
    if not os.path.isfile(file_path):
        raise Exception("The given path should be a file")
