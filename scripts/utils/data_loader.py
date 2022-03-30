#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os


class DataLoader:
    """Class designed to load all data from a given path

    Parameters:
        data_root_path (str): The absolute path to the data root directory
    """

    def __init__(self, data_root_path: str) -> None:

        if not os.path.isdir(data_root_path):
            raise Exception("The data root path should be a valid directory")

        self.data_root_path = data_root_path
