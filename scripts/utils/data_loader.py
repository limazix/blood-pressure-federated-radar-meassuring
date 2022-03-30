#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np
import scipy.io as sio


class DataLoader:
    """Class designed to load all data from a given path

    Parameters:
        data_root_path (str): The absolute path to the data root directory
    """

    def __init__(self, data_root_path: str) -> None:

        if not os.path.isdir(data_root_path):
            raise Exception("The data root path should be a valid directory")

        self.data_root_path = data_root_path

    def load_file(self, filepath):
        """Method used to load a .mat file

        Parameters:
            filename (str): Absolute path to the .mat file

        Returns:
            dict: All file contents as a dictionary
        """
        return sio.loadmat(filepath)

    def clean_data_columns(self, data: dict):
        """Method used to remove .mat control columns from the dictionary

        Parameters:
            data (dict): Dictionary created from a .mat file

        Returns:
            dict: A dictionary instancy without .mat control columns
        """
        return {k: np.array(v).flatten() for k, v in data.items() if k[0] != "_"}
