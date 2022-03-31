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
        self.validate_directory_path(data_root_path)
        self.data_root_path = data_root_path

    def validate_directory_path(self, directory_path: str):
        """Method used to validate a given directory path

        Parameters:
            directory_path (str): Absolute directory path
        """
        if not os.path.isdir(directory_path):
            raise Exception("The data root path should be a valid directory")

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

    def load_subject_data(self, subject_root_path):
        """Method used to aggregate all subject data .mat files to a single dictionary

        Parameters:
            subject_root_path (str): Absolute path to a subject data directory

        Returns:
            dict: A directory instancy with all subject data extracted from all of his .mat files
        """
        self.validate_directory_path(subject_root_path)

        data = {}
        for scenario_filename in os.listdir(subject_root_path):
            scenario_filepath = os.path.join(subject_root_path, scenario_filename)
            if scenario_filepath.endswith(".mat"):
                scenario_data = self.load_file(filepath=scenario_filepath)
                scenario_data = self.clean_data_columns(scenario_data)
                data = {k: np.concatenate([data[k], v]) for k, v in scenario_data}
        return data

    def run(self):
        """Method used to convert all subjects data to a dictionary

        Returns:
            dict: A dictionary instance where each key is a subject reference
                and the respective value is another dictionary with the subject's data
        """
        data = {}
        for subject in os.listdir(self.data_root_path):
            subject_root_path = os.path.join(self.data_root_path, subject)
            data[subject] = self.load_subject_data(subject_root_path)
        return data
