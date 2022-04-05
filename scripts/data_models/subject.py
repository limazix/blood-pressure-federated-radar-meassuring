#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from numpy import concatenate

import numpy as np

from utils.validator import validate_directory_path
from .scenario import Scenario


class Subject:
    """Class designer to represent a subject

    Parameters:
        code (str): Unique subject identifier
        scenarios (list): A list instance with all scenarios of a subject
    """

    def __init__(self, code: str) -> None:
        self.code = code
        self.scenarios = []

    def setup(self, data_dir: str):
        """Method used to load all scenario data from files

        Parameters:
            data_dir (str): Absolute path the subject directory with all his files
        """
        validate_directory_path(data_dir)

        for scenario_filename in os.listdir(data_dir):
            scenario_filepath = os.path.join(data_dir, scenario_filename)
            scenario = Scenario()
            scenario.setup(data_file=scenario_filepath)
            self.scenarios.append(scenario)

    def get_all_data(self):
        """Method used load join all scenarios data

        Returns:
            Numpy.array: Radar data in a bi-dimensional array
            Numpy.array: Blood Pressure data in a single array
        """
        return np.concatenate(
            [scenario.radar for scenario in self.scenarios], axis=1
        ).T, np.concatenate([scenario.bp for scenario in self.scenarios])
