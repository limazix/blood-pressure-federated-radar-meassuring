#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from utils.data_loader import DataLoader
from utils.validator import validate_file_path
from .scenario_type import ScenarioType


class Scenario:
    """Class used to store a scenario data

    Parameters:
        scenario_type (ScenarioType): It defines the type of the current scenario
        radar (DataFrame): It contains the scenario radar data
        bp (DataFrame): It contains the scenario bp data
        data_file (str): The absolute path to a given scenario
    """

    def __init__(self, data_file) -> None:
        self.scenario_type = None
        self.radar = None
        self.bp = None
        self.data_file = data_file

    def set_scenario_type(self, filename):
        """Method used to define the scenario type from a given file name

        Parameters:
            filename (str): The scenario file name
        """
        name, _ = os.path.splitext(filename)
        scenario = name.split("_")[2]
        self.scenario_type = ScenarioType(scenario)

    def load_data(self):
        loader = DataLoader()
        data = loader.load_file(self.data_file)
        data = loader.clean_data_columns(data)

        self.radar = np.array([data["radar_i"], data["radar_q"]])
        self.bp = np.array(data["tfm_bp"])


    def setup(self):
        """Method used to load a scenario data from a file
        """
        validate_file_path(self.data_file)
        self.set_scenario_type(os.path.basename(self.data_file))
