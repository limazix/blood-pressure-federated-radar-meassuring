#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from utils.validator import validate_file_path
from .scenario_type import ScenarioType


class Scenario:
    """Class used to store a scenario data

    Parameters:
        scenario_type (ScenarioType): It defines the type of the current scenario
        data (dict): It contains the scenario data
    """

    def __init__(self) -> None:
        self.scenario_type = None
        self.data = None

    def set_scenario_type(self, filename):
        """Method used to define the scenario type from a given file name

        Parameters:
            filename (str): The scenario file name
        """
        name, _ = os.path.splitext(filename)
        scenario = name.split("_")[2]
        self.scenario_type = ScenarioType(scenario)

    def setup(self, data_file: str):
        """Method used to load a scenario data from a file

        Parameters:
            data_file (str): The absolute path to a given scenario
        """
        validate_file_path(data_file)
        self.set_scenario_type(os.path.basename(data_file))
