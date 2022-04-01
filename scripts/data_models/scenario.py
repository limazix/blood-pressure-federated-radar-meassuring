#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    def setup(self, data_file:str):
        """Method used to load a scenario data from a file
        
        Parameters:
            data_file (str): The absolute path to a given scenario
        """
        validate_file_path(data_file)

