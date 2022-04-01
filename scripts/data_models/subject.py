#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from utils.validator import validate_directory_path
from .scenario import Scenario


class Subject:
    """Class designer to represent a subject

    Parameters:
        id (str): Unique subject identifier
        scenarios (dict): A dictionary instance with all scenarios of a subject
    """

    def __init__(self, code: str) -> None:
        self.code = code
        self.scenarios = {}

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
            self.scenarios[scenario.scenario_type] = scenario
