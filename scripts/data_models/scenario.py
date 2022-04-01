#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd

from scipy import signal

from utils.logger import logger
from utils.data_loader import DataLoader
from utils.validator import validate_file_path
from .scenario_type import ScenarioType


class Scenario:
    """Class used to store a scenario data

    Parameters:
        scenario_type (ScenarioType): It defines the type of the current scenario
        data (DataFrame): It contains the scenario data
    """

    def __init__(self) -> None:
        logger.debug("[Scenario] New scenario")
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
        logger.debug("[Scenario] Set scenario type %s", self.scenario_type.value)

    def setup(self, data_file: str):
        """Method used to load a scenario data from a file

        Parameters:
            data_file (str): The absolute path to a given scenario
        """
        logger.debug("[Scenario] Setup")
        validate_file_path(data_file)
        self.set_scenario_type(os.path.basename(data_file))
        loader = DataLoader()
        data = loader.load_file(data_file)
        data = loader.clean_data_columns(data)

        self.data = pd.DataFrame(
            {
                "radar_q": signal.resample(data["radar_q"], data["tfm_bp"].shape[0]),
                "radar_i": signal.resample(data["radar_i"], data["tfm_bp"].shape[0]),
                "bp": data["tfm_bp"],
            }
        )
