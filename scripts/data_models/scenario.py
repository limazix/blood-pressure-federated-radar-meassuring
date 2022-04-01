#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .scenario_type import ScenarioType


class Scenario:
    """Class used to store a scenario data

    Parameters:
        scenario_type (ScenarioType): It defines the type of the current scenario
        data (dict): It contains the scenario data
    """

    def __init__(self) -> None:
        self.scenario_type = None
