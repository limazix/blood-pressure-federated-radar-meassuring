#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum


class ScenarioType(Enum):
    """Class used to list all types of scenarios"""

    RESTING = "Resting"
    VALSALVA = "Valsalva"
    APNEA = "Apnea"
    TILTUP = "TiltUp"
    TILTDOWN = "TiltDown"
