#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .fl_agent import FLAgent


class FLLocalAgent(FLAgent):
    """Class used to represent the architecture local (in-device) agent"""

    def fit(self, X, y):
        pass