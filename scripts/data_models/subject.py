#!/usr/bin/env python
# -*- coding: utf-8 -*-


class Subject:
    """Class designer to represent a subject

    Parameters:
        id (str): Unique subject identifier
        scenarios (dict): A dictionary instance with all scenarios of a subject
    """

    def __init__(self, code:str) -> None:
        self.code = code
        self.scenarios = {}
