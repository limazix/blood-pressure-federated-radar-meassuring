#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import numpy as np

from utils.configurator import config
from utils.validator import validate_directory_path

from .subject import Subject


class DataBuilder:
    def get_subject_data(self, subject_id):
        subject_data_dir = os.path.abspath(
            os.path.normpath(os.path.join(config["setup"]["datadir"], subject_id))
        )

        if os.path.isfile(subject_data_dir):
            return None

        subject = Subject(code=subject_id)
        subject.setup(data_dir=subject_data_dir)
        return subject.get_all_data()

    def get_data(self, subject_id=None):
        radar = None
        bp = None
        if subject_id is not None:
            radar, bp = self.get_subject_data(subject_id)
        else:
            data_dir = os.path.abspath(os.path.normpath(config["setup"]["datadir"]))
            validate_directory_path(data_dir)
            for subject_dir in os.listdir(data_dir):
                data = self.get_subject_data(subject_dir)
                if data is not None:
                    radar = (
                        np.concatenate([radar, data[0]], axis=1)
                        if radar is not None
                        else data[0]
                    )
                    bp = np.concatenate([bp, data[1]]) if bp is not None else data[1]

        return radar.T, bp
