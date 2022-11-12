#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

import numpy as np

from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data_models.subject_dataset import SubjectDataset

from utils.configurator import config
from utils.spline import Spline

from data_transforms.to_tensor import ToTensor
from data_transforms.butter_transform import ButterTransform
from data_transforms.fill_nan import FillNan


class DataLoaderBuilder:
    def build_dataloader(self, dataset, start_index, end_index):
        _dataset = copy.deepcopy(dataset)
        _dataset.prune(start_index, end_index)
        return DataLoader(
            dataset=_dataset,
            shuffle=False,
            num_workers=int(config["dataloader"]["num_workers"]),
            batch_size=int(config["dataloader"]["batch_size"]),
        )

    def build_global_splines(self, radar, bp):
        bp_spline = Spline(bp)
        bp_spline.build()

        radar_spline_0 = Spline(radar[0])
        radar_spline_0.build()

        radar_spline_1 = Spline(radar[1])
        radar_spline_1.build()

        return np.array([radar_spline_0, radar_spline_1]), np.array([bp_spline])

    def build_loaders(self, radar, bp):
        radar_spline, bp_spline = self.build_global_splines(radar, bp)

        dataset = SubjectDataset(
            radar=radar,
            radar_sr=int(config["dataset"]["radar_sr"]),
            bp=bp,
            bp_sr=int(config["dataset"]["bp_sr"]),
            window_size=float(config["dataset"]["window_size"]),
            overlap=float(config["dataset"]["overlap"]),
            transform=Compose(
                [
                    FillNan(radar_spline),
                    ButterTransform(),
                    ToTensor(),
                ]
            ),
            target_transform=Compose(
                [
                    FillNan(bp_spline),
                    ToTensor(),
                ]
            ),
        )
        data_size = len(dataset)

        end_train_index = int(data_size * float(config["setup"]["train_size"]))
        end_val_index = end_train_index + int(
            data_size * float(config["setup"]["val_size"])
        )

        train_loader = self.build_dataloader(
            dataset, start_index=0, end_index=end_train_index
        )
        val_loader = self.build_dataloader(
            dataset, start_index=end_train_index, end_index=end_val_index
        )
        test_loader = self.build_dataloader(
            dataset, start_index=end_val_index, end_index=data_size
        )
        return dataset, train_loader, val_loader, test_loader
