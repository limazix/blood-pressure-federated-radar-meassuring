#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import nn


class SelectItemModel(nn.Module):
    def __init__(self, item_index) -> None:
        super(SelectItemModel, self).__init__()
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]
