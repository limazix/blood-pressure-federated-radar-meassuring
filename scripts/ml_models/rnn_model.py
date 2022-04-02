#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import nn


class RNNModel(nn.Module):
    """Class used to represent a simple RNN implementation in Pytorch for Radar

    Parameters:
        input_size (int): The size of the input data array
        hidden_size (int): Number of hidden layers
        output_size (int): The size of the output data array
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.in2hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.in2output = nn.Linear(input_size + hidden_size, output_size)
        self.hidden = self.init_hidden()

    def forward(self, x):
        combined = torch.cat((x, self.hidden), 1)
        self.hidden = torch.sigmoid(self.in2hidden(combined))
        output = self.in2output(combined)
        return output

    def init_hidden(self):
        return nn.init.kaiming_uniform_(torch.empty(1, self.hidden_size))
