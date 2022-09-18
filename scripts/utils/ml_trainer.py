#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import tqdm


class MLTrainer:
    def __init__(self, model, optimizer, criterion, lr):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr = lr

    def configure_optimizer(self):
        return self.optimizer(params=list(self.model.parameters()), lr=self.lr)

    def train(self, epochs, train_loader):
        """Train the model on the training set."""
        print("Start training")
        optimizer = self.configure_optimizer()
        for _ in tqdm(range(epochs)):
            for X, y in tqdm(train_loader):
                optimizer.zero_grad()
                self.criterion(self.model(X), y).backward()
                optimizer.step()

    def test(self, test_loader):
        """Validate the model on the test set."""
        print("Start test")
        loss = 0.0
        actual: torch.Tensor = None
        predicted: torch.Tensor = None
        with torch.no_grad():
            for X, y in tqdm(test_loader):
                outputs = self.model(X)
                loss += self.criterion(outputs, y).item()
                predicted = torch.concat([predicted, outputs])
                actual = torch.concat([actual, y])
        return loss / len(test_loader.dataset), self.mean_square_error(
            actual, predicted
        )

    def mean_square_error(self, actual, predicted):
        difference = np.subtract(actual, predicted)
        squared_difference = np.sqrt(difference)
        return squared_difference.mean()
