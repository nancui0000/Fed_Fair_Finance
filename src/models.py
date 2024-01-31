#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F


class SimpleMLP(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(dim_in, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, dim_out)  # Output layer with a single unit for binary classification
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.5)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = F.relu(self.batch_norm1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return torch.sigmoid(x)  # Use sigmoid activation for binary classification

class LogisticRegression(nn.Module):
    def __init__(self, dim_in):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(dim_in, 1)

    def forward(self, x):
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
