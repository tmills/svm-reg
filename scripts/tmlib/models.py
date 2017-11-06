#!/usr/bin/env python

## pytorch imports
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch

import numpy as np

from sklearn.metrics import accuracy_score

class SimpleModel(nn.Module):
    def train(self):
        nn.Module.train(self)
        self.optimizer.zero_grad()

    def forward(self, batch):
        x = self.fc1(batch)
        return x

    def criterion(self, system, actual):
        return self.loss(system, actual)

    def update(self):
        self.optimizer.step()

class SvmlikeModel(SimpleModel):
    def __init__(self, input_dims, lr=0.1, c=0.1, decay=0.1):
        super(SvmlikeModel, self).__init__()
        self.fc1 = nn.Linear(input_dims, 1)
        self.loss = HingeLoss(c=c)
        self.optimizer = optim.SGD(self.parameters(), lr=lr, weight_decay=decay)

    def forward(self, batch):
        x = self.fc1(batch)
        return x

class ExtendedModel(SimpleModel):
    def __init__(self, input_dims, hidden_dims, lr=0.1, c=0.1, init=None):
        super(ExtendedModel, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.output = nn.Linear(hidden_dims, 1)
        if not init is None:
            self.fc1.weight[0,:].data.zero_()
            self.fc1.weight[0,:].data.add_(init.data.cpu())
            self.output.weight.data.zero_()
            self.output.weight.data[0,0] = 1.

        self.relu = nn.ReLU()
        self.loss = HingeLoss(c=c)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, batch):
        x = self.fc1(batch)
        x = self.relu(x)
        x = self.output(x)
        return x

class HingeLoss(nn.Module):
    def __init__(self, c=1.0):
        super(HingeLoss, self).__init__()
        self.C = c

    def forward(self, input, target):
        _assert_no_grad(target)
        return torch.mean(torch.clamp(self.C - target * input, 0))

class L2VectorLoss(nn.Module):
    def __init__(self):
        super(L2VectorLoss, self).__init__()
        self.epsilon = 1e-3

    def forward(self, input, target):
        _assert_no_grad(target)
        #return torch.sqrt( torch.sum( ))
        return torch.dist(input, target) + self.epsilon

def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as volatile or not requiring gradients"
