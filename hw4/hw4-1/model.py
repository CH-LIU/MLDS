import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

class Model(nn.Module):
    def __init__(self,input_shape, output_shape):
        super(Model, self).__init__()
        self.input_shape = input_shape
        self.flatten_shape = input_shape[0]*input_shape[1]*input_shape[2]
        self.fc1 = nn.Linear(self.flatten_shape, 256)
        self.fc2 = nn.Linear(256, output_shape)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(0)
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.xavier_normal(self.fc2.weight)

        '''self.main = nn.Sequential(
                nn.Linear(self.flatten_shape, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, output_shape),
                nn.Softmax(0)
        )'''
    def forward(self, state):
        state = state.view(-1)
        o1 = self.relu(self.fc1(state))
        action_prob = self.softmax(self.fc2(o1))

        return action_prob