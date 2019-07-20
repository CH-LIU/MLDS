import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

class Generator(nn.Module):
    def __init__(self, input_size, tag_size):
        super(Generator, self).__init__()
        self.input_size = input_size
        #self.embedding = nn.Embedding(num_classes, embed_dim)
        self.hidden = nn.Linear(input_size + tag_size , 256)
        self.main = nn.Sequential(
                nn.ConvTranspose2d(256, 1024, 4, 1, 0, bias=False),
                nn.BatchNorm2d(1024),
                nn.ReLU(True),
                nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(True),
                nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
                nn.Tanh()
        )
        #self.fc = nn.Linear(embed_dim, project_dim)
        
    def forward(self, noise, tag):
        #message = self.embedding(tag).unsqueeze(2).unsqueeze(3)
        feature = torch.cat((noise, tag), dim = 1)
        hidden =  self.hidden(feature)
        hidden = hidden.unsqueeze(2).unsqueeze(3)
        output = self.main(hidden)

        return output

class Discriminator(nn.Module):
    def __init__(self, num_classes):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes

        self.main = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(512, 1024, 4, 1, 0, bias=False),
                #nn.Sigmoid()
                )
        self.fc1 = nn.Sequential(
                nn.Linear(1024, num_classes),
                nn.Sigmoid()
                )
        
        self.fc2 = nn.Sequential(
                nn.Linear(1024, 1),
                #nn.Sigmoid()            #wgan loss
                )
    def forward(self, img):
        tmp = self.main(img)
        tmp = tmp.view(tmp.shape[0], -1)
        output_d = self.fc2(tmp)
        output_tag = self.fc1(tmp)

        return output_d, output_tag
