import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time
'''对模型进行重新设计'''

class Head_Motion_Predict(nn.Module):
    def __init__(self):
        super(Head_Motion_Predict, self).__init__()

        self.cnn0_theta = nn.Sequential(
            nn.Conv1d(1, 1, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch size, 1, 18)
        self.cnn0_phi = nn.Sequential(
            nn.Conv1d(1, 1, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch size, 1, 18)
        self.cnn0_psi = nn.Sequential(
            nn.Conv1d(1, 1, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch_size, 1, 18)
        self.dropout = nn.Dropout(0.2)
        self.GRU1_1 = nn.GRU(
            input_size=1,
            hidden_size=18,
            num_layers=1,
            batch_first=True,
        )  # (Batch_size, 18, 18)
        self.GRU1_2 = nn.GRU(
            input_size=1,
            hidden_size=18,
            num_layers=1,
            batch_first=True,
        )  # (Batch_size, 18, 18)
        self.GRU1_3 = nn.GRU(
            input_size=1,
            hidden_size=18,
            num_layers=1,
            batch_first=True,
        )  # (Batch_size, 18, 18)
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(True),

            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(True),

            nn.MaxPool2d(2, 2, 0),

        )
        # (Batch_size, 512, 1, 1)
        self.fnn = nn.Sequential(
            nn.Linear(512 * 1, 30 * 1),
            nn.ReLU(),
            nn.Linear(30 * 1, 30)
        )

    def forward(self, x):
        input1_1 = torch.zeros((len(x), 1, 20)).cuda()
        input1_1[:, 0, :] = x[:, 0, :]
        out1_1 = self.cnn0_theta(input1_1)

        input1_2 = torch.zeros((len(x), 1, 20)).cuda()
        input1_2[:, 0, :] = x[:, 1, :]
        out1_2 = self.cnn0_phi(input1_2)

        input1_3 = torch.zeros((len(x), 1, 20)).cuda()
        input1_3[:, 0, :] = x[:, 2, :]
        out1_3 = self.cnn0_psi(input1_3)
        out1_1, out1_2, out1_3 = out1_1.cuda(), out1_2.cuda(), out1_3.cuda()

        out1_1 = out1_1.view(len(x), 18, 1).cuda()
        out1_2 = out1_2.view(len(x), 18, 1).cuda()
        out1_3 = out1_3.view(len(x), 18, 1).cuda()

        r_out1_1, h1_1 = self.GRU1_1(out1_1, None)
        r_out1_2, h1_2 = self.GRU1_2(out1_2, None)
        r_out1_3, h1_3 = self.GRU1_3(out1_3, None)
        r_out1_1, r_out1_2, r_out1_3 = r_out1_1.cuda(), r_out1_2.cuda(), r_out1_3.cuda()

        input2 = torch.zeros((len(r_out1_1), 3, 18, 18)).cuda()
        input2[:, 0, :, :] = r_out1_1
        input2[:, 1, :, :] = r_out1_2
        input2[:, 2, :, :] = r_out1_3

        out2 = self.features(input2)

        input3 = out2[:, :, 0, 0]

        out = self.fnn(input3)

        return out
