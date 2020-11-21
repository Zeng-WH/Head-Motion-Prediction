import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time

class Head_Motion_Predict(nn.Module):

    def __init__(self):
        super(Head_Motion_Predict, self).__init__()

        self.cnn0_theta = nn.Sequential(
            nn.Conv1d(1, 1, 10),
            nn.BatchNorm1d(1),
        ) #(Batch size, 1, 10)
        self.cnn0_phi = nn.Sequential(
            nn.Conv1d(1, 1, 10),
            nn.BatchNorm1d(1),
        ) #(Batch size, 1, 10)
        self.cnn0_psi = nn.Sequential(
            nn.Conv1d(1, 1, 10),
            nn.BatchNorm1d(1),
        ) #(Batch_size, 1, 10)
        self.GRU1 = nn.GRU(
            input_size = 1,
            hidden_size = 64,
            num_layers = 30,
            batch_first = True,
        ) #(Batch_size, 30, 64)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
        ) #(Batch_size, 30 , 1 ,35)

        self.GRU2 = nn.GRU(
            input_size = 35,
            hidden_size = 64,
            num_layers = 30,
            batch_first = True,
        ) #(Batch_size, 30, 64)

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
        ) #(Batch_size, 30, 1, 35)

        self.GRU3 = nn.GRU(
            input_size=35,
            hidden_size=64,
            num_layers=30,
            batch_first=True,
        ) #(Batch_size, 30, 64)

        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
        )

        self.GRU4 = nn.GRU(
            input_size=35,
            hidden_size=64,
            num_layers=30,
            batch_first=True,
        )

        self.cnn4 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
        )

        self.GRU5 = nn.GRU(
            input_size=35,
            hidden_size=64,
            num_layers=30,
            batch_first=True,
        )

        self.cnn5 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
        )

        self.GRU6 = nn.GRU(
            input_size=35,
            hidden_size=64,
            num_layers=30,
            batch_first=True,
        )

        self.cnn6 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
        )

        self.max_p1 = nn.MaxPool2d((1, 35), stride=(1, 1))

        self.fnn = nn.Sequential(
            nn.Linear(30*1, 30*1),
            nn.ReLU(),
            nn.Linear(30*1, 27)
        )
    def forward(self, x):
        input1_1 = torch.zeros((len(x), 1, 19), dtype=torch.float32).cuda()
        input1_1[:, 0, :] = x[:, 0, :]
        out1_1 = self.cnn0_theta(input1_1)
        input1_2 = torch.zeros((len(x), 1, 19), dtype=torch.float32).cuda()
        input1_2[:, 0, :] = x[:, 1, :]
        out1_2 = self.cnn0_phi(input1_2)

        input1_3 = torch.zeros((len(x), 1, 19), dtype=torch.float32).cuda()
        input1_3[:, 0, :] = x[:, 2, :]
        out1_3 = self.cnn0_psi(input1_3)

        out1 = torch.cat((out1_1, out1_2, out1_3), 1)
        out1 = out1.view(len(out1), 30)

        input2 = torch.zeros((len(out1), 30, 1), dtype=torch.float32).cuda()
        input2[:, :, 0] = out1
        r_out1, h1 = self.GRU1(input2, None)

        input3 = torch.zeros((len(r_out1), 1 , 30, 64),dtype=torch.float32).cuda()
        input3[:, 0, :,:] = r_out1

        out2 = self.cnn1(input3)


        input4 = torch.zeros((len(out2), 30, 35), dtype=torch.float32).cuda()
        input4 = out2[:, 0, :, :]

        r_out2, h_2 = self.GRU2(input4, None)


        #self.cnn2:
        input5 = torch.zeros((len(r_out2), 1, 30, 64), dtype=torch.float32).cuda()
        input5[:, 0, :, :] = r_out2
        out3 = self.cnn2(input5)

        #self.GRU3:
        input6 = torch.zeros((len(out3), 30, 35), dtype=torch.float32).cuda()
        input6 = out3[:, 0, :, :]
        r_out3, h_3 = self.GRU3(input6, None)

        #self.cnn3:
        input7 = torch.zeros((len(r_out3), 1, 30, 64), dtype=torch.float32).cuda()
        input7[:, 0, :, :] = r_out3
        out4 = self.cnn3(input7)

        #self.GRU4:
        input8 = torch.zeros((len(out4), 30, 35), dtype=torch.float32).cuda()
        input8 = out4[:, 0, :, :]
        r_out4, h_4 = self.GRU4(input8, None)

        #self.cnn4:
        input9 = torch.zeros((len(r_out4), 1, 30, 64), dtype=torch.float32).cuda()
        input9[:, 0, :, :] = r_out4
        out5 = self.cnn4(input9)

        #self.GRU5:
        input10 = torch.zeros((len(out5), 30, 35), dtype=torch.float32).cuda()
        input10 = out5[:, 0, :, :]
        r_out5, h_5 = self.GRU5(input10, None)


        #self.cnn5:
        input11 = torch.zeros((len(r_out5), 1, 30, 64), dtype=torch.float32).cuda()
        input11[:, 0, :, :] = r_out5
        out6 = self.cnn5(input11)

        #self.GRU6:
        input12 = torch.zeros((len(out6), 30, 35), dtype=torch.float32).cuda()
        input12 = out6[:, 0, :, :]
        r_out6, h_6 = self.GRU6(input12, None)

        #self.cnn6:
        input13 = torch.zeros((len(r_out6), 1, 30, 64), dtype=torch.float32).cuda()
        input13[:, 0, :, :] = r_out6
        out7 = self.cnn6(input13)


        #self.max_p1:
        out8 = self.max_p1(out7)

        #self.fnn:
        input14 = torch.zeros((len(out8), 30), dtype=torch.float32).cuda()
        input14 = out8[:, :, 0, 0]
        out = self.fnn(input14)

        return out
