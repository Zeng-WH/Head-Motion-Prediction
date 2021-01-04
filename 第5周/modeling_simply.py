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
            nn.ReLU(),
        )  # (Batch size, 1, 10)
        self.cnn0_phi = nn.Sequential(
            nn.Conv1d(1, 1, 10),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch size, 1, 10)
        self.cnn0_psi = nn.Sequential(
            nn.Conv1d(1, 1, 10),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch_size, 1, 10)
        self.dropout = nn.Dropout(0.2)
        self.GRU1 = nn.GRU(
            input_size=1,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )  # (Batch_size, 30, 64)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )  # (Batch_size, 30 , 1 ,35)

        self.max_p1 = nn.MaxPool2d((1, 35), stride=(1, 1))

        self.fnn = nn.Sequential(
            nn.Linear(30 * 1, 30 * 1),
            nn.ReLU(),
            nn.Linear(30 * 1, 30)
        )

    def forward(self, x):
        #out0 = self.norm_diff_test(x).clone().detach().requires_grad_(True).cuda()
        out0 = (x[:, :, 1: 20]).clone().detach().cuda()
        #out0 = torch.tensor(self.norm_diff(x), requires_grad=True).cuda()
        input1_1 = torch.zeros((len(out0), 1, 19)).cuda()
        input1_1[:, 0, :] = out0[:, 0, :]
        out1_1 = self.cnn0_theta(input1_1)
        input1_2 = torch.zeros((len(out0), 1, 19)).cuda()
        input1_2[:, 0, :] = out0[:, 1, :]
        out1_2 = self.cnn0_phi(input1_2)

        input1_3 = torch.zeros((len(out0), 1, 19)).cuda()
        input1_3[:, 0, :] = out0[:, 2, :]
        out1_3 = self.cnn0_psi(input1_3)

        out1 = torch.cat((out1_1, out1_2, out1_3), 1).cuda()
        out1 = out1.view(len(out1), 30).cuda()
       # out1 = self.dropout(out1)

        input2 = torch.zeros((len(out1), 30, 1)).cuda()
        input2[:, :, 0] = out1
        r_out1, h1 = self.GRU1(input2, None)
        #print('r_out1:')
        #print(r_out1)
        input3 = torch.zeros((len(r_out1), 1, 30, 64)).cuda()
        input3[:, 0, :, :] = r_out1
        '''
        print('************self.GRU1:*********')
        for name, param in self.GRU1.named_parameters():
            print(name, param)
        '''
        out2 = self.cnn1(input3)

        out8 = self.max_p1(out2)

        # self.fnn:
        input14 = torch.zeros((len(out8), 30)).cuda()
        input14 = out8[:, :, 0, 0]
        out = self.fnn(input14)
        # print(self.fnn.parameters())

        # out_final = torch.tensor(self.inv_norm_def(x, out), dtype=torch.float32, requires_grad=True).cuda()
        #out_final = self.inv_norm_diff_test(x, out).clone().requires_grad_().cuda()
        # print('*********out_final********')
        # print(out_final)
        # print(out_final.dtype)

        return out

    def norm_diff_test(self, x):
        input_temp1 = (x[:, :, 0: 19]).clone().detach().cuda()
        #input_temp1 = torch.tensor(x[:, :, 0: 19]).cuda()
        input_temp2 = (x[:, :, 1: 20]).clone().detach().cuda()
        #input_temp2 = torch.tensor(x[:, :, 1: 20]).cuda()
        input_temp2 = input_temp2 - input_temp1
        x_max = torch.max(torch.abs(input_temp2), 2)[0].cuda() + 1e-8
        x_max = torch.repeat_interleave(x_max.unsqueeze(dim=2), repeats=19, dim=2).cuda()
        input_temp2 = input_temp2 / x_max

        return input_temp2

    def inv_norm_diff_test(self, x, out):
        out_inv = out.view(len(out), 3, 10).clone().detach().cuda()
        x_max = torch.max(torch.abs(out_inv), 2)[0].cuda()
        temp1 = torch.tensor(x_max * out_inv[:, :, 0], dtype=torch.float32).cuda()
        x = torch.tensor(x, dtype=torch.float32).cuda()
        out_inv[:, :, 0] = x[:, :, -1] + temp1
        for i in range(9):
            out_inv[:, :, i + 1] = out_inv[:, :, i] + x_max * out_inv[:, :, i + 1]
        out_fin = out_inv.view(len(out), 30).clone().detach().cuda()

        return out_fin