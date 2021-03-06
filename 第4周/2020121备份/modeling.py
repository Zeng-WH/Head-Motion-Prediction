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
            num_layers=30,
            batch_first=True,
        )  # (Batch_size, 30, 64)
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )  # (Batch_size, 30 , 1 ,35)

        self.GRU2 = nn.GRU(
            input_size=35,
            hidden_size=64,
            num_layers=30,
            batch_first=True,
        )  # (Batch_size, 30, 64)

        self.cnn2 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
            nn.ReLU(),
        )  # (Batch_size, 30, 1, 35)

        self.GRU3 = nn.GRU(
            input_size=35,
            hidden_size=64,
            num_layers=30,
            batch_first=True,
        )  # (Batch_size, 30, 64)

        self.cnn3 = nn.Sequential(
            nn.Conv2d(1, 30, 30),
            nn.BatchNorm2d(30),
            nn.ReLU(),
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
            nn.ReLU(),
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
            nn.ReLU(),
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
            nn.ReLU(),
        )

        self.max_p1 = nn.MaxPool2d((1, 35), stride=(1, 1))

        self.fnn = nn.Sequential(
            nn.Linear(30 * 1, 30 * 1),
            nn.ReLU(),
            nn.Linear(30 * 1, 30)
        )

    def forward(self, x):
        out0 = self.norm_diff_test(x).clone().detach().requires_grad_(True).cuda()
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
        out1 = self.dropout(out1)

        input2 = torch.zeros((len(out1), 30, 1)).cuda()
        input2[:, :, 0] = out1
        r_out1, h1 = self.GRU1(input2, None)

        input3 = torch.zeros((len(r_out1), 1, 30, 64)).cuda()
        input3[:, 0, :, :] = r_out1

        out2 = self.cnn1(input3)

        input4 = torch.zeros((len(out2), 30, 35)).cuda()
        input4 = out2[:, 0, :, :]
        #for name, param in self.cnn1.named_parameters():
         #   print(name, param)

        r_out2, h_2 = self.GRU2(input4, None)

        # self.cnn2:
        input5 = torch.zeros((len(r_out2), 1, 30, 64)).cuda()
        input5[:, 0, :, :] = r_out2
        out3 = self.cnn2(input5)

        # self.GRU3:
        input6 = torch.zeros((len(out3), 30, 35)).cuda()
        input6 = out3[:, 0, :, :]
        r_out3, h_3 = self.GRU3(input6, None)

        # self.cnn3:
        input7 = torch.zeros((len(r_out3), 1, 30, 64)).cuda()
        input7[:, 0, :, :] = r_out3
        out4 = self.cnn3(input7)

        # self.GRU4:
        input8 = torch.zeros((len(out4), 30, 35)).cuda()
        input8 = out4[:, 0, :, :]
        r_out4, h_4 = self.GRU4(input8, None)

        # self.cnn4:
        input9 = torch.zeros((len(r_out4), 1, 30, 64)).cuda()
        input9[:, 0, :, :] = r_out4
        out5 = self.cnn4(input9)

        # self.GRU5:
        input10 = torch.zeros((len(out5), 30, 35)).cuda()
        input10 = out5[:, 0, :, :]
        r_out5, h_5 = self.GRU5(input10, None)

        # self.cnn5:
        input11 = torch.zeros((len(r_out5), 1, 30, 64)).cuda()
        input11[:, 0, :, :] = r_out5
        out6 = self.cnn5(input11)

        # self.GRU6:
        input12 = torch.zeros((len(out6), 30, 35)).cuda()
        input12 = out6[:, 0, :, :]
        r_out6, h_6 = self.GRU6(input12, None)

        # self.cnn6:
        input13 = torch.zeros((len(r_out6), 1, 30, 64)).cuda()
        input13[:, 0, :, :] = r_out6
        out7 = self.cnn6(input13)
        #print(self.cnn6.parameters())
        #for name, param in self.cnn6.named_parameters():
         #   print(name, param)

        # self.max_p1:
        out8 = self.max_p1(out7)

        # self.fnn:
        input14 = torch.zeros((len(out8), 30)).cuda()
        input14 = out8[:, :, 0, 0]
        out = self.fnn(input14)
       # print(self.fnn.parameters())

        #out_final = torch.tensor(self.inv_norm_def(x, out), dtype=torch.float32, requires_grad=True).cuda()
        out_final = self.inv_norm_diff_test(x, out).clone().requires_grad_().cuda()
        #print(out_final.dtype)

        return out_final
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

    def norm_diff(self, x):
        input0 = torch.zeros((len(x), 3, 20 - 1), requires_grad=True).cuda()
        for t in range(len(x)):
            # print(t)
            for i in range(20 - 1):
                input0[t:, :, i] = x[t, :, i + 1] - x[t, :, i]
           # c = input0[t, :, :]
            x_max = torch.max(torch.abs(input0[t, :, :]), 1)[0].cuda()
            input0[t, 0, :] = input0[t, 0, :] / (x_max[0] + 1e-8)
            input0[t, 1, :] = input0[t, 1, :] / (x_max[1] + 1e-8)
            input0[t, 2, :] = input0[t, 2, :] / (x_max[2] + 1e-8)
        return input0
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
    def inv_norm_def(self, x, out):
        out_inv = out.view(len(out), 3, 10).clone().detach().requires_grad_(True).cuda()
        #out_inv = torch.tensor(out.view(len(out), 3, 10).cuda(), dtype=torch.float32, requires_grad=True).cuda()
        for t in range(len(out)):
            x_max = torch.max(torch.abs(out_inv[t, :, :]), 1)[0].cuda()
            out_inv[t, 0, 0] = x[t, 0, -1] + x_max[0] * out_inv[t, 0, 0]
            out_inv[t, 1, 0] = x[t, 1, -1] + x_max[1] * out_inv[t, 1, 0]
            out_inv[t, 2, 0] = x[t, 2, -1] + x_max[2] * out_inv[t, 2, 0]
            for i in range(9):
                out_inv[t, :, i + 1] = out_inv[t, :, i] + x_max * out_inv[t, :, i + 1]
                #out_inv[t, 0, i + 1] = out_inv[t, 0, i] + x_max[0] * out_inv[t, 0, i + 1]
                #out_inv[t, 1, i + 1] = out_inv[t, 1, i] + x_max[1] * out_inv[t, 1, i + 1]
                #out_inv[t, 2, i + 1] = out_inv[t, 2, i] + x_max[2] * out_inv[t, 2, i + 1]
        #out_fin = torch.zeros((len(out), 30), dtype=torch.float32).cuda()
        #out_fin = torch.tensor(out_inv.view(len(out), 30), dtype=torch.float32, requires_grad=True).cuda()
        out_fin = out_inv.view(len(out), 30).clone().detach().requires_grad_(True).cuda()
        return out_fin





        

