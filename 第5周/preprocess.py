import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time


def read_all_file(path):
    # 读取文件夹下所有文件
    files = os.listdir(path)
    # 得到文件夹下的所有文件名称
    lines = []
    for file in files:
        # 遍历文件夹
        if not os.path.isdir(file):
            f = open(path + "/" + file)
            # print(file)
            iter_f = f.readlines()
            for line in iter_f:
                lines.append(line.strip('\n').split(','))
    Oirent_Values = np.zeros((len(lines) - 1, 3))
    for i in range(len(lines) - 1):
        try:
            Oirent_Values[i, 0] = int(lines[i][2]) / 10
            Oirent_Values[i, 1] = int(lines[i][3]) / 10
            Oirent_Values[i, 2] = int(lines[i][4]) / 10
        except IndexError:
            Oirent_Values[i, 0] = Oirent_Values[i - 1, 0]
            Oirent_Values[i, 1] = Oirent_Values[i - 1, 1]
            Oirent_Values[i, 2] = Oirent_Values[i - 1, 2]
            print('IndexError at :' + str(i))
    return Oirent_Values

def read_all_file_new(path):
    # 读取文件夹下所有文件
    files = os.listdir(path)
    # 得到文件夹下的所有文件名称
    lines = []
    for file in files:
        # 遍历文件夹
        if not os.path.isdir(file):
            f = open(path + "/" + file)
            # print(file)
            iter_f = f.readlines()
            for line in iter_f:
                lines.append(line.strip('\n').split(','))
    Oirent_Values = np.zeros((len(lines) - 1, 3))
    for i in range(len(lines)-1):
        try:
            Oirent_Values[i, 0] = int(lines[i][2]) / 10
            Oirent_Values[i, 1] = int(lines[i][3]) / 10
            Oirent_Values[i, 2] = int(lines[i][4]) / 10
        except IndexError:
            Oirent_Values[i, :] = Oirent_Values[i - 1, :]
            print('IndexError at :' + str(i))
    return Oirent_Values

def read_file(path):
    # 直接读取角度
    with open(path) as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(',') for line in lines]
    Oirent_Values = np.zeros((len(lines) - 1, 3))
    for i in range(len(lines) - 1):
        Oirent_Values[i, 0] = int(lines[i][2]) / 10
        Oirent_Values[i, 1] = int(lines[i][3]) / 10
        Oirent_Values[i, 2] = int(lines[i][3]) / 10
    return Oirent_Values


class Prepross_Data():
    # 预处理阶段就不采取归一化，把归一化的步骤放到网络里
    def __init__(self, Oirent_Values, Window_size, Predict_size):
        self.Ova = Oirent_Values
        self.WinSize = Window_size
        self.PreSize = Predict_size
        self.size = math.floor(len(self.Ova) - self.PreSize - self.WinSize + 1)

    def data_split(self):
        x_train_set_size = math.floor(len(self.Ova) - self.PreSize - self.WinSize + 1)
        x_train_set = np.zeros((x_train_set_size, 3, self.WinSize))
        # x_train_set = np.zeros((3, x_train_set_size, self.WinSize))
        y_train_set = np.zeros((x_train_set_size, 3, int(self.PreSize / 5)))
        y_step = range(5, self.PreSize + 1, 5)
        # y_train_set = np.zeros((x_train_set_size, self.PreSize, 3))
        for t in range(x_train_set_size):
            # print(t)
            x_train_set[t, :, :] = np.transpose(self.Ova[t: t + self.WinSize, :])
            y_train_set[t, :, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), :])
            # 依次代表theta, phi, psi三个角度
        y_train_set_out = torch.tensor(y_train_set.reshape(x_train_set_size, 3 * int(self.PreSize / 5)),
                                           dtype=torch.float32)
        return x_train_set, y_train_set_out

    # 2021/1/1

    def data_split_new(self):
        x_train_set_size = math.floor(len(self.Ova) - self.PreSize - self.WinSize + 1)
        x_train_set = np.zeros((x_train_set_size, 3, self.WinSize))
        # self.WinSize = 20
        # self.PreSize = 50
        y_train_set = np.zeros((x_train_set_size, 3, int(self.PreSize / 5)))
        y_step = range(5, self.PreSize + 1, 5)
        for t in range(x_train_set_size):
            x_train_set[t, :, :] = np.transpose(self.Ova[t: t + self.WinSize, :])
            y_train_set[t, :, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), :])
            #y_train_set_out = torch.tensor(y_train_set.reshape(x_train_set_size, 3 * int(self.PreSize / 5)),
             #                              dtype=torch.float32)
        return x_train_set, y_train_set

    def norm_diff_x_test_new(self, x_train_set):
        input_temp1 = x_train_set[:, :, 0: 19]
        input_temp2 = x_train_set[:, :, 1: 20]
        input_temp2 = input_temp2 - input_temp1
        input_temp_tensor = torch.from_numpy(input_temp2)
        x_max = torch.max(torch.abs(input_temp_tensor), 2)[0] + 1e-8
        x_max = torch.repeat_interleave(x_max.unsqueeze(dim=2), repeats=19, dim=2)
        input_temp_tensor = input_temp_tensor / x_max
        # input_temp2是经过数据预处理的x_train_set

        return input_temp_tensor

    def norm_diff_y_test_new(self, x_train_set, y_train_set_out):
        x_train_set_size = math.floor(len(self.Ova) - self.PreSize - self.WinSize + 1)
        input_temp = np.zeros((x_train_set_size, 3, int(self.PreSize / 5)+1))
        input_temp[:, :, 0] = x_train_set[:, :, self.WinSize-1]
        input_temp[:, :, 1: int(self.PreSize / 5)+1] = y_train_set_out
        input_temp1 = input_temp[:, :, 0: int(self.PreSize / 5)]
        input_temp2 = input_temp[:, :, 1: int(self.PreSize / 5) + 1]
        input_temp2 = input_temp2 - input_temp1
        input_temp_tensor = torch.from_numpy(input_temp2)
        x_max = torch.max(torch.abs(input_temp_tensor), 2)[0] + 1e-8
        x_max = torch.repeat_interleave(x_max.unsqueeze(dim=2), repeats=int(self.PreSize / 5), dim=2)
        input_temp_tensor = input_temp_tensor / x_max
        input_temp_tensor_out = torch.tensor(input_temp_tensor.reshape(x_train_set_size, 3 * int(self.PreSize / 5)),
                                             dtype=torch.float32)
        # input_temp2是经过标准化处理的y_train_set

        return input_temp_tensor_out

    def inv_norm_diff_test_new(self, x, out):
        #似乎不用更改
        out_inv = out.view(len(out), 3, 10).clone().detach().cuda()
        x_max = torch.max(torch.abs(out_inv), 2)[0].cuda()
        temp1 = torch.tensor(x_max * out_inv[:, :, 0], dtype=torch.float32).cuda()
        x = torch.tensor(x, dtype=torch.float32).cuda()
        out_inv[:, :, 0] = x[:, :, -1] + temp1
        for i in range(9):
            out_inv[:, :, i + 1] = out_inv[:, :, i] + x_max * out_inv[:, :, i + 1]
        out_fin = out_inv.view(len(out), 30).clone().detach().cuda()

        return out_fin

    def norm_diff(self):
        x_origin, y_origin = self.data_split()
        x_train_set = np.zeros((self.size, 3, self.WinSize - 1))
        y_train_set = np.zeros((self.size, 3, int(self.PreSize / 5) - 1), dtype=np.float32)
        for t in range(self.size):
            # print(t)
            for i in range(self.WinSize - 1):
                x_train_set[t, :, i] = x_origin[t, :, i + 1] - x_origin[t, :, i]
            for j in range(int(self.PreSize / 5) - 1):
                y_train_set[t, :, j] = y_origin[t, :, j + 1] - y_origin[t, :, j]
            x_max = np.max(np.abs(x_train_set[t, :, :]), axis=1)
            y_max = np.max(np.abs(y_train_set[t, :, :]), axis=1)
            x_train_set[t, 0, :] = x_train_set[t, 0, :] / (x_max[0] + 1e-8)
            x_train_set[t, 1, :] = x_train_set[t, 1, :] / (x_max[1] + 1e-8)
            x_train_set[t, 2, :] = x_train_set[t, 2, :] / (x_max[2] + 1e-8)
            y_train_set[t, 0, :] = y_train_set[t, 0, :] / (y_max[0] + 1e-8)
            y_train_set[t, 1, :] = y_train_set[t, 1, :] / (y_max[1] + 1e-8)
            y_train_set[t, 2, :] = y_train_set[t, 2, :] / (y_max[2] + +1e-8)
        y_train_set_return = np.zeros((self.size, 3 * (int(self.PreSize / 5) - 1)), dtype=np.float32)
        y_train_set_return = y_train_set.reshape(self.size, 3 * (int(self.PreSize / 5) - 1))
        return x_train_set, y_train_set_return


class Head_Motion_Dataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def inv_norm_diff_test_new(x, out):
    #似乎不用更改
    out_inv = out.view(len(out), 3, 10).clone().detach().cuda()
    x_max = torch.max(torch.abs(out_inv), 2)[0].cuda()
    temp1 = torch.tensor(x_max * out_inv[:, :, 0], dtype=torch.float32).cuda()
    x = torch.tensor(x, dtype=torch.float32).cuda()
    out_inv[:, :, 0] = x[:, :, -1] + temp1
    for i in range(9):
        out_inv[:, :, i + 1] = out_inv[:, :, i] + x_max * out_inv[:, :, i + 1]
    out_fin = out_inv.view(len(out), 30).clone().detach().cuda()

    return out_fin