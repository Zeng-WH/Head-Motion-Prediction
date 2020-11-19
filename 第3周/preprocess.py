import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time

def read_all_file(path):
    #读取文件夹下所有文件
    files = os.listdir(path)
    #得到文件夹下的所有文件名称
    lines = []
    for file in files:
        #遍历文件夹
        if not os.path.isdir(file):
            f = open(path+"/"+file)
            print(file)
            iter_f = f.readlines()
            for line in iter_f:
                lines.append(line.strip('\n').split(','))
    Oirent_Values = np.zeros((len(lines) - 1, 3))
    for i in range(len(lines) - 1):
        try:
            Oirent_Values[i, 0] = int(lines[i][2]) / 10
            Oirent_Values[i, 1] = int(lines[i][3]) / 10
            Oirent_Values[i, 2] = int(lines[i][3]) / 10
        except IndexError:
            Oirent_Values[i, 0] = Oirent_Values[i-1, 0]
            Oirent_Values[i, 1] = Oirent_Values[i-1, 1]
            Oirent_Values[i, 2] = Oirent_Values[i-1, 1]
            print('IndexError at :'+str(i))
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
            x_train_set[t, 0, :] = np.transpose(self.Ova[t: t + self.WinSize, 0])
            x_train_set[t, 1, :] = np.transpose(self.Ova[t: t + self.WinSize, 1])
            x_train_set[t, 2, :] = np.transpose(self.Ova[t: t + self.WinSize, 2])
            y_train_set[t, 0, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 0])
            y_train_set[t, 1, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 1])
            y_train_set[t, 2, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 2])
            # 依次代表theta, phi, psi三个角度
        return x_train_set, y_train_set

    def norm_diff(self):
        x_origin, y_origin = self.data_split()
        x_train_set = np.zeros((self.size, 3, self.WinSize - 1))
        y_train_set = np.zeros((self.size, 3, int(self.PreSize / 5) - 1))
        for t in range(self.size):
            # print(t)
            for i in range(self.WinSize - 1):
                x_train_set[t, :, i] = x_origin[t, :, i + 1] - x_origin[t, :, i]
            for j in range(int(self.PreSize / 5) - 1):
                y_train_set[t, :, j] = y_origin[t, :, j + 1] - y_origin[t, :, j]
            x_max = np.max(np.abs(x_train_set[t, :, :]), axis=1)
            y_max = np.max(np.abs(y_train_set[t, :, :]), axis=1)
            x_train_set[t, 0, :] = x_train_set[t, 0, :] / (x_max[0]+1e-8)
            x_train_set[t, 1, :] = x_train_set[t, 1, :] / (x_max[1]+1e-8)
            x_train_set[t, 2, :] = x_train_set[t, 2, :] / (x_max[2]+1e-8)
            y_train_set[t, 0, :] = y_train_set[t, 0, :] / (y_max[0]+1e-8)
            y_train_set[t, 1, :] = y_train_set[t, 1, :] / (y_max[1]+1e-8)
            y_train_set[t, 2, :] = y_train_set[t, 2, :] / (y_max[2]++1e-8)
        y_train_set = np.zeros((self.size, 3 * (int(self.PreSize / 5) - 1)), dtype=np.float32)
        y_train_set = y_train_set.reshape(self.size, 3 * (int(self.PreSize / 5) - 1))
        return x_train_set, y_train_set


class Head_Motion_Dataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
