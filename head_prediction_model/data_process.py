import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import pandas as pd
import time
'''机器每隔0.01s测量一次数据，利用前160ms的数据预测从160ms开始第'''

def read_file(file, window_size, pred_distance):
    data_frame = pd.read_csv(file, header=None)
    data_values = data_frame.values
    data_length = len(data_values)
    actual_length = data_length - pred_distance - window_size
    x_data = np.zeros((actual_length, 3, window_size))
    y_data = np.zeros((actual_length, 3))
    #y_data = torch.tensor(y_data, dtype=torch.float32)
    for i in range(actual_length):
        x_temp = data_values[i:i+window_size, 0:3]
        x_temp = np.transpose(x_temp)
        y_temp = data_values[i+window_size+pred_distance, 0:3]
        y_temp = np.transpose(y_temp)
        x_data[i, :, :] = x_temp
        y_data[i, :] = y_temp
    y_data = torch.tensor(y_data, dtype=torch.float32)
    return x_data, y_data

class Head_Motion_Dataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def main( ):
    x_data, y_data = read_file('./indata.csv', 20, 16)
    seed_1 = np.arange(len(x_data))
    np.random.shuffle(seed_1)
    np.random.shuffle(seed_1)
    x_train_set = x_data[seed_1[0: math.floor(0.8 * len(x_data))]]
    y_train_set = y_data[seed_1[0: math.floor(0.8 * len(x_data))]]
    x_val_set = x_data[seed_1[math.floor(0.8 * len(x_data)): len(x_data)]]
    y_val_set = y_data[seed_1[math.floor(0.8 * len(x_data)): len(x_data)]]
    np.save('./x_train_set.npy', x_train_set)
    np.save('./y_train_set.npy', y_train_set)
    np.save('./x_val_set.npy', x_val_set)
    np.save('./y_val_set.npy', y_val_set)


if __name__ == '__main__':
    main( )

