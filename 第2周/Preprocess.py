import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
'''
这是之前写的读入文件的代码
def read_file(path):
    #读取四元数
    with open(path) as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(' ') for line in lines]
    Quat = np.zeros((len(lines)-1,4))
    for i in range(len(lines)-1):
        Quat[i, 0] = int(lines[i][7])
        Quat[i, 1] = int(lines[i][8])
        Quat[i, 2] = int(lines[i][9])
        Quat[i, 3] = int(lines[i][10])
    return Quat
'''
def read_file(path):
    #直接读取角度
    with open(path) as f:
        lines = f.readlines()
        lines = [line.strip('\n').split(',') for line in lines]
    Oirent_Values = np.zeros((len(lines)-1, 3))
    for i in range(len(lines)-1):
        Oirent_Values[i, 0] = int(lines[i][2])/10
        Oirent_Values[i, 1] = int(lines[i][3])/10
        Oirent_Values[i, 2] = int(lines[i][3])/10
    return Oirent_Values
'''
这是之前写的预处理数据的代码
class Preprocess_Data():
    def __init__(self, Quat):
        self.Q = Quat
        print(Quat)
        self.Eula = np.zeros((np.size(Quat), 3))
    def quat2eua(self):
        print(self.Q[0,1])
        print(self.Q[:,1]*self.Q[:,3])
        print(np.pi)
        self.Eula=np.c_[np.arctan2(-(2*(self.Q[:,1]*self.Q[:,3] - self.Q[:,0]*self.Q[:,2])),(np.power(self.Q[:,0],2) + np.power(self.Q[:,1],2) - np.power(self.Q[:,2],2) - np.power(self.Q[:,3],2))), np.arctan2((2*(self.Q[:,1]*self.Q[:,2] + self.Q[:,0]*self.Q[:,3])),np.sqrt(1-np.power(2*(self.Q[:,1]*self.Q[:,2] + self.Q[:,0]*self.Q[:,3]),2))),np.arctan2(-(2*(self.Q[:,2]*self.Q[:,3] - self.Q[:,0]*self.Q[:,1])),(np.power(self.Q[:,0],2) - np.power(self.Q[:,1],2) + np.power(self.Q[:,2],2)- np.power(self.Q[:,3],2)))]
        return self.Eula*180/np.pi
'''
class Prepross_Data():
    #预处理阶段就不采取归一化，把归一化的步骤放到网络里
    def __init__(self, Oirent_Values, Window_size, Predict_size):
        self.Ova = Oirent_Values
        self.WinSize = Window_size
        self.PreSize = Predict_size
    def data_split(self):
        x_train_set_size = math.floor(len(self.Ova)-self.PreSize-self.WinSize+1)
        '''
        x_train_set_theta = np.zeros((x_train_set_size, self.WinSize))
        x_train_set_phi = np.zeros((x_train_set_size, self.WinSize))
        x_train_set_psi = np.zeros((x_train_set_size, self.WinSize))
        '''
        x_train_set = np.zeros((3, x_train_set_size, self.WinSize))
        y_train_set = np.zeros((3, x_train_set_size, int(self.PreSize/5)))
        y_step = range(5, self.PreSize+1, 5)
        #y_train_set = np.zeros((x_train_set_size, self.PreSize, 3))
        for t in range(x_train_set_size):
            print(t)
            x_train_set[0, t, :] = np.transpose(self.Ova[t: t+self.WinSize, 0])
            x_train_set[1, t, :] = np.transpose(self.Ova[t: t + self.WinSize, 1])
            x_train_set[2, t, :] = np.transpose(self.Ova[t: t + self.WinSize, 2])
            y_train_set[0, t, :] = np.transpose(self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 0])
            y_train_set[1, t, :] = np.transpose(self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 1])
            y_train_set[2, t, :] = np.transpose(self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 2])
            # 依次代表theta, phi, psi三个角度
        return x_train_set,y_train_set
def main( ):
    O = read_file('D:\模式识别\头部姿态预测\第2周\data\SAVE2020_11_11_21-54-55.DAT')
    print(len(O))
    print(O[1,:])
    a = Prepross_Data(O, 20, 50)
    w, p=a.data_split()
    print("happy")
def test():
    a = np.transpose(np.array([1,2,3,4]))
   # a = np.array([1,2,3,4])
    print(a)
    print(np.size(a))
    t =Prepross_Data(a, 20, 50)
    t.data_split()

if __name__ == '__main__':
    main( )



