import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
def main( ):
    Qt = np.zeros((1,4))
    Qt [0, 0] = -0.043
    Qt [0, 1] = 0.603
    Qt [0, 2] = 0.314
    Qt [0, 3] = 0.59
    #Qt = np.array([-0.435, 0.603, 0.314, 0.59]).reshape(1,4)
    print(Qt)
    Pre = Preprocess_Data(Qt)
    eu = Pre.quat2eua()
    print('*******')
    print(eu)
def test():
    a = np.transpose(np.array([1,2,3,4]))
   # a = np.array([1,2,3,4])
    print(a)
    print(np.size(a))

if __name__ == '__main__':
    main( )



