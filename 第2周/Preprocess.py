import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time
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
        self.size = math.floor(len(self.Ova)-self.PreSize-self.WinSize+1)
    def data_split(self):
        x_train_set_size = math.floor(len(self.Ova)-self.PreSize-self.WinSize+1)
        '''
        x_train_set_theta = np.zeros((x_train_set_size, self.WinSize))
        x_train_set_phi = np.zeros((x_train_set_size, self.WinSize))
        x_train_set_psi = np.zeros((x_train_set_size, self.WinSize))
        '''
        x_train_set = np.zeros((x_train_set_size, 3, self.WinSize))
        #x_train_set = np.zeros((3, x_train_set_size, self.WinSize))
        y_train_set = np.zeros((x_train_set_size, 3, int(self.PreSize/5)))
        y_step = range(5, self.PreSize+1, 5)
        #y_train_set = np.zeros((x_train_set_size, self.PreSize, 3))
        for t in range(x_train_set_size):
            #print(t)
            x_train_set[t, 0, :] = np.transpose(self.Ova[t: t + self.WinSize, 0])
            x_train_set[t, 1, :] = np.transpose(self.Ova[t: t + self.WinSize, 1])
            x_train_set[t, 2, :] = np.transpose(self.Ova[t: t + self.WinSize, 2])
            y_train_set[t, 0, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 0])
            y_train_set[t, 1, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 1])
            y_train_set[t, 2, :] = np.transpose(
                self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 2])
            '''x_train_set[0, t, :] = np.transpose(self.Ova[t: t+self.WinSize, 0])
            x_train_set[1, t, :] = np.transpose(self.Ova[t: t + self.WinSize, 1])
            x_train_set[2, t, :] = np.transpose(self.Ova[t: t + self.WinSize, 2])
            y_train_set[0, t, :] = np.transpose(self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 0])
            y_train_set[1, t, :] = np.transpose(self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 1])
            y_train_set[2, t, :] = np.transpose(self.Ova[np.arange(t + self.WinSize + 4, t + self.WinSize + self.PreSize + 1, 5), 2])'''

            # 依次代表theta, phi, psi三个角度
        return x_train_set,y_train_set
    def norm_diff(self):
        x_origin, y_origin = self.data_split()
        x_train_set = np.zeros((self.size, 3, self.WinSize-1))
        y_train_set = np.zeros((self.size, 3, int(self.PreSize/5)-1))
        for t in range(self.size):
            #print(t)
            for i in range(self.WinSize-1):
                x_train_set[t, :, i] = x_origin[t, :, i+1] - x_origin[t, :, i]
            for j in range(int(self.PreSize/5)-1):
                y_train_set[t, :, j] = y_origin[t, :, j+1] - y_origin[t, :, j]
            x_max = np.max(np.abs(x_train_set[t, :, :]), axis=1)
            y_max = np.max(np.abs(y_train_set[t, :, :]), axis=1)
            x_train_set[t, 0, :] = x_train_set[t, 0, :]/x_max[0]
            x_train_set[t, 1, :] = x_train_set[t, 1, :] / x_max[1]
            x_train_set[t, 2, :] = x_train_set[t, 2, :] / x_max[2]
            y_train_set[t, 0, :] = y_train_set[t, 0, :] / y_max[0]
            y_train_set[t, 1, :] = y_train_set[t, 1, :] / y_max[1]
            y_train_set[t, 2, :] = y_train_set[t, 2, :] / y_max[2]
        y_train_set = np.zeros((self.size, 3*(int(self.PreSize/5)-1)), dtype=np.float32)
        y_train_set = y_train_set.reshape(self.size, 3*(int(self.PreSize/5)-1))
        return x_train_set, y_train_set
class Head_Motion_Dataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

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
            nn.Linear(30*1, 27),
            nn.ReLU(),
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

        out1 = torch.cat((out1_1, out1_2, out1_3), 1).cuda()
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


def main( ):
   # O = read_file('D:\模式识别\头部姿态预测\第2周\data\SAVE2020_11_11_21-54-55.DAT')
   # a = Prepross_Data(O, 20, 50)
   # x,y =a.norm_diff()
   # print("hap
   data_set = read_file('D:\模式识别\头部姿态预测\第2周\data\SAVE2020_11_11_21-54-55.DAT')
   a = Prepross_Data(data_set, 20, 50)
   train_x, train_y = a.norm_diff()
   batch_size = 4
   train_set = Head_Motion_Dataset(train_x, train_y)
   train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
   model = Head_Motion_Predict().cuda()
   loss = nn.L1Loss().cuda()
   optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
   num_epoch = 2
   for epoch in range(num_epoch):
       epoch_start_time = time.time()
       train_acc = 0.0
       train_loss = 0.0

       model.train()
       for i, data in enumerate(train_loader):
           optimizer.zero_grad()
           train_pred = model(data[0].cuda())
           batch_loss =loss(train_pred, data[1].cuda())
           batch_loss.backward()
           optimizer.step()

           train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
           train_loss += batch_loss.item()




def test():
    a = np.transpose(np.array([1,2,3,4]))
   # a = np.array([1,2,3,4])
    print(a)
    print(np.size(a))
    t =Prepross_Data(a, 20, 50)
    batch_size = 12
if __name__ == '__main__':
    main( )



