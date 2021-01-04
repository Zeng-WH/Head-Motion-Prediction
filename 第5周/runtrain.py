import preprocess
import modeling
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time
import modeling_simply
import modeling_simply_pro

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_set = preprocess.read_all_file('/home/wlw2020/zwh/数据--11.19/数据--11.19')
    train = data_set[0: math.floor(0.8 * len(data_set)), :]
    val = data_set[math.floor(0.8 * len(data_set)): len(data_set), :]
    train_pre = preprocess.Prepross_Data(train, 20, 50)
    train_x, train_y = train_pre.data_split()
    val_pre = preprocess.Prepross_Data(val, 20, 50)
    val_x, val_y = val_pre.data_split()
    batch_size = 512
    train_set = preprocess.Head_Motion_Dataset(train_x, train_y)
    val_set = preprocess.Head_Motion_Dataset(val_x, val_y)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    #torch.save(train_loader, 'train_loader.pt')
    model = modeling_simply.Head_Motion_Predict().cuda()
    loss = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 5000
    print('--------------------------------Run Training----------------------------------')
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss.item()

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())
                val_loss += batch_loss.item()
            # 将结果print出来
            print('[%03d/%03d] %2.2f sec(s)  Train Loss: %3.6f | Val Loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time,
                    train_loss / len(train_loader),
                   val_loss / len(val_loader) ))

if __name__ == '__main__':
    main()
