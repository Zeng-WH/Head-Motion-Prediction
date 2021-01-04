import preprocess
import modeling
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time
import modeling_new_fly_net_innorm

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_set = preprocess.read_all_file_new('/home/wlw2020/zwh/数据--11.19/数据--11.19')
    data_pre = preprocess.Prepross_Data(data_set, 20, 50)
    x_set, y_set = data_pre.data_split()
    seed_1 = np.arange(len(x_set))
    #print(seed_1)
    np.random.shuffle(seed_1)
    x_train_set = x_set[seed_1[0: math.floor(0.8 * len(x_set))]]
    y_train_set = y_set[seed_1[0: math.floor(0.8 * len(x_set))]]
    x_val_set = x_set[seed_1[math.floor(0.8 * len(x_set)): len(x_set)]]
    y_val_set = y_set[seed_1[math.floor(0.8 * len(x_set)): len(x_set)]]
    batch_size = 512

    train_set = preprocess.Head_Motion_Dataset(x_train_set, y_train_set)
    val_set = preprocess.Head_Motion_Dataset(x_val_set, y_val_set)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = modeling_new_fly_net_innorm.Head_Motion_Predict().cuda()
    loss = nn.L1Loss().cuda()
    # loss1 = nn.L1Loss()
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
                   val_loss / len(val_loader)))


if __name__ == '__main__':
    main()