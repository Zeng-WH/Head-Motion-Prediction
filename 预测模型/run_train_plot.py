import preprocess
import modeling
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    x_train_set = np.load('x_train_set.npy')
    y_train_set = np.load('y_train_set.npy')
    x_val_set = np.load('x_val_set.npy')
    y_val_set = np.load('y_val_set.npy')
    batch_size = 512

    train_set = preprocess.Head_Motion_Dataset(x_train_set, y_train_set)
    val_set = preprocess.Head_Motion_Dataset(x_val_set, y_val_set)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = modeling.Head_Motion_Predict().cuda()
    loss = nn.L1Loss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 2000
    epoch_list = []
    train_loss_list = []
    val_loss_list = []
    print('--------------------------------Run Training----------------------------------')
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_loss = 0.0
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
            epoch_list.append(epoch+1)
            train_loss_list.append(train_loss / len(train_loader))
            val_loss_list.append(val_loss / len(val_loader))
    np.save('epoch.npy', epoch_list)
    np.save('train_loss.npy', train_loss_list)
    np.save('val_loss.npy', val_loss_list)
if __name__ == '__main__':
    main()




