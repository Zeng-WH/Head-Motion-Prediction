import data_process
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import math
import time
import modeling

def main( ):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.set_num_threads(1)
    x_train_set = np.load('./x_train_set.npy')
    x_train_set = torch.tensor(x_train_set).cuda()
    y_train_set = np.load('./y_train_set.npy')
    y_train_set = torch.tensor(y_train_set).cuda()
    x_val_set = np.load('./x_val_set.npy')
    x_val_set = torch.tensor(x_val_set).cuda()
    y_val_set = np.load('./y_val_set.npy')
    y_val_set = torch.tensor(y_val_set).cuda()
    batch_size = 128
    print('batch_size', batch_size)
    print('7')

    train_set = data_process.Head_Motion_Dataset(x_train_set, y_train_set)
    val_set = data_process.Head_Motion_Dataset(x_val_set, y_val_set)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    model = modeling.Head_Motion_Prediction().cuda()
    loss = nn.L1Loss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epoch = 3000
    train_loss_list = []
    val_loss_list = []
    print('--------------------------------Run Training----------------------------------')
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_loss = torch.tensor(0.0).cuda()
        val_loss = torch.tensor(0.0).cuda()

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            train_pred = model(data[0].cuda())
            batch_loss = loss(train_pred, data[1].cuda())
            batch_loss.backward()
            optimizer.step()

            train_loss += batch_loss

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].cuda())
                batch_loss = loss(val_pred, data[1].cuda())
                val_loss += batch_loss
            # 将结果print出来
            print('[%03d/%03d] %2.2f sec(s)  Train Loss: %3.6f | Val Loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time() - epoch_start_time, train_loss / len(train_loader),
                   val_loss / len(val_loader)))
            train_loss_list.append(train_loss / len(train_loader))
            val_loss_list.append(val_loss / len(val_loader))
    '''
    print('-----------------------------Save Model----------------------------------------')
    torch.save(model, './motion_predict_little_predict_5000.pkl')
    np.save('train_loss_curve_5000.npy', train_loss_list)
    np.save('val_loss_curve_5000.npy', val_loss_list)
    '''


if __name__ == '__main__':
    main()

