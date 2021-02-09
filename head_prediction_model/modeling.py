import torch
import torch.nn as nn

'''搭建模型'''

class Head_Motion_Prediction(nn.Module):
    def __init__(self):
        super(Head_Motion_Prediction, self).__init__()

        self.cnn0_theta = nn.Sequential(
            nn.Conv1d(1, 1, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch size, 1, 18)
        self.cnn0_phi = nn.Sequential(
            nn.Conv1d(1, 1, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch size, 1, 18)
        self.cnn0_psi = nn.Sequential(
            nn.Conv1d(1, 1, 3),
            nn.BatchNorm1d(1),
            nn.ReLU(),
        )  # (Batch_size, 1, 18)
        self.dropout = nn.Dropout(0.2)

        self.GRU1_1 = nn.GRU(
            input_size=1,
            hidden_size=18,
            num_layers=2,
            batch_first=True,
        )  # (Batch_size, 18, 18)
        self.GRU1_2 = nn.GRU(
            input_size=1,
            hidden_size=18,
            num_layers=2,
            batch_first=True,
        )  # (Batch_size, 18, 18)
        self.GRU1_3 = nn.GRU(
            input_size=1,
            hidden_size=18,
            num_layers=2,
            batch_first=True,
        )  # (Batch_size, 18, 18)
        # (Batch_size, 3, 18, 18)

        self.capture_feature = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # (Batch_size, 64, 18, 18)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # (Batch_size, 64, 18, 18)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # (Batch_size, 64, 9, 9)
            nn.Conv2d(64, 128, 3, 1, 1),  # (Batch_size, 128, 9, 9)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),  # (Batch_size, 128, 9, 9)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, 1, 1),  # (Batch_size, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # (Batch_size, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),  # (Batch_size, 256, 4, 4)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # (Batch_size, 256, 2, 2)
            nn.Conv2d(256, 512, 3, 1, 1),  # (Batch_size, 512, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # (Batch_size, 512, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),  # (Batch_size, 512, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0), # (Batch_size, 512, 1, 1)
        )

        self.fnn = nn.Sequential(
            nn.Linear(512 * 1, 15 * 1),
            nn.ReLU(),
            nn.Linear(15 * 1, 3)
        )

    def forward(self, x):
        input1_1 = torch.zeros((len(x), 1, 20)).cuda()
        input1_1[:, 0, :] = x[:, 0, :]
        out1_1 = self.cnn0_theta(input1_1)

        input1_2 = torch.zeros((len(x), 1, 20)).cuda()
        input1_2[:, 0, :] = x[:, 1, :]
        out1_2 = self.cnn0_phi(input1_2)

        input1_3 = torch.zeros((len(x), 1, 20)).cuda()
        input1_3[:, 0, :] = x[:, 2, :]
        out1_3 = self.cnn0_psi(input1_3)
        # out1_1, out1_2, out1_3 = out1_1.cuda(), out1_2.cuda(), out1_3.cuda()

        out1_1 = out1_1.view(len(x), 18, 1).cuda()
        out1_2 = out1_2.view(len(x), 18, 1).cuda()
        out1_3 = out1_3.view(len(x), 18, 1).cuda()


        r_out1_1, h1_1 = self.GRU1_1(out1_1, None)
        r_out1_2, h1_2 = self.GRU1_2(out1_2, None)
        r_out1_3, h1_3 = self.GRU1_3(out1_3, None)
        # r_out1_1, r_out1_2, r_out1_3 = r_out1_1.cuda(), r_out1_2.cuda(), r_out1_3.cuda()

        input2 = torch.zeros((len(r_out1_1), 3, 18, 18)).cuda()
        input2[:, 0, :, :] = r_out1_1
        input2[:, 1, :, :] = r_out1_2
        input2[:, 2, :, :] = r_out1_3

        out2 = self.capture_feature(input2)

        input3 = out2[:, :, 0, 0]

        out = self.fnn(input3)

        return out