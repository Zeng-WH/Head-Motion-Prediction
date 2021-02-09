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
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=10,
            num_layers=2,
            batch_first=True,
        )
        #(Batch_size, 18, 10)
        self.fnn = nn.Sequential(
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
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

        input2 = torch.zeros((len(x), 18, 3)).cuda()
        input2[:, :, 0] = out1_1[:, 0, :]
        input2[:, :, 1] = out1_2[:, 0, :]
        input2[:, :, 2] = out1_3[:, 0, :]


        out2,_ = self.lstm(input2)

        # (Batch_size, 18, 10)
        out2 = out2.reshape(len(x)*18, 10)

        out3 = self.fnn(out2)

        out3 = out3.reshape(len(x), 18, 3)

        return out3[:, -1, :]






