import torch
import torch.nn as nn

'''搭建模型'''
class Head_Motion_Prediction(nn.Module):
    def __init__(self):
        super(Head_Motion_Prediction, self).__init__()

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
        input1 = torch.zeros((len(x), 20, 3)).cuda()
        input1[:, :, 0] = x[:, 0, :]
        input1[:, :, 1] = x[:, 1, :]
        input1[:, :, 2] = x[:, 2, :]


        out2,_ = self.lstm(input1)

        # (Batch_size, 18, 10)
        out2 = out2.reshape(len(x)*20, 10)

        out3 = self.fnn(out2)

        out3 = out3.reshape(len(x), 20, 3)

        return out3[:, -1, :]






