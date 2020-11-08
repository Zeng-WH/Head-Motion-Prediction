import torch
import torch.nn as nn
import torch.nn.functional as F
rnn = nn.GRU(10,20,2)
input = torch.randn(5,3,10)
h0 = torch.randn(2,3,20)
output, hn = rnn(input,  h0)
print(output)