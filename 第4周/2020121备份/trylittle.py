import torch
import torch.nn as nn

a = torch.ones((3, 5, 4))
b = torch.zeros((3, 5, 4))
loss = nn.L1Loss()

c = loss(a, b)

print('debug')