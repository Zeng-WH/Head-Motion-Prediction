import torch
import torch.nn as nn
a = torch.randn((2, 1, 10))

b = a.view(2, 10, 1)

print('bupt')
