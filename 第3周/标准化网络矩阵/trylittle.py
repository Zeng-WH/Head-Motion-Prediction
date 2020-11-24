import torch
import torch.nn as nn

a = torch.randn((5, 3, 19))
b = torch.max(a, 2)[0]
print(b[1, :])
c = torch.repeat_interleave(b.unsqueeze(dim = 2), repeats=19, dim=2)
print(c[1, :, 1])
#x, y = torch.broadcast_tensors(b, a)
d = a/c


print('debug')