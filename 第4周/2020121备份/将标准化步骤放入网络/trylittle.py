import torch
import torch.nn as nn

a = torch.tensor([[1, 2, 3 , 6, 8, 9],[4, 5, 6, 7, 10, 11]])
b = a.view(2, 3, 2)
for i in range(2):
    print(b[0,: ,:])
c = b.view(2, 6)
#print(b)
print('test')