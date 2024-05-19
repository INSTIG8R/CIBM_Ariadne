import torch
import torch.nn.functional as F

x = torch.rand(8,1,224,224)
F.one_hot(x.long())

print(x.shape)