import torch
torch.arange(0, 2)
torch.arange(-4, -1)
torch.arange(0, 8, 2)
torch.arange(4, 0, -2)
torch.arange(0, -6, -1)
torch.arange(start=0, end=2)
torch.arange(end=2, start=0)

res2 = torch.Tensor()
torch.arange(4, 30, out=res2)
res3 = torch.Tensor()
torch.arange(1, -1, -1, out=res3)
res4 = torch.arange(-2, 0)

features_size = 5
torch.arange(0, features_size)
torch.arange(0, features_size - 2 + 1)
torch.arange(0, 2 * features_size - features_size)

torch.arange(start=0, end=features_size)
torch.arange(start=0, end=-1 + features_size + 1)

dim = 5
torch.arange(1, dim + 1, dtype=torch.long)
torch.arange(1, dim + 1 + 1, dtype=torch.long)
torch.arange(1, end=dim + 1, step=1, dtype=torch.long)

# will not codemod
torch.range(0, 1, 1.0)
torch.range(-4, -2, 1 + 0)
torch.range(0, 6, 2.0)
torch.range(4, 2, 0 - 2)
