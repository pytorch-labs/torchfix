import torch
torch.range(0, 1)
torch.range(-4, -2)
torch.range(0, 6, 2)
torch.range(4, 2, -2)
torch.range(0, -5, -1)
torch.range(start=0, end=1)
torch.range(end=1, start=0)

res2 = torch.Tensor()
torch.range(4, 29, out=res2)
res3 = torch.Tensor()
torch.range(1, 0, -1, out=res3)
res4 = torch.range(-2, -1)

features_size = 5
torch.range(0, features_size - 1)
torch.range(0, features_size - 2)
torch.range(0, 2 * features_size - features_size - 1)

torch.range(start=0, end=features_size - 1)
torch.range(start=0, end=-1 + features_size)

dim = 5
torch.range(1, dim, dtype=torch.long)
torch.range(1, dim + 1, dtype=torch.long)
torch.range(1, end=dim, step=1, dtype=torch.long)

# will not codemod
torch.range(0, 1, 1.0)
torch.range(-4, -2, 1 + 0)
torch.range(0, 6, 2.0)
torch.range(4, 2, 0 - 2)
