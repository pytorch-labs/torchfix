import torch
A = torch.randn(3, 3)
b = torch.randn(3)
torch.solve(A, b).solution
