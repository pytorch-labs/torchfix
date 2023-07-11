import torch

A = torch.arange(2 * 3).view(2, 3)
B = torch.arange(3 * 2).view(3, 2)
C = torch.arange(2 * 2).view(2, 2)

old = torch.chain_matmul(A, B, C)
new = torch.linalg.multi_dot([A, B, C])
assert torch.allclose(old, new)

old = torch.randint_like(C, 10)
new = torch.randint_like(C, 10)
torch.chain_matmul(A, B, C, out=old)
torch.linalg.multi_dot([A, B, C], out=new)
assert torch.allclose(old, new)
