import torch
a = torch.randn(5)
b = torch.exp(a) - 1
c = torch.exp(a) - 1.0

ret = (torch.exp(a) - 1) * torch.exp(2 * b)

# False negative: can not detect currently
x = a.exp() - 1

# False negative: should be rare and would complicate implementation
x = -1 + torch.exp(a)
