import torch
a = torch.randn(5)
b = torch.log(1 + a)
c = torch.log(a + 1)
b = torch.log(1.0 + a)
c = torch.log(a + 1.0)

# False negative: can not detect currently
x = (a + 1).log()
