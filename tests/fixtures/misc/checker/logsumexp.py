import torch
a = torch.randn(5)
b = torch.randn(5)

# logsumexp
y = torch.log(torch.sum(torch.exp(x), 1, keepdim=True))
y = torch.log(torch.sum(torch.exp(2.5 + x), 1))

# not logsumexp
y = torch.log(torch.sum(torch.exp(x), 1, keepdim=True) + 2.5)
y = torch.log(torch.sum(torch.exp(x) + 2.5, 1))
y = torch.log(2 + x)
y = torch.sum(torch.log(torch.exp(x)), 1)
y = torch.exp(torch.sum(torch.log(x), 1, keepdim=True))
y = torch.log(torch.sum(torch.exp(2.5)))
