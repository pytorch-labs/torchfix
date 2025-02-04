import torch

x = torch.randn(5)

# logsumexp
y = torch.log(torch.sum(torch.exp(x), 1, keepdim=True))
y = torch.log(torch.sum(torch.exp(x), dim=1, keepdim=True))
y = torch.log(torch.sum(torch.exp(2.5 + x), 1))
y = torch.log(torch.sum(torch.exp(2.5 + x), dim=1))

# not logsumexp
y = torch.log(torch.sum(torch.exp(x), 1, keepdim=True) + 2.5)
y = torch.log(torch.sum(torch.exp(x) + 2.5, 1))
y = torch.log(2 + x)
y = torch.sum(torch.log(torch.exp(x)), 1)
y = torch.exp(torch.sum(torch.log(x), 1, keepdim=True))

# not logsumexp because of https://github.com/pytorch/pytorch/issues/144339
y = torch.log(torch.sum(torch.exp(x), None, keepdim=True))
y = torch.log(torch.sum(torch.exp(x), dim=None, keepdim=True))
y = torch.log(torch.sum(torch.exp(x), keepdim=True))
