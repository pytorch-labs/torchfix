import torch
def gn(x, y):
    return torch.sigmoid(torch.matmul(x, y))

import torch.utils.checkpoint
def fn(x, y):
    return checkpoint(gn, torch.sin(x), y)
    return checkpoint(gn, torch.sin(x), y, use_reentrant=False)

from torch.utils.checkpoint import checkpoint
def fn(x, y):
    return checkpoint(gn, torch.sin(x), y)
    return checkpoint(gn, torch.sin(x), y, use_reentrant=True)
