import torch
from torch.utils.checkpoint import checkpoint
def gn(x, y):
    return torch.sigmoid(torch.matmul(x, y))
def fn(x, y):
    return checkpoint(gn, torch.sin(x), y)
