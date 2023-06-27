import torch
x = torch.zeros(1)
x.require_grad = False
x.require_grad = True

# Don't trigger
x.requires_grad = False
require_grad = False
x.require_grad = 10
