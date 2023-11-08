import random
import torch
x = torch.zeros(1)
x.require_grad = False
x.require_grad = True
grad = random.choice([False, True])
x.require_grad = grad

# Don't trigger
x.requires_grad = False
require_grad = False
