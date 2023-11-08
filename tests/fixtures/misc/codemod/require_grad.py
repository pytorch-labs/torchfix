import torch
x = torch.zeros(1)
x.require_grad = False
x.require_grad = True

# from https://github.com/pytorch/test-infra/issues/4687
import torch.nn as nn
model = nn.Module()
for name, param in model.named_parameters():
    param.require_grad = 'specific_layer' in name
