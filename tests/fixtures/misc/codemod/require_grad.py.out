import torch
x = torch.zeros(1)
x.requires_grad = False
x.requires_grad = True

# from https://github.com/pytorch/test-infra/issues/4687
import torch.nn as nn
model = nn.Module()
for name, param in model.named_parameters():
    param.requires_grad = 'specific_layer' in name
