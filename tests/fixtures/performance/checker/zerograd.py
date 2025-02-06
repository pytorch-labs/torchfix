import torch
import torch.nn as nn

x = torch.ones((100, 100))
model = nn.Sequential()
optimizer = torch.optim.Adam(model.parameters(),foreach=True)

# This should raise flags
optimizer.zero_grad(set_to_none=False)
model.zero_grad(set_to_none=False)

# This should not raise flags 
optimizer.zero_grad()
model.zero_grad()


