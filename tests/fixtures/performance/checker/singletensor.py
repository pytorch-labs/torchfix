import torch
import torch.nn as nn

x = torch.ones((100, 100))
model = nn.Sequential()


# These should raise flags
optimizer_adam = torch.optim.Adam(model.parameters())
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer_adamw = torch.optim.AdamW(model.parameters())

# These should not raise flags
optimizer_adam = torch.optim.Adam(model.parameters(), foreach=True)
optimizer_sgd = torch.optim.SGD(model.parameters(), lr=0.01, foreach=True)
optimizer_adamw = torch.optim.AdamW(model.parameters(), foreach=True)
optimizer_adamw = torch.optim.AdamW(model.parameters(), foreach=False)