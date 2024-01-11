from torch import nn
m = nn.utils.weight_norm(nn.Linear(20, 40), name='weight')
