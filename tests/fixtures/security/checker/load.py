import random
import torch
# Not OK
torch.load('tensors.pt')
# All these are OK
torch.load('tensors.pt', weights_only=True)
torch.load('tensors.pt', weights_only=False)
use_weights_only = random.choice([False, True])
torch.load('tensors.pt', weights_only=use_weights_only)
