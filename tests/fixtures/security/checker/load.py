import random
import dill
import torch

# Not OK
torch.load('tensors.pt')
torch.load('f.pt', pickle_module=dill, encoding='utf-8')

# All these are OK
torch.load('tensors.pt', weights_only=True)
torch.load('tensors.pt', weights_only=False)
use_weights_only = random.choice([False, True])
torch.load('tensors.pt', weights_only=use_weights_only)
