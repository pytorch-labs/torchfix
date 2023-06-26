import torch
from torch import symeig as xxx

E, Z = torch.symeig(A, eigenvectors, True)
E, Z = xxx(A, eigenvectors, True)

OK = symeig(A, eigenvectors, True)
