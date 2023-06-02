import torch
bad = torch.qr()
good = torch.linalg.qr()

import torch as tt
bad = tt.qr()
good = tt.linalg.qr()

from torch import linalg, qr
bad = qr()
good = torch.linalg.qr()

from torch import qr as old_qr
from torch.linalg import qr as new_qr
good = new_qr()
bad = old_qr()
