import torch
from torch.backends import cuda
from torch.backends.cuda import sdp_kernel

with torch.backends.cuda.sdp_kernel() as context:
    pass

with cuda.sdp_kernel() as context:
    pass

with sdp_kernel() as context:
    pass
