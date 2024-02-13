import torch
from something import _check_with

# private
torch._check_with()
torch._warn_typed_storage_removal()

# considered public
torch._stack()
torch._log_softmax()

# Non-PyTorch calls
_check_with()

def _f():
    pass
_f()

# Dunder methods
torch.nn.modules.module.Module.__init__()
torch.__config__.show()
torch.__version__.split()
