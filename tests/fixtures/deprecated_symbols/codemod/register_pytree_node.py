from torch.utils._pytree import _register_pytree_node

_register_pytree_node()

from torch.utils import _pytree as xx
xx._register_pytree_node()

import torch
torch.utils._pytree._register_pytree_node()
