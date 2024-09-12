from torch.utils._pytree import register_pytree_node

register_pytree_node()

from torch.utils import _pytree as xx
xx.register_pytree_node()

import torch
torch.utils._pytree.register_pytree_node()
