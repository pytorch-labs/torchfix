# Check no crash on no-module import
from . import residue_constants as rc

import torch
import functorch
from torch.func import (
    vmap,
    grad,
    vjp,
    jvp,
    jacrev,
    jacfwd,
    hessian,
    functionalize,
)

batch_size, feature_size = 3, 5
weights = torch.randn(feature_size, requires_grad=True)

def model(feature_vec):
    # Very simple linear model with activation
    return feature_vec.dot(weights).relu()

examples = torch.randn(batch_size, feature_size)
result = torch.func.vmap(model)(examples)
print(result)

# Non-runnable, just to check name changes.
def f():
    td_out = torch.func.vmap(tdmodule, (None, 0))(td, params)
    torch.func.vmap()
    torch.func.grad()
    torch.func.vjp()
    torch.func.jvp()
    torch.func.jacrev()
    torch.func.jacfwd()
    torch.func.hessian()
    torch.func.functionalize()

# Don't modify these, change the imports in the beginning
def f():
    td_out = vmap(tdmodule, (None, 0))(td, params)
    vmap()
    grad()
    vjp()
    jvp()
    jacrev()
    jacfwd()
    hessian()
    functionalize()

# Don't modify, as some symbols are not in func.torch.
from functorch import (
    make_functional_with_buffers as make_functional_functorch,
    vmap,
)
from functorch import FunctionalModule, FunctionalModuleWithBuffers
