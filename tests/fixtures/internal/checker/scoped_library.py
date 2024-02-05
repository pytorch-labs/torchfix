import torch
from torch.library import Library, impl, fallthrough_kernel
my_lib1 = Library("aten", "IMPL")
