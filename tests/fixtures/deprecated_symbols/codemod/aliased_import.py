import torch
import torch as foo
import torch as bar
import torch as baz

A = torch.arange(9.0).reshape(3, 3)
A3 = foo.chain_matmul(A, A, A)
rc1 = bar.cholesky(torch.mm(A3.t(), A3))
rc2 = baz.qr(torch.mm(A.t(), A))

