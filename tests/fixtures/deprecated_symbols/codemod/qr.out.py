import torch

a = torch.tensor([[12., -51, 4], [6, 167, -68], [-4, 24, -41]])
old_q, old_r = torch.linalg.qr(a)
new_q, new_r = torch.linalg.qr(a)
assert torch.allclose(old_q, new_q)
assert torch.allclose(old_r, new_r)

a = torch.randn(3, 4, 5)
old_q, old_r = torch.linalg.qr(a, mode="complete")
new_q, new_r = torch.linalg.qr(a, mode="complete")
assert torch.allclose(old_q, new_q)
assert torch.allclose(old_r, new_r)

a = torch.randn(3, 4, 5)
old_q, old_r = torch.linalg.qr(a)
new_q, new_r = torch.linalg.qr(a)
assert torch.allclose(old_q, new_q)
assert torch.allclose(old_r, new_r)
