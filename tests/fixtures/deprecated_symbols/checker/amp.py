import torch

torch.cuda.amp.autocast()
torch.cuda.amp.custom_fwd()
torch.cuda.amp.custom_bwd()

dtype = torch.float32
maybe_autocast = torch.cpu.amp.autocast()
maybe_autocast = torch.cpu.amp.autocast(dtype=torch.bfloat16)
maybe_autocast = torch.cpu.amp.autocast(dtype=dtype)
