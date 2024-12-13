import torch

dtype = torch.float32

maybe_autocast = torch.cuda.amp.autocast()
maybe_autocast = torch.cuda.amp.autocast(dtype=torch.bfloat16)
maybe_autocast = torch.cuda.amp.autocast(dtype=dtype)

maybe_autocast = torch.cpu.amp.autocast()
maybe_autocast = torch.cpu.amp.autocast(dtype=torch.bfloat16)
maybe_autocast = torch.cpu.amp.autocast(dtype=dtype)
