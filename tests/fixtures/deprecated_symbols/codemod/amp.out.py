import torch

dtype = torch.float32

maybe_autocast = torch.amp.autocast("cuda")
maybe_autocast = torch.amp.autocast("cuda", dtype=torch.bfloat16)
maybe_autocast = torch.amp.autocast("cuda", dtype=dtype)

maybe_autocast = torch.amp.autocast("cpu")
maybe_autocast = torch.amp.autocast("cpu", dtype=torch.bfloat16)
maybe_autocast = torch.amp.autocast("cpu", dtype=dtype)
