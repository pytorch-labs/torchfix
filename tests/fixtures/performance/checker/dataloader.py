import torch
dataset = torch.rand((100, 100))
sync_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
async_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=4)
sync_again_dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, num_workers=0)
sync3_dataloader = torch.utils.data.DataLoader(dataset, 10, False, None, None, 0)
async2_dataloader = torch.utils.data.DataLoader(dataset, 10, False, None, None, 2)
