from torch.utils.data import _utils
batch = _utils.collate.default_collate(batch)

from torch.utils.data._utils.collate import default_collate
inputs, labels, video_idx = default_collate(inputs), default_collate(labels), default_collate(video_idx)

from torch.utils.data._utils.collate import default_convert
values = default_convert(values)

import torch
values = torch.utils.data._utils.collate.default_convert(values)
