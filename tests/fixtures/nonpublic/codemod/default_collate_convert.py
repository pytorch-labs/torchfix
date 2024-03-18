from torch.utils.data import _utils # will not be removed as it could be used for something besides default_collate
batch = _utils.collate.default_collate(batch)

from torch.utils.data._utils import collate # also will not be removed
batch = collate.default_collate(batch)

from torch.utils.data._utils.collate import default_collate
inputs, labels, video_idx = default_collate(inputs), default_collate(labels), default_collate(video_idx)

from torch.utils.data._utils.collate import default_convert
values = default_convert(values)

import torch
values = torch.utils.data._utils.collate.default_convert(values)
