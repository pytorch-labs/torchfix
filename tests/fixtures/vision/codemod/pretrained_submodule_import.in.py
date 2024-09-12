# Check that the codemod works when a model
# imported from a submodule
# i.e. from torchvision.models.resnet instead of
# directly from torchvision.models.

from torchvision import models
backbone = models.resnet.resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])

from torchvision.models import resnet
backbone = resnet.resnet101(pretrained=True, replace_stride_with_dilation=[False, True, True])

from torchvision.models.resnet import resnet152
resnet152(pretrained=True)
resnet152(pretrained=False)
