# Check that the codemod works when a model
# imported from a submodule
# i.e. from torchvision.models.resnet instead of
# directly from torchvision.models.

from torchvision import models
backbone = models.resnet.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, True, True])

from torchvision.models import resnet
backbone = resnet.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1, replace_stride_with_dilation=[False, True, True])

from torchvision.models.resnet import resnet152
resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
resnet152(weights=None)
