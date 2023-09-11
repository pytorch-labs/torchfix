# To check for no extra imports when models already imported
from torchvision import models

models.resnet50(pretrained=True)
models.resnet50(pretrained=False)
models.segmentation.deeplabv3_resnet50(pretrained=True)
