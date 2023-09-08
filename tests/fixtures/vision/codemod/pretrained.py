import torchvision
import random
from torchvision.models.segmentation import deeplabv3_resnet50

torchvision.models.resnet50(pretrained=True)
torchvision.models.resnet50(True)
torchvision.models.resnet50(pretrained=False)
torchvision.models.resnet50(False)

torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
torchvision.models.segmentation.deeplabv3_resnet50(True)
deeplabv3_resnet50(pretrained=True)
deeplabv3_resnet50(True)

torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True,
                                                           pretrained_backbone=True)
torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained_backbone=True)

pretrained = random.choice([False, True])
# can't codemod
torchvision.models.resnet50(pretrained=pretrained)
