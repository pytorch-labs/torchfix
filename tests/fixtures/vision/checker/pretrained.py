import torchvision
import random

# deprecated
torchvision.models.resnet50(pretrained=True)
torchvision.models.resnet50(True)
torchvision.models.resnet50(pretrained=False)
torchvision.models.resnet50(False)
torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained_backbone=True)
torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True, pretrained_backbone=True)

pretrained = random.choice([False, True])
# can't codemod, but can report
torchvision.models.resnet50(pretrained=pretrained)

# ok
from torchvision.prototype.models import ResNet50_Weights
torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
torchvision.models.resnet50(weights=None)
