# To check for no extra imports when models already imported
from torchvision import models

models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
models.resnet50(weights=None)
models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
