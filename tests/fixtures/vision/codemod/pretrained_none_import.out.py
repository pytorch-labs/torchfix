# To check for no extra imports when weights is None
import torchvision

torchvision.models.resnet50(weights=None)
torchvision.models.segmentation.deeplabv3_resnet50(weights=None)
