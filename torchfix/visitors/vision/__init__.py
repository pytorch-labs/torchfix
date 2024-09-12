from .pretrained import TorchVisionDeprecatedPretrainedVisitor
from .singleton_import import TorchVisionSingletonImportVisitor
from .to_tensor import TorchVisionDeprecatedToTensorVisitor

__all__ = [
    "TorchVisionDeprecatedPretrainedVisitor",
    "TorchVisionDeprecatedToTensorVisitor",
    "TorchVisionSingletonImportVisitor",
]
