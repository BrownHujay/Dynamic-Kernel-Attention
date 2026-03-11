"""DKA models and baselines.

Exports the core DKA models and all baseline models
for comparison experiments.
"""

from .kernel_generator import FactoredKernelGenerator
from .dka_module import DKAModule
from .dka_block import DKABlock
from .dka_model import DKAImageModel, DKATextModel

from .baselines import (
    # ViT
    ViT,
    ViTForTextClassification,
    vit_small_cifar10,
    vit_small_tinyimagenet,
    vit_small_agnews,
    # ResNet
    ResNet,
    BasicBlock,
    resnet18_cifar10,
    resnet18_tinyimagenet,
    # ConvNeXt
    ConvNeXt,
    convnext_tiny_cifar10,
    convnext_tiny_tinyimagenet,
)

__all__ = [
    # DKA
    "FactoredKernelGenerator",
    "DKAModule",
    "DKABlock",
    "DKAImageModel",
    "DKATextModel",
    # ViT
    "ViT",
    "ViTForTextClassification",
    "vit_small_cifar10",
    "vit_small_tinyimagenet",
    "vit_small_agnews",
    # ResNet
    "ResNet",
    "BasicBlock",
    "resnet18_cifar10",
    "resnet18_tinyimagenet",
    # ConvNeXt
    "ConvNeXt",
    "convnext_tiny_cifar10",
    "convnext_tiny_tinyimagenet",
]
