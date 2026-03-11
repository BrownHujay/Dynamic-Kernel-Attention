"""Baseline models for DKA comparison experiments.

Exports:
- ViT / ViTForTextClassification: Minimal Vision Transformer (from scratch)
- ResNet / BasicBlock: ResNet-18 adapted for CIFAR-10
- ConvNeXt: Minimal ConvNeXt-Tiny (from scratch)

Plus factory functions for standard configurations:
- vit_small_cifar10, vit_small_tinyimagenet, vit_small_agnews
- resnet18_cifar10, resnet18_tinyimagenet
- convnext_tiny_cifar10, convnext_tiny_tinyimagenet
"""

from .vit import (
    ViT,
    ViTForTextClassification,
    vit_small_cifar10,
    vit_small_tinyimagenet,
    vit_small_agnews,
)
from .resnet import (
    ResNet,
    BasicBlock,
    resnet18_cifar10,
    resnet18_tinyimagenet,
)
from .convnext import (
    ConvNeXt,
    convnext_tiny_cifar10,
    convnext_tiny_tinyimagenet,
)

__all__ = [
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
