"""Data loaders for DKA experiments.

Provides dataset-specific loaders for CIFAR-10, Tiny ImageNet, AG News,
and WikiText-2, each with appropriate augmentation and preprocessing.

Usage:
    from dka.data import get_cifar10_loaders, get_tinyimagenet_loaders
    from dka.data import get_agnews_loaders, get_wikitext2_loaders
    from dka.data import Mixup, CutMix, MixupCutMix
"""

from .cifar10 import get_cifar10_loaders, Mixup, CutMix, MixupCutMix
from .tinyimagenet import get_tinyimagenet_loaders
from .agnews import get_agnews_loaders
from .wikitext2 import get_wikitext2_loaders

__all__ = [
    "get_cifar10_loaders",
    "get_tinyimagenet_loaders",
    "get_agnews_loaders",
    "get_wikitext2_loaders",
    "Mixup",
    "CutMix",
    "MixupCutMix",
]
