"""Training infrastructure for Dynamic Kernel Attention.

Provides losses, optimizer configuration, LR scheduling, EMA, and the
main training loop.

Usage:
    from dka.training import (
        DKALoss, LabelSmoothingCrossEntropy, KernelDiversityLoss,
        build_optimizer, clip_grad_norms,
        build_scheduler, WarmupCosineScheduler,
        EMA,
        Trainer,
    )
"""

from .losses import (
    DKALoss,
    KernelDiversityLoss,
    LabelSmoothingCrossEntropy,
)
from .optimizer import build_optimizer, clip_grad_norms
from .scheduler import WarmupCosineScheduler, build_scheduler
from .ema import EMA
from .trainer import Trainer

__all__ = [
    "DKALoss",
    "KernelDiversityLoss",
    "LabelSmoothingCrossEntropy",
    "build_optimizer",
    "clip_grad_norms",
    "build_scheduler",
    "WarmupCosineScheduler",
    "EMA",
    "Trainer",
]
