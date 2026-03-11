"""CIFAR-10 data loader with full augmentation pipeline.

Training augmentations (applied per-sample via torchvision transforms):
    - RandAugment (2 ops, magnitude 9)
    - Random Horizontal Flip
    - Random Crop 32x32 with 4px padding

Batch-level augmentations (returned as separate callables):
    - Mixup (alpha=0.8)
    - CutMix (alpha=1.0)

Validation: normalize only.

Reference: DKA Build Guide, Sections 4.1, 5.3, 5.4.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# CIFAR-10 channel-wise mean and std (precomputed over training set)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class Mixup:
    """Mixup augmentation applied at the batch level.

    Generates mixed samples:  x' = lam * x + (1 - lam) * x[perm]
    and mixed labels:         y' = lam * y_onehot + (1 - lam) * y_onehot[perm]

    Args:
        alpha: Beta distribution parameter. Higher alpha -> more mixing.
        num_classes: Number of classes for one-hot encoding.
    """

    def __init__(self, alpha: float = 0.8, num_classes: int = 10):
        self.alpha = alpha
        self.num_classes = num_classes

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup to a batch.

        Args:
            images: (B, C, H, W) batch of images.
            targets: (B,) integer class labels.

        Returns:
            mixed_images: (B, C, H, W) mixed images.
            mixed_targets: (B, num_classes) soft label distribution.
        """
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        batch_size = images.size(0)
        perm = torch.randperm(batch_size, device=images.device)

        mixed_images = lam * images + (1.0 - lam) * images[perm]

        targets_onehot = F.one_hot(targets, self.num_classes).float()
        mixed_targets = lam * targets_onehot + (1.0 - lam) * targets_onehot[perm]

        return mixed_images, mixed_targets


class CutMix:
    """CutMix augmentation applied at the batch level.

    Cuts a random rectangular region from one image and pastes it onto another.
    Labels are mixed proportionally to the area of the cut region.

    Args:
        alpha: Beta distribution parameter for lambda sampling.
        num_classes: Number of classes for one-hot encoding.
    """

    def __init__(self, alpha: float = 1.0, num_classes: int = 10):
        self.alpha = alpha
        self.num_classes = num_classes

    def _rand_bbox(
        self, H: int, W: int, lam: float
    ) -> tuple[int, int, int, int]:
        """Sample a random bounding box whose area ratio is (1 - lam)."""
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(H * cut_ratio)
        cut_w = int(W * cut_ratio)

        cy = np.random.randint(H)
        cx = np.random.randint(W)

        y1 = np.clip(cy - cut_h // 2, 0, H)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        x1 = np.clip(cx - cut_w // 2, 0, W)
        x2 = np.clip(cx + cut_w // 2, 0, W)

        return y1, y2, x1, x2

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply CutMix to a batch.

        Args:
            images: (B, C, H, W) batch of images.
            targets: (B,) integer class labels.

        Returns:
            mixed_images: (B, C, H, W) images with cut-and-pasted regions.
            mixed_targets: (B, num_classes) soft label distribution.
        """
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0
        batch_size = images.size(0)
        perm = torch.randperm(batch_size, device=images.device)

        _, _, H, W = images.shape
        y1, y2, x1, x2 = self._rand_bbox(H, W, lam)

        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        # Adjust lambda to the actual area ratio of the pasted region
        lam_adjusted = 1.0 - ((y2 - y1) * (x2 - x1)) / (H * W)

        targets_onehot = F.one_hot(targets, self.num_classes).float()
        mixed_targets = lam_adjusted * targets_onehot + (1.0 - lam_adjusted) * targets_onehot[perm]

        return mixed_images, mixed_targets


class MixupCutMix:
    """Randomly applies either Mixup or CutMix with equal probability.

    This is the standard approach used in DeiT and other ViT training recipes.

    Args:
        mixup_alpha: Alpha for Mixup Beta distribution.
        cutmix_alpha: Alpha for CutMix Beta distribution.
        num_classes: Number of classes for one-hot encoding.
        mixup_prob: Probability of choosing Mixup over CutMix.
    """

    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        num_classes: int = 10,
        mixup_prob: float = 0.5,
    ):
        self.mixup = Mixup(alpha=mixup_alpha, num_classes=num_classes)
        self.cutmix = CutMix(alpha=cutmix_alpha, num_classes=num_classes)
        self.mixup_prob = mixup_prob

    def __call__(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply Mixup or CutMix randomly.

        Args:
            images: (B, C, H, W) batch of images.
            targets: (B,) integer class labels.

        Returns:
            mixed_images: (B, C, H, W) augmented images.
            mixed_targets: (B, num_classes) soft label distribution.
        """
        if np.random.rand() < self.mixup_prob:
            return self.mixup(images, targets)
        else:
            return self.cutmix(images, targets)


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, MixupCutMix]:
    """Create CIFAR-10 training and validation data loaders.

    Args:
        data_dir: Root directory for dataset download/storage.
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.

    Returns:
        train_loader: DataLoader for training set with augmentations.
        val_loader: DataLoader for validation/test set (normalize only).
        mixup_cutmix: MixupCutMix callable for batch-level augmentation.
            Call as: images, targets = mixup_cutmix(images, targets)
            during training. Returns soft labels of shape (B, 10).
    """
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )

    mixup_cutmix = MixupCutMix(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        num_classes=10,
    )

    return train_loader, val_loader, mixup_cutmix
