"""Tiny ImageNet data loader with full augmentation pipeline.

Tiny ImageNet: 200 classes, 64x64 images, 100k train / 10k val.

Training augmentations (per-sample):
    - Random Resized Crop to 64x64
    - Random Horizontal Flip
    - RandAugment (2 ops, magnitude 9)

Batch-level augmentations (returned as separate callable):
    - Mixup (alpha=0.8) / CutMix (alpha=1.0) via MixupCutMix

Validation: Resize(72) -> CenterCrop(64) -> Normalize.

Reference: DKA Build Guide, Sections 4.1, 5.3, 5.4.
"""

import os
import zipfile
import shutil
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .cifar10 import MixupCutMix


# ImageNet-style normalization (Tiny ImageNet is a subset)
TINYIMAGENET_MEAN = (0.4802, 0.4481, 0.3975)
TINYIMAGENET_STD = (0.2770, 0.2691, 0.2821)

# Tiny ImageNet download URL
TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def _organize_val_folder(val_dir: str) -> None:
    """Reorganize the Tiny ImageNet val folder into class subfolders.

    The raw download has val/images/ with a val_annotations.txt mapping
    filenames to class IDs. torchvision.ImageFolder needs images organized
    as val/<class>/<image>.JPEG. This function does that reorganization
    in-place.

    Args:
        val_dir: Path to the val/ directory inside tiny-imagenet-200/.
    """
    val_dir = Path(val_dir)
    annotations_file = val_dir / "val_annotations.txt"
    images_dir = val_dir / "images"

    if not annotations_file.exists():
        # Already organized or missing — skip
        return

    if not images_dir.exists():
        # Check if it looks already organized (class subdirs exist)
        return

    # Parse annotations: filename -> class_id
    with open(annotations_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            filename = parts[0]
            class_id = parts[1]

            class_dir = val_dir / class_id
            class_dir.mkdir(exist_ok=True)

            src = images_dir / filename
            dst = class_dir / filename
            if src.exists():
                shutil.move(str(src), str(dst))

    # Clean up the now-empty images directory
    if images_dir.exists() and not any(images_dir.iterdir()):
        images_dir.rmdir()


def _download_and_extract(data_dir: str) -> str:
    """Download and extract Tiny ImageNet if not already present.

    Args:
        data_dir: Root directory for dataset storage.

    Returns:
        Path to the tiny-imagenet-200/ directory.
    """
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "tiny-imagenet-200"

    if dataset_dir.exists() and (dataset_dir / "train").exists():
        # Already downloaded and extracted
        return str(dataset_dir)

    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "tiny-imagenet-200.zip"

    if not zip_path.exists():
        print(f"Downloading Tiny ImageNet to {zip_path}...")
        try:
            import urllib.request
            urllib.request.urlretrieve(TINYIMAGENET_URL, str(zip_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to download Tiny ImageNet from {TINYIMAGENET_URL}. "
                f"Please download manually and place at {zip_path}. Error: {e}"
            )

    print(f"Extracting Tiny ImageNet to {data_dir}...")
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        zf.extractall(str(data_dir))

    # Organize val folder for ImageFolder compatibility
    _organize_val_folder(str(dataset_dir / "val"))

    return str(dataset_dir)


def get_tinyimagenet_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    pin_memory: bool = True,
    image_size: int = 64,
) -> tuple[DataLoader, DataLoader, MixupCutMix]:
    """Create Tiny ImageNet training and validation data loaders.

    Downloads the dataset automatically if not found at data_dir.

    Args:
        data_dir: Root directory for dataset download/storage.
        batch_size: Batch size for both loaders.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for GPU transfer.
        image_size: Target image size (default 64 for Tiny ImageNet).

    Returns:
        train_loader: DataLoader for training set with augmentations.
        val_loader: DataLoader for validation set (resize + center crop + normalize).
        mixup_cutmix: MixupCutMix callable for batch-level augmentation.
            Call as: images, targets = mixup_cutmix(images, targets)
            during training. Returns soft labels of shape (B, 200).
    """
    dataset_dir = _download_and_extract(data_dir)
    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
    ])

    # Resize slightly larger then center crop to image_size
    resize_dim = int(image_size * 1.125)  # 72 for 64px images
    val_transform = transforms.Compose([
        transforms.Resize(resize_dim),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(TINYIMAGENET_MEAN, TINYIMAGENET_STD),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

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
        num_classes=200,
    )

    return train_loader, val_loader, mixup_cutmix
