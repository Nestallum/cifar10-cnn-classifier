"""
dataset.py — Data loading, transforms, and DataLoader setup for CIFAR-10.
"""

from typing import Tuple

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Return train and validation transforms.

    Training applies data augmentation to improve generalization.
    Validation only normalizes — no augmentation, for unbiased evaluation.
    """
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.2),
            ratio=(0.3, 3.3),
        ),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        ),
    ])

    return train_transform, val_transform


def get_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Download CIFAR-10 and return train and validation DataLoaders.

    Args:
        data_dir: Path where the dataset will be stored.
        batch_size: Number of samples per batch.
        num_workers: Number of subprocesses for data loading.

    Returns:
        A tuple (train_loader, val_loader).
    """
    train_transform, val_transform = get_transforms()

    train_dataset = CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    val_dataset = CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader