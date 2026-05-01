"""
train.py — Training loop with TensorBoard logging and checkpoint saving.
"""

import os
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Run a single training epoch.

    Returns:
        A dict with 'loss' and 'accuracy' for this epoch.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate the model on the validation set.

    Returns:
        A dict with 'loss' and 'accuracy'.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return {
        "loss": running_loss / total,
        "accuracy": correct / total,
    }


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: dict,
    device: torch.device,
) -> None:
    """
    Full training loop: trains for N epochs, logs to TensorBoard,
    and saves checkpoints periodically.

    Args:
        model: The CNN model to train.
        train_loader: DataLoader for the training set.
        val_loader: DataLoader for the validation set.
        config: Full config dict loaded from config.yaml.
        device: Device to train on (CPU or CUDA).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    warmup_epochs = config["training"]["warmup_epochs"]
    epochs = config["training"]["epochs"]
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=epochs - warmup_epochs,
            ),
        ],
        milestones=[warmup_epochs],
    )

    writer = SummaryWriter(log_dir=config["logging"]["log_dir"])
    checkpoint_dir = config["logging"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = validate(model, val_loader, criterion, device)
        scheduler.step()

        # TensorBoard logging
        writer.add_scalars("Loss", {
            "train": train_metrics["loss"],
            "val": val_metrics["loss"],
        }, epoch)
        writer.add_scalars("Accuracy", {
            "train": train_metrics["accuracy"],
            "val": val_metrics["accuracy"],
        }, epoch)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f}"
        )

        # Save checkpoint
        if epoch % config["logging"]["save_every"] == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_metrics["accuracy"],
            }, checkpoint_path)

    writer.close()