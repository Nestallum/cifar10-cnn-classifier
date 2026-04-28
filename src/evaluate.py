"""
evaluate.py — Final model evaluation with metrics and confusion matrix.
"""

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[List[int], List[int]]:
    """
    Run inference on a DataLoader and collect all predictions and ground truths.

    Returns:
        A tuple (all_preds, all_labels) of flat integer lists.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return all_preds, all_labels


def print_classification_report(
    all_preds: List[int],
    all_labels: List[int],
) -> None:
    """Print per-class precision, recall, F1-score and overall accuracy."""
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=CIFAR10_CLASSES))


def plot_confusion_matrix(
    all_preds: List[int],
    all_labels: List[int],
    save_path: str = "confusion_matrix.png",
) -> None:
    """
    Plot and save a normalized confusion matrix.

    Args:
        all_preds: List of predicted class indices.
        all_labels: List of ground truth class indices.
        save_path: Path where the figure will be saved.
    """
    cm = confusion_matrix(all_labels, all_preds, normalize="true")

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_yticks(np.arange(len(CIFAR10_CLASSES)))
    ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CIFAR10_CLASSES)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Normalized Confusion Matrix — CIFAR-10")

    for i in range(len(CIFAR10_CLASSES)):
        for j in range(len(CIFAR10_CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str = "confusion_matrix.png",
) -> None:
    """
    Run full evaluation: classification report + confusion matrix.

    Args:
        model: Trained model to evaluate.
        loader: DataLoader for the evaluation set.
        device: Device to run inference on.
        save_path: Path to save the confusion matrix figure.
    """
    all_preds, all_labels = get_predictions(model, loader, device)
    print_classification_report(all_preds, all_labels)
    plot_confusion_matrix(all_preds, all_labels, save_path=save_path)