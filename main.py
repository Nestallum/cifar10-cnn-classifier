"""
main.py — Entry point: loads config, builds model, trains and evaluates.
"""

import yaml
import torch

from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.train import train
from src.evaluate import evaluate
from src.utils import get_device, get_logger, set_seed


def main() -> None:
    # Load config
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup
    logger = get_logger(__name__)
    set_seed(config["training"]["seed"])
    device = get_device()

    logger.info("Loading dataset...")
    train_loader, val_loader = get_dataloaders(
        data_dir="data/",
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
    )

    logger.info("Building model...")
    model = SimpleCNN(
        num_classes=config["data"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    logger.info(f"Model: {config['model']['name']} | Device: {device}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    logger.info("Starting training...")
    train(model, train_loader, val_loader, config, device)

    logger.info("Evaluating model...")
    evaluate(model, val_loader, device, save_path="confusion_matrix.png")

    logger.info("Done.")


if __name__ == "__main__":
    main()