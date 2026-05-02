"""
model.py — CNN architecture for CIFAR-10 classification.
"""

import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic residual block used in ResNet18.

    Two 3x3 convolutions with a skip connection.
    When stride > 1 or in_channels != out_channels, a 1x1 convolution
    adapts the skip connection to match the output shape.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut path: identity if shapes match, otherwise 1x1 conv
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = nn.functional.relu(out, inplace=True)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity                    # the residual addition
        out = nn.functional.relu(out, inplace=True)
        return out

class ResNet18(nn.Module):
    """
    ResNet18 adapted for CIFAR-10 (32x32 images).

    Differs from the original ImageNet ResNet18 in two places:
        - Initial conv is 3x3 stride 1 (instead of 7x7 stride 2)
        - No initial maxpool (would reduce 32x32 too aggressively)

    Architecture:
        - Initial conv: 3 -> 64 channels
        - Stage 1: 2 BasicBlocks at 64 channels   (32x32)
        - Stage 2: 2 BasicBlocks at 128 channels  (16x16, downsample)
        - Stage 3: 2 BasicBlocks at 256 channels  (8x8, downsample)
        - Stage 4: 2 BasicBlocks at 512 channels  (4x4, downsample)
        - Global average pooling + linear classifier
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Initial conv (adapted for 32x32 input)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 4 stages of 2 BasicBlocks each
        self.stage1 = self._make_stage(in_channels=64,  out_channels=64,  stride=1)
        self.stage2 = self._make_stage(in_channels=64,  out_channels=128, stride=2)
        self.stage3 = self._make_stage(in_channels=128, out_channels=256, stride=2)
        self.stage4 = self._make_stage(in_channels=256, out_channels=512, stride=2)

        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
    ) -> nn.Sequential:
        """Build a stage of 2 BasicBlocks. Only the first block downsamples."""
        return nn.Sequential(
            BasicBlock(in_channels, out_channels, stride=stride),
            BasicBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = nn.functional.relu(x, inplace=True)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class SimpleCNN(nn.Module):
    """
    A simple CNN for CIFAR-10 classification.

    Architecture:
        - 3 convolutional blocks (Conv -> BatchNorm -> ReLU -> MaxPool)
        - Fully connected classifier with dropout
    """

    def __init__(self, num_classes: int, dropout: float) -> None:
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x