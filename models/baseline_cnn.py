from __future__ import annotations

import torch
from torch import nn


class BaselineCNN(nn.Module):
    """Simple 1D-CNN baseline for UCI-HAR classification.

    Expected input shape:
        (batch, time, channels) = (B, 128, 6)
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 6,
    ) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input shape (B, T, C), but got {tuple(x.shape)}"
            )

        # Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.squeeze(-1)
        logits = self.classifier(x)

        return logits


if __name__ == "__main__":
    model = BaselineCNN()
    dummy_x = torch.randn(8, 128, 6)
    dummy_y = model(dummy_x)

    print("Input shape:", dummy_x.shape)
    print("Output shape:", dummy_y.shape)
