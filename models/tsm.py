from __future__ import annotations

import torch
import torch.nn as nn


class TemporalShift(nn.Module):
    """
    Temporal Shift Module (TSM).

    Input:
        (B, T, C)

    Output:
        (B, T, C)
    """

    def __init__(self, fold_div: int = 4):
        super().__init__()
        self.fold_div = fold_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        fold = C // self.fold_div

        out = torch.zeros_like(x)

        # Shift forward (features from next time step)
        out[:, :-1, :fold] = x[:, 1:, :fold]

        # Shift backward (features from previous time step)
        out[:, 1:, fold:2 * fold] = x[:, :-1, fold:2 * fold]

        # Keep remaining channels unchanged
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]

        return out


if __name__ == "__main__":
    model = TemporalShift()

    x = torch.randn(8, 128, 64)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
    print("Same tensor?:", torch.equal(x, y))
