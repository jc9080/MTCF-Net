from __future__ import annotations

import torch
from torch import nn


class AdaptiveFeatureFusion(nn.Module):
    """
    Adaptive Feature Fusion (ADF)

    Input:
        (B, T, C)

    Output:
        pooled feature (B, C)
    """

    def __init__(
        self,
        embed_dim: int = 2048,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        self.query = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):

        B = x.shape[0]

        query = self.query.expand(B, -1, -1)

        weighted, attn = self.attention(
            query=query,
            key=x,
            value=x,
        )

        weighted = weighted.expand(-1, x.shape[1], -1)

        x = self.norm(x + weighted)

        pooled = x.mean(dim=1)

        return pooled


if __name__ == "__main__":

    model = AdaptiveFeatureFusion()

    x = torch.randn(2,16,2048)

    y = model(x)

    print("Input :", x.shape)
    print("Output:", y.shape)
