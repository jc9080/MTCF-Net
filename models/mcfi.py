from __future__ import annotations

import torch
from torch import nn


class ConvolutionalPath(nn.Module):
    """Local feature path used by MCFI.

    This path applies residual 1D convolutions to capture local and
    modality-specific patterns.

    Input/Output:
        (B, T, C)
    """

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than zero.")

        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")

        padding = kernel_size // 2

        self.layers = nn.Sequential(
            nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                embed_dim,
                embed_dim,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(embed_dim),
        )

        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # (B, T, C) -> (B, C, T)
        x = x.transpose(1, 2)
        x = self.layers(x)

        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)

        return self.activation(x + residual)


class TransformerPath(nn.Module):
    """Global interaction path used by MCFI.

    Multi-head self-attention models interactions among feature tokens.

    Input/Output:
        (B, T, C)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_ratio: int = 4,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, "
                f"but received embed_dim={embed_dim}, "
                f"num_heads={num_heads}."
            )

        hidden_dim = embed_dim * feedforward_ratio

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_x = self.norm1(x)

        attention_output, attention_weights = self.attention(
            query=normalized_x,
            key=normalized_x,
            value=normalized_x,
            need_weights=True,
            average_attn_weights=False,
        )

        x = x + attention_output
        x = x + self.feedforward(self.norm2(x))

        return x, attention_weights


class MCFI(nn.Module):
    """Multimodal Cooperative Feature Interaction module.

    The module contains two parallel paths:

    1. Convolutional path:
       captures local and modality-specific patterns.

    2. Transformer path:
       captures global and cross-feature interactions.

    Their outputs are added element-wise and batch-normalized.

    Input:
        (B, T, C)

    Output:
        (B, T, C)
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        kernel_size: int = 3,
        dropout: float = 0.1,
        feedforward_ratio: int = 4,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim

        self.convolutional_path = ConvolutionalPath(
            embed_dim=embed_dim,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        self.transformer_path = TransformerPath(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            feedforward_ratio=feedforward_ratio,
        )

        # BatchNorm1d expects (B, C, T).
        self.output_norm = nn.BatchNorm1d(embed_dim)
        self.output_activation = nn.GELU()

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (B, T, C), "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected feature dimension {self.embed_dim}, "
                f"but received {x.shape[-1]}."
            )

        convolutional_output = self.convolutional_path(x)

        transformer_output, attention_weights = self.transformer_path(x)

        # Equation-inspired fusion:
        # R_MCFI = BN(ConvPath(X) + TransformerPath(X))
        output = convolutional_output + transformer_output

        output = output.transpose(1, 2)
        output = self.output_norm(output)
        output = output.transpose(1, 2)

        output = self.output_activation(output)

        if return_attention:
            return output, attention_weights

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    model = MCFI(
        embed_dim=256,
        num_heads=8,
        kernel_size=3,
        dropout=0.1,
    )

    dummy_input = torch.randn(2, 128, 256)

    model.eval()

    with torch.no_grad():
        output, attention = model(
            dummy_input,
            return_attention=True,
        )

    print("Input shape:    ", tuple(dummy_input.shape))
    print("Output shape:   ", tuple(output.shape))
    print("Attention shape:", tuple(attention.shape))

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"Parameters:      {total_parameters:,}")
