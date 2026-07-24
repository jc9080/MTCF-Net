from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class TDM(nn.Module):
    """Temporal Dependency Modeling module.

    The module divides an input sequence into overlapping windows,
    aggregates each window through average pooling, applies global
    attention across windows, and projects the attended window
    representations back to the original temporal resolution.

    Input:
        (B, T, C)

    Output:
        (B, T, C)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        window_size: int = 8,
        stride: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        feedforward_ratio: int = 4,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError("embed_dim must be greater than zero.")

        if window_size <= 0:
            raise ValueError("window_size must be greater than zero.")

        if stride <= 0:
            raise ValueError("stride must be greater than zero.")

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads, "
                f"but received embed_dim={embed_dim}, "
                f"num_heads={num_heads}."
            )

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.stride = stride

        hidden_dim = embed_dim * feedforward_ratio

        self.norm1 = nn.LayerNorm(embed_dim)

        self.global_attention = nn.MultiheadAttention(
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

        self.output_norm = nn.LayerNorm(embed_dim)

    def _pad_sequence(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        """Pad the temporal dimension so all windows are complete."""

        batch_size, sequence_length, embed_dim = x.shape

        if sequence_length <= self.window_size:
            padded_length = self.window_size
        else:
            num_windows = (
                sequence_length - self.window_size + self.stride - 1
            ) // self.stride + 1

            padded_length = (
                (num_windows - 1) * self.stride
                + self.window_size
            )

        padding_length = padded_length - sequence_length

        if padding_length == 0:
            return x, 0

        # F.pad expects the last dimensions:
        # (feature_left, feature_right, time_left, time_right)
        x = F.pad(
            x,
            pad=(0, 0, 0, padding_length),
            mode="replicate",
        )

        return x, padding_length

    def _create_windows(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Create overlapping windows.

        Input:
            (B, T, C)

        Output:
            (B, N, P, C)

        N:
            number of windows

        P:
            window size
        """

        return x.unfold(
            dimension=1,
            size=self.window_size,
            step=self.stride,
        ).permute(0, 1, 3, 2).contiguous()

    def _restore_sequence(
        self,
        window_features: torch.Tensor,
        padded_length: int,
    ) -> torch.Tensor:
        """Restore window-level features to the temporal resolution.

        Overlapping positions are reconstructed by averaging the
        contributions of all windows covering each time step.
        """

        batch_size, num_windows, embed_dim = window_features.shape
        device = window_features.device
        dtype = window_features.dtype

        restored = torch.zeros(
            batch_size,
            padded_length,
            embed_dim,
            device=device,
            dtype=dtype,
        )

        counts = torch.zeros(
            batch_size,
            padded_length,
            1,
            device=device,
            dtype=dtype,
        )

        for window_index in range(num_windows):
            start = window_index * self.stride
            end = start + self.window_size

            expanded_window = window_features[
                :,
                window_index,
                :,
            ].unsqueeze(1)

            restored[:, start:end, :] += expanded_window
            counts[:, start:end, :] += 1.0

        restored = restored / counts.clamp_min(1.0)

        return restored

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

        original_length = x.shape[1]
        residual = x

        padded_x, padding_length = self._pad_sequence(x)
        padded_length = padded_x.shape[1]

        # (B, T, C) -> (B, N, P, C)
        windows = self._create_windows(padded_x)

        # Equation (6)-inspired average pooling:
        # W_i = (1 / P) * sum_j X_i,j
        window_features = windows.mean(dim=2)

        normalized_windows = self.norm1(window_features)

        attended_windows, attention_weights = self.global_attention(
            query=normalized_windows,
            key=normalized_windows,
            value=normalized_windows,
            need_weights=True,
            average_attn_weights=False,
        )

        window_features = window_features + attended_windows

        window_features = (
            window_features
            + self.feedforward(self.norm2(window_features))
        )

        global_features = self._restore_sequence(
            window_features=window_features,
            padded_length=padded_length,
        )

        if padding_length > 0:
            global_features = global_features[:, :original_length, :]

        # Equation (8)-inspired residual fusion:
        # R_output = R_global + X
        output = residual + global_features
        output = self.output_norm(output)

        if return_attention:
            return output, attention_weights

        return output


if __name__ == "__main__":
    torch.manual_seed(42)

    model = TDM(
        embed_dim=512,
        window_size=8,
        stride=4,
        num_heads=8,
        dropout=0.1,
    )

    dummy_input = torch.randn(2, 64, 512)

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
