from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn

from models.tsm import TemporalShift


class Bottleneck1D(nn.Module):
    """ResNet bottleneck block for one-dimensional time-series features.

    The block follows the ResNet-50 bottleneck structure:

        1x1 Conv -> 3x1 Conv -> 1x1 Conv -> Residual addition

    Input and output use PyTorch Conv1d format: (B, C, T).
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        base_channels: int,
        stride: int = 1,
        use_tsm: bool = False,
        tsm_fold_div: int = 4,
    ) -> None:
        super().__init__()

        out_channels = base_channels * self.expansion

        self.use_tsm = use_tsm
        self.temporal_shift = (
            TemporalShift(fold_div=tsm_fold_div) if use_tsm else nn.Identity()
        )

        self.conv1 = nn.Conv1d(
            in_channels,
            base_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)

        self.conv2 = nn.Conv1d(
            base_channels,
            base_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(base_channels)

        self.conv3 = nn.Conv1d(
            base_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def _apply_temporal_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Apply TSM while preserving Conv1d input convention."""

        if not self.use_tsm:
            return x

        # Conv1d format: (B, C, T)
        # TSM format:    (B, T, C)
        x = x.transpose(1, 2)
        x = self.temporal_shift(x)
        x = x.transpose(1, 2)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self._apply_temporal_shift(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        identity = self.downsample(identity)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet1DBackbone(nn.Module):
    """One-dimensional ResNet-50 backbone for multimodal time series.

    External input:
        (B, T, C_in)

    Returned stage features:
        stem:   (B, T, 64)
        stage1: (B, T, 256)
        stage2: (B, T/2, 512)
        stage3: (B, T/4, 1024)
        stage4: (B, T/8, 2048)

    Stage outputs are returned in (B, T, C) format so that subsequent
    temporal modules can consume them consistently.
    """

    def __init__(
        self,
        input_channels: int = 6,
        stage_blocks: List[int] | None = None,
        use_tsm: bool = True,
        tsm_fold_div: int = 4,
    ) -> None:
        super().__init__()

        if stage_blocks is None:
            stage_blocks = [3, 4, 6, 3]

        if len(stage_blocks) != 4:
            raise ValueError(
                "stage_blocks must contain four integers, "
                f"but received {stage_blocks}."
            )

        if input_channels <= 0:
            raise ValueError("input_channels must be greater than zero.")

        self.in_channels = 64
        self.use_tsm = use_tsm
        self.tsm_fold_div = tsm_fold_div

        # No temporal downsampling in the stem.
        # This keeps UCI-HAR length 128 until the first residual stage.
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                64,
                kernel_size=7,
                stride=1,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(
            base_channels=64,
            num_blocks=stage_blocks[0],
            stride=1,
        )
        self.stage2 = self._make_stage(
            base_channels=128,
            num_blocks=stage_blocks[1],
            stride=2,
        )
        self.stage3 = self._make_stage(
            base_channels=256,
            num_blocks=stage_blocks[2],
            stride=2,
        )
        self.stage4 = self._make_stage(
            base_channels=512,
            num_blocks=stage_blocks[3],
            stride=2,
        )

        self._initialize_weights()

    def _make_stage(
        self,
        base_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be greater than zero.")

        blocks = OrderedDict()

        blocks["block1"] = Bottleneck1D(
            in_channels=self.in_channels,
            base_channels=base_channels,
            stride=stride,
            use_tsm=self.use_tsm,
            tsm_fold_div=self.tsm_fold_div,
        )

        self.in_channels = base_channels * Bottleneck1D.expansion

        for block_index in range(1, num_blocks):
            blocks[f"block{block_index + 1}"] = Bottleneck1D(
                in_channels=self.in_channels,
                base_channels=base_channels,
                stride=1,
                use_tsm=self.use_tsm,
                tsm_fold_div=self.tsm_fold_div,
            )

        return nn.Sequential(blocks)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def _to_time_first(x: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, T) to (B, T, C)."""

        return x.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError(
                "Expected input shape (B, T, C), "
                f"but received {tuple(x.shape)}."
            )

        # External format: (B, T, C)
        # Conv1d format: (B, C, T)
        x = x.transpose(1, 2)

        stem = self.stem(x)
        stage1 = self.stage1(stem)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)

        return {
            "stem": self._to_time_first(stem),
            "stage1": self._to_time_first(stage1),
            "stage2": self._to_time_first(stage2),
            "stage3": self._to_time_first(stage3),
            "stage4": self._to_time_first(stage4),
        }


def resnet50_1d(
    input_channels: int = 6,
    use_tsm: bool = True,
    tsm_fold_div: int = 4,
) -> ResNet1DBackbone:
    """Create a ResNet-50-style one-dimensional backbone."""

    return ResNet1DBackbone(
        input_channels=input_channels,
        stage_blocks=[3, 4, 6, 3],
        use_tsm=use_tsm,
        tsm_fold_div=tsm_fold_div,
    )


if __name__ == "__main__":
    model = resnet50_1d(
        input_channels=6,
        use_tsm=True,
    )

    dummy_input = torch.randn(2, 128, 6)

    model.eval()
    with torch.no_grad():
        features = model(dummy_input)

    print("Input:  ", tuple(dummy_input.shape))

    for name, feature in features.items():
        print(f"{name:7s}: {tuple(feature.shape)}")

    total_parameters = sum(
        parameter.numel() for parameter in model.parameters()
    )
    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Total parameters:     {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
