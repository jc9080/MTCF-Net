from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Sequence

import torch
from torch import Tensor, nn

from models.tsm import TemporalShift


class Bottleneck1D(nn.Module):
    """
    1D ResNet bottleneck block.

    The block accepts and returns tensors in convolution format:

        (batch_size, channels, time_steps)

    When TSM is enabled, the tensor is temporarily converted to:

        (batch_size, time_steps, channels)
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

        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")

        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        if stride not in (1, 2):
            raise ValueError("stride must be either 1 or 2.")

        out_channels = base_channels * self.expansion

        self.temporal_shift = (
            TemporalShift(fold_div=tsm_fold_div)
            if use_tsm
            else nn.Identity()
        )

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=base_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(base_channels)

        self.conv2 = nn.Conv1d(
            in_channels=base_channels,
            out_channels=base_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(base_channels)

        self.conv3 = nn.Conv1d(
            in_channels=base_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm1d(out_channels)

        self.activation = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def _apply_temporal_shift(self, x: Tensor) -> Tensor:
        """
        Apply TSM using time-major feature format.

        Args:
            x: Tensor with shape (B, C, T).

        Returns:
            Tensor with shape (B, C, T).
        """
        if isinstance(self.temporal_shift, nn.Identity):
            return x

        x = x.transpose(1, 2).contiguous()
        x = self.temporal_shift(x)
        return x.transpose(1, 2).contiguous()

    def forward(self, x: Tensor) -> Tensor:
        """
        Run one bottleneck block.

        Args:
            x: Tensor with shape (B, C_in, T).

        Returns:
            Tensor with shape (B, C_out, T_out).
        """
        identity = self.downsample(x)

        out = self._apply_temporal_shift(x)

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.activation(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + identity
        out = self.activation(out)

        return out


class ResNet1DBackbone(nn.Module):
    """
    ResNet-50-style 1D backbone for multivariate time-series data.

    External input and stage outputs use time-major format:

        (batch_size, time_steps, channels)

    Internal convolution operations use:

        (batch_size, channels, time_steps)

    Stage output dimensions for an input of shape (B, 128, 6):

        stem:   (B, 128, 64)
        stage1: (B, 128, 256)
        stage2: (B, 64, 512)
        stage3: (B, 32, 1024)
        stage4: (B, 16, 2048)
    """

    def __init__(
        self,
        input_channels: int = 6,
        layers: Sequence[int] = (3, 4, 6, 3),
        use_tsm: bool = True,
        tsm_fold_div: int = 4,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")

        if len(layers) != 4:
            raise ValueError("layers must contain exactly four stage depths.")

        if any(depth <= 0 for depth in layers):
            raise ValueError("Every stage depth must be positive.")

        self.input_channels = input_channels
        self.use_tsm = use_tsm
        self.tsm_fold_div = tsm_fold_div
        self.current_channels = 64

        # Kernel size 3 and stride 1 preserve the original sequence length.
        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels=input_channels,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(
            base_channels=64,
            num_blocks=layers[0],
            stride=1,
        )
        self.stage2 = self._make_stage(
            base_channels=128,
            num_blocks=layers[1],
            stride=2,
        )
        self.stage3 = self._make_stage(
            base_channels=256,
            num_blocks=layers[2],
            stride=2,
        )
        self.stage4 = self._make_stage(
            base_channels=512,
            num_blocks=layers[3],
            stride=2,
        )

        self._initialize_weights()

    def _make_stage(
        self,
        base_channels: int,
        num_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """
        Construct one ResNet stage.

        Temporal downsampling is performed by the first block when stride=2.
        """
        blocks: list[nn.Module] = [
            Bottleneck1D(
                in_channels=self.current_channels,
                base_channels=base_channels,
                stride=stride,
                use_tsm=self.use_tsm,
                tsm_fold_div=self.tsm_fold_div,
            )
        ]

        self.current_channels = base_channels * Bottleneck1D.expansion

        for _ in range(1, num_blocks):
            blocks.append(
                Bottleneck1D(
                    in_channels=self.current_channels,
                    base_channels=base_channels,
                    stride=1,
                    use_tsm=self.use_tsm,
                    tsm_fold_div=self.tsm_fold_div,
                )
            )

        return nn.Sequential(*blocks)

    @staticmethod
    def _to_conv_format(x: Tensor) -> Tensor:
        """
        Convert (B, T, C) to (B, C, T).
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor (B, T, C), but received {tuple(x.shape)}."
            )

        return x.transpose(1, 2).contiguous()

    @staticmethod
    def _to_time_format(x: Tensor) -> Tensor:
        """
        Convert (B, C, T) to (B, T, C).
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected a 3D tensor (B, C, T), but received {tuple(x.shape)}."
            )

        return x.transpose(1, 2).contiguous()

    def stem_forward(self, x: Tensor) -> Tensor:
        """
        Run the input stem.

        Args:
            x: Tensor with shape (B, T, input_channels).

        Returns:
            Tensor with shape (B, T, 64).
        """
        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"but received {x.shape[-1]}."
            )

        x = self._to_conv_format(x)
        x = self.stem(x)
        return self._to_time_format(x)

    def stage1_forward(self, x: Tensor) -> Tensor:
        """
        Run ResNet stage 1.

        Input:  (B, T, 64)
        Output: (B, T, 256)
        """
        x = self._to_conv_format(x)
        x = self.stage1(x)
        return self._to_time_format(x)

    def stage2_forward(self, x: Tensor) -> Tensor:
        """
        Run ResNet stage 2.

        Input:  (B, T, 256)
        Output: (B, ceil(T / 2), 512)
        """
        x = self._to_conv_format(x)
        x = self.stage2(x)
        return self._to_time_format(x)

    def stage3_forward(self, x: Tensor) -> Tensor:
        """
        Run ResNet stage 3.

        Input:  (B, T, 512)
        Output: (B, ceil(T / 2), 1024)
        """
        x = self._to_conv_format(x)
        x = self.stage3(x)
        return self._to_time_format(x)

    def stage4_forward(self, x: Tensor) -> Tensor:
        """
        Run ResNet stage 4.

        Input:  (B, T, 1024)
        Output: (B, ceil(T / 2), 2048)
        """
        x = self._to_conv_format(x)
        x = self.stage4(x)
        return self._to_time_format(x)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Run the complete backbone and return intermediate feature maps.

        This method remains useful for standalone feature extraction and shape
        verification. MTCF-Net can instead call each stage-forward method
        separately to insert MCFI and TDM between ResNet stages.
        """
        stem = self.stem_forward(x)
        stage1 = self.stage1_forward(stem)
        stage2 = self.stage2_forward(stage1)
        stage3 = self.stage3_forward(stage2)
        stage4 = self.stage4_forward(stage3)

        return OrderedDict(
            stem=stem,
            stage1=stage1,
            stage2=stage2,
            stage3=stage3,
            stage4=stage4,
        )

    def _initialize_weights(self) -> None:
        """
        Initialize convolution and normalization layers.
        """
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


def resnet50_1d(
    input_channels: int = 6,
    use_tsm: bool = True,
    tsm_fold_div: int = 4,
) -> ResNet1DBackbone:
    """
    Build a ResNet-50-style 1D backbone.

    Stage depths follow the standard ResNet-50 configuration:

        [3, 4, 6, 3]
    """
    return ResNet1DBackbone(
        input_channels=input_channels,
        layers=(3, 4, 6, 3),
        use_tsm=use_tsm,
        tsm_fold_div=tsm_fold_div,
    )


def _run_shape_test() -> None:
    """
    Run a standalone backbone shape test.
    """
    torch.manual_seed(42)

    model = resnet50_1d(
        input_channels=6,
        use_tsm=True,
    )
    model.eval()

    dummy = torch.randn(2, 128, 6)

    with torch.no_grad():
        features = model(dummy)

    print(f"Input:   {tuple(dummy.shape)}")

    for name, feature in features.items():
        print(f"{name:8s}: {tuple(feature.shape)}")

    expected_shapes = {
        "stem": (2, 128, 64),
        "stage1": (2, 128, 256),
        "stage2": (2, 64, 512),
        "stage3": (2, 32, 1024),
        "stage4": (2, 16, 2048),
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(features[name].shape)

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"expected {expected_shape}, received {actual_shape}."
            )

    print("\nAll backbone shape tests passed.")


if __name__ == "__main__":
    _run_shape_test()
