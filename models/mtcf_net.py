from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import Tensor, nn

from models.adf import AdaptiveFeatureFusion
from models.mcfi import MCFI
from models.resnet_backbone import ResNet1DBackbone, resnet50_1d
from models.tdm import TDM


class MTCFNet(nn.Module):
    """
    MTCF-Net for multivariate time-series classification.

    Model flow:

        Input
          ↓
        ResNet stem
          ↓
        ResNet stage 1
          ↓
        MCFI
          ↓
        ResNet stage 2
          ↓
        TDM
          ↓
        ResNet stage 3
          ↓
        ResNet stage 4
          ↓
        ADF
          ↓
        Linear classifier

    Default UCI-HAR input:

        Input:  (B, 128, 6)
        Output: (B, 6)
    """

    def __init__(
        self,
        input_channels: int = 6,
        num_classes: int = 6,
        use_tsm: bool = True,
        tsm_fold_div: int = 4,
        mcfi_num_heads: int = 8,
        tdm_window_size: int = 8,
        tdm_stride: int = 4,
        tdm_num_heads: int = 8,
        adf_num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if input_channels <= 0:
            raise ValueError("input_channels must be positive.")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1.")

        self.input_channels = input_channels
        self.num_classes = num_classes

        self.register_buffer(
            "_positional_encoding_cache",
            torch.empty(0),
            persistent=False,
        )

        self.backbone: ResNet1DBackbone = resnet50_1d(
            input_channels=input_channels,
            use_tsm=use_tsm,
            tsm_fold_div=tsm_fold_div,
        )

        # Stage 1 output dimension: 256
        self.mcfi = MCFI(
            embed_dim=256,
            num_heads=mcfi_num_heads,
            dropout=dropout,
        )

        # Stage 2 output dimension: 512
        self.tdm = TDM(
            embed_dim=512,
            window_size=tdm_window_size,
            stride=tdm_stride,
            num_heads=tdm_num_heads,
            dropout=dropout,
        )

        # Stage 4 output dimension: 2048
        self.adf = AdaptiveFeatureFusion(
            embed_dim=2048,
            num_heads=adf_num_heads,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2048, num_classes),
        )

        self._initialize_classifier()

    def forward_features(
        self,
        x: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        Extract fused features before classification.

        Args:
            x:
                Input tensor with shape (B, T, C).
            return_intermediates:
                If True, also return intermediate feature maps.

        Returns:
            ADF pooled feature with shape (B, 2048), or a tuple containing
            the pooled feature and intermediate feature dictionary.
        """
        self._validate_input(x)

        features: Dict[str, Tensor] = OrderedDict()

        x = x + self._get_temporal_positional_encoding(x)

        x = self.backbone.stem_forward(x)
        features["stem"] = x

        x = self.backbone.stage1_forward(x)
        features["stage1"] = x

        x = self.mcfi(x)
        features["mcfi"] = x

        x = self.backbone.stage2_forward(x)
        features["stage2"] = x

        x = self.tdm(x)
        features["tdm"] = x

        x = self.backbone.stage3_forward(x)
        features["stage3"] = x

        x = self.backbone.stage4_forward(x)
        features["stage4"] = x

        x = self.adf(x)
        features["adf"] = x

        if return_intermediates:
            return x, features

        return x

    def forward(
        self,
        x: Tensor,
        return_intermediates: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        Run a complete forward pass.

        Args:
            x:
                Input tensor with shape (B, T, input_channels).
            return_intermediates:
                If True, return logits and all intermediate features.

        Returns:
            Logits with shape (B, num_classes), or a tuple containing logits
            and intermediate feature maps.
        """
        if return_intermediates:
            pooled, features = self.forward_features(
                x,
                return_intermediates=True,
            )

            logits = self.classifier(pooled)
            features["logits"] = logits

            return logits, features

        pooled = self.forward_features(x)
        logits = self.classifier(pooled)

        return logits

    def _get_temporal_positional_encoding(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Create sinusoidal temporal positional encoding.

        Input:
            x: (B, T, C)

        Returns:
            positional encoding: (1, T, C)
        """
        sequence_length = x.size(1)
        channels = x.size(2)

        cache = self._positional_encoding_cache

        cache_is_valid = (
            cache.numel() > 0
            and cache.size(1) == sequence_length
            and cache.size(2) == channels
            and cache.device == x.device
            and cache.dtype == x.dtype
        )

        if cache_is_valid:
            return cache

        position = torch.arange(
            sequence_length,
            device=x.device,
            dtype=torch.float32,
        ).unsqueeze(1)

        even_indices = torch.arange(
            0,
            channels,
            2,
            device=x.device,
            dtype=torch.float32,
        )

        div_term = torch.exp(
            even_indices
            * (-torch.log(torch.tensor(10000.0, device=x.device))
               / max(channels, 1))
        )

        encoding = torch.zeros(
            1,
            sequence_length,
            channels,
            device=x.device,
            dtype=torch.float32,
        )

        encoding[0, :, 0::2] = torch.sin(position * div_term)

        if channels > 1:
            encoding[0, :, 1::2] = torch.cos(
                position * div_term[: encoding[0, :, 1::2].shape[-1]]
            )

        encoding = encoding.to(dtype=x.dtype)
        self._positional_encoding_cache = encoding

        return encoding

    def _validate_input(self, x: Tensor) -> None:
        """
        Validate model input shape.
        """
        if x.ndim != 3:
            raise ValueError(
                "MTCFNet expects a 3D tensor with shape (B, T, C), "
                f"but received {tuple(x.shape)}."
            )

        if x.shape[-1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, "
                f"but received {x.shape[-1]}."
            )

    def _initialize_classifier(self) -> None:
        """
        Initialize the final classification layer.
        """
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)


def mtcf_net_uci_har(
    num_classes: int = 6,
    dropout: float = 0.1,
) -> MTCFNet:
    """
    Build MTCF-Net configured for the UCI-HAR dataset.
    """
    return MTCFNet(
        input_channels=6,
        num_classes=num_classes,
        use_tsm=True,
        tsm_fold_div=4,
        mcfi_num_heads=8,
        tdm_window_size=8,
        tdm_stride=4,
        tdm_num_heads=8,
        adf_num_heads=8,
        dropout=dropout,
    )


def mtcf_net_pamap2(
    num_classes: int = 18,
    dropout: float = 0.1,
) -> MTCFNet:
    """
    Build MTCF-Net configured for the PAMAP2 dataset.

    Default PAMAP2 input:
        Input:  (B, 256, 28)
        Output: (B, 18)
    """
    return MTCFNet(
        input_channels=28,
        num_classes=num_classes,
        use_tsm=True,
        tsm_fold_div=4,
        mcfi_num_heads=8,
        tdm_window_size=8,
        tdm_stride=4,
        tdm_num_heads=8,
        adf_num_heads=8,
        dropout=dropout,
    )



def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _run_shape_test() -> None:
    """
    Run a standalone MTCF-Net shape test.
    """
    torch.manual_seed(42)

    model = mtcf_net_uci_har()
    model.eval()

    dummy = torch.randn(2, 128, 6)

    with torch.no_grad():
        logits, features = model(
            dummy,
            return_intermediates=True,
        )

    print(f"Input:   {tuple(dummy.shape)}")

    for name, feature in features.items():
        print(f"{name:8s}: {tuple(feature.shape)}")

    expected_shapes = {
        "stem": (2, 128, 64),
        "stage1": (2, 128, 256),
        "mcfi": (2, 128, 256),
        "stage2": (2, 64, 512),
        "tdm": (2, 64, 512),
        "stage3": (2, 32, 1024),
        "stage4": (2, 16, 2048),
        "adf": (2, 2048),
        "logits": (2, 6),
    }

    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(features[name].shape)

        if actual_shape != expected_shape:
            raise RuntimeError(
                f"{name} shape mismatch: "
                f"expected {expected_shape}, received {actual_shape}."
            )

    if tuple(logits.shape) != (2, 6):
        raise RuntimeError(
            f"Logit shape mismatch: expected (2, 6), "
            f"received {tuple(logits.shape)}."
        )

    parameter_count = count_trainable_parameters(model)

    print(f"\nTrainable parameters: {parameter_count:,}")
    print("All MTCF-Net shape tests passed.")


if __name__ == "__main__":
    _run_shape_test()
