"""Minimal ConvNeXt-Tiny baseline for DKA comparison.

Implements ConvNeXt from scratch (no timm dependency) following the
"A ConvNet for the 2020s" design:
- ConvNeXt block: depthwise conv -> LayerNorm -> 1x1 expand (4x) -> GELU -> 1x1 project
- 4 stages with downsampling via strided convolution
- Adapted for CIFAR-10 input size (no aggressive initial downsampling)
- ~8M params target

Reference: DKA Build Guide, Sections 7.1 and 10.7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for channels-first tensors (B, C, H, W)."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> normalize over C dimension
        mean = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return x


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block.

    depthwise conv (7x7) -> permute -> LayerNorm -> linear expand (4x) ->
    GELU -> linear project -> permute -> residual

    Uses the "inverted bottleneck" pattern: expand channels 4x in the
    middle of the block with pointwise (1x1) convolutions.

    Args:
        dim: Number of input/output channels.
        kernel_size: Depthwise conv kernel size (default 7).
        expansion: Channel expansion factor in the bottleneck (default 4).
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 7,
        expansion: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        hidden_dim = dim * expansion

        # Depthwise convolution
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=dim, bias=True
        )
        # LayerNorm on channels (applied after permute to channels-last)
        self.norm = nn.LayerNorm(dim)
        # Pointwise expand
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        # Pointwise project back
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        # Layer scale: learnable per-channel scaling initialized to small value
        self.gamma = nn.Parameter(1e-6 * torch.ones(dim))

        self.drop_path_rate = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob
        return x * mask / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        residual = x

        # Depthwise conv
        x = self.dwconv(x)

        # Channels-last for LayerNorm and linear layers
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = F.gelu(x)
        x = self.pwconv2(x)
        x = x * self.gamma  # layer scale

        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        x = residual + self._drop_path(x)
        return x


class ConvNeXtStage(nn.Module):
    """One ConvNeXt stage: optional downsampling + N ConvNeXt blocks.

    Args:
        in_dim: Input channels.
        out_dim: Output channels.
        depth: Number of ConvNeXt blocks.
        kernel_size: Depthwise conv kernel size.
        downsample: Whether to downsample at the start of this stage.
        drop_path_rates: List of stochastic depth rates for each block.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        depth: int,
        kernel_size: int = 7,
        downsample: bool = True,
        drop_path_rates: list = None,
    ):
        super().__init__()
        if drop_path_rates is None:
            drop_path_rates = [0.0] * depth

        # Downsampling: LayerNorm + strided conv
        if downsample:
            self.downsample = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
            )
        else:
            # First stage: just match channels if needed
            if in_dim != out_dim:
                self.downsample = nn.Sequential(
                    LayerNorm2d(in_dim),
                    nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1),
                )
            else:
                self.downsample = nn.Identity()

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(
                dim=out_dim,
                kernel_size=kernel_size,
                drop_path=drop_path_rates[i],
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        x = self.blocks(x)
        return x


class ConvNeXt(nn.Module):
    """Minimal ConvNeXt model adapted for small input sizes (CIFAR-10).

    Architecture:
    - Stem: small-kernel patchify (4x4 conv for ImageNet, 2x2 or 1x1 for CIFAR)
    - 4 stages with ConvNeXt blocks
    - Global average pool -> LayerNorm -> Linear

    For CIFAR-10 (32x32), we avoid aggressive downsampling:
    - Stem uses 2x2 conv with stride 2 (halves to 16x16)
    - Stage 1: no downsample (16x16)
    - Stage 2: downsample to 8x8
    - Stage 3: downsample to 4x4
    - Stage 4: downsample to 2x2
    - GAP -> 1x1

    Args:
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        depths: Number of blocks per stage.
        dims: Channel dimensions per stage.
        stem_kernel: Stem convolution kernel size.
        stem_stride: Stem convolution stride.
        block_kernel_size: Depthwise conv kernel size in ConvNeXt blocks.
        drop_path: Maximum stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 10,
        depths: list = None,
        dims: list = None,
        stem_kernel: int = 2,
        stem_stride: int = 2,
        block_kernel_size: int = 7,
        drop_path: float = 0.1,
    ):
        super().__init__()
        if depths is None:
            depths = [2, 2, 6, 2]
        if dims is None:
            dims = [64, 128, 256, 512]

        # Total number of blocks for stochastic depth scheduling
        total_blocks = sum(depths)
        dp_rates = [drop_path * i / max(total_blocks - 1, 1) for i in range(total_blocks)]

        # Stem: patchify with small kernel for CIFAR
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=stem_kernel, stride=stem_stride),
            LayerNorm2d(dims[0]),
        )

        # Build stages
        self.stages = nn.ModuleList()
        cursor = 0
        for i in range(4):
            in_dim = dims[i - 1] if i > 0 else dims[0]
            out_dim = dims[i]
            stage_dp_rates = dp_rates[cursor:cursor + depths[i]]
            cursor += depths[i]

            self.stages.append(ConvNeXtStage(
                in_dim=in_dim,
                out_dim=out_dim,
                depth=depths[i],
                kernel_size=block_kernel_size,
                downsample=(i > 0),  # Don't downsample in stage 1
                drop_path_rates=stage_dp_rates,
            ))

        # Head: global average pool -> LayerNorm -> Linear
        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following ConvNeXt conventions."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.
        Returns:
            (B, num_classes) logits.
        """
        x = self.stem(x)

        for stage in self.stages:
            x = stage(x)

        # Global average pool -> (B, C)
        x = x.mean(dim=[-2, -1])
        x = self.norm(x)
        x = self.head(x)
        return x


def convnext_tiny_cifar10(num_classes: int = 10, **kwargs) -> ConvNeXt:
    """ConvNeXt-Tiny configured for CIFAR-10 (32x32 input), ~8M params.

    Channel dims [64, 128, 256, 512] with depths [2, 2, 6, 2] and
    CIFAR-adapted stem (2x2 conv, stride 2).
    """
    defaults = dict(
        in_channels=3,
        num_classes=num_classes,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        stem_kernel=2,
        stem_stride=2,
        block_kernel_size=7,
        drop_path=0.1,
    )
    defaults.update(kwargs)
    return ConvNeXt(**defaults)


def convnext_tiny_tinyimagenet(**kwargs) -> ConvNeXt:
    """ConvNeXt-Tiny configured for Tiny ImageNet (64x64, 200 classes).

    Same architecture but with slightly larger stem (4x4 conv, stride 4)
    to handle the larger input.
    """
    defaults = dict(
        in_channels=3,
        num_classes=200,
        depths=[2, 2, 6, 2],
        dims=[64, 128, 256, 512],
        stem_kernel=4,
        stem_stride=4,
        block_kernel_size=7,
        drop_path=0.1,
    )
    defaults.update(kwargs)
    return ConvNeXt(**defaults)
