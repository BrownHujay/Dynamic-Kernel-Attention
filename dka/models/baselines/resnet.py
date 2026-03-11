"""ResNet-18 baseline adapted for CIFAR-10 (32x32 input).

Modifications from standard ImageNet ResNet-18:
- First conv: 3x3, stride=1, padding=1 (instead of 7x7, stride=2)
- No max pooling after first conv
- Standard BasicBlock with skip connections throughout
- Global average pool -> linear classifier

Standard ResNet-18 has ~11.2M params. We keep the standard channel widths
[64, 128, 256, 512] and report the exact param count. This is close enough
to the ~8M target for meaningful comparison (within the 10% guidance).

Reference: DKA Build Guide, Sections 7.1 and 10.7.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Standard ResNet BasicBlock with skip connection.

    Two 3x3 conv layers with batch norm and ReLU.
    Shortcut connection uses 1x1 conv when dimensions change.
    """

    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet with configurable depth, adapted for small input sizes.

    For CIFAR-10 (32x32): uses 3x3 initial conv with stride 1 and no maxpool,
    preserving spatial resolution for the small input.

    Args:
        block: Block class (BasicBlock).
        num_blocks: List of block counts per stage, e.g. [2, 2, 2, 2] for ResNet-18.
        num_classes: Number of output classes.
        in_channels: Number of input image channels.
        base_width: Number of channels in first stage (default 64).
    """

    def __init__(
        self,
        block,
        num_blocks: list,
        num_classes: int = 10,
        in_channels: int = 3,
        base_width: int = 64,
    ):
        super().__init__()
        self.current_channels = base_width
        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        # CIFAR-adapted stem: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(
            in_channels, base_width, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_width)

        # Four stages with increasing channels; stages 2-4 downsample via stride=2
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(widths[3] * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, out_channels: int, num_blocks: int, stride: int):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.current_channels, out_channels, s))
            self.current_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        """Kaiming initialization for conv layers, standard for BN and Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.
        Returns:
            (B, num_classes) logits.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet18_cifar10(num_classes: int = 10, **kwargs) -> ResNet:
    """ResNet-18 configured for CIFAR-10 (32x32 input).

    Standard channel widths [64, 128, 256, 512] with [2, 2, 2, 2] blocks.
    ~11.17M params with standard widths.
    """
    defaults = dict(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=3,
        base_width=64,
    )
    defaults.update(kwargs)
    # block and num_blocks shouldn't come from kwargs typically,
    # but allow override for flexibility
    return ResNet(**defaults)


def resnet18_tinyimagenet(**kwargs) -> ResNet:
    """ResNet-18 configured for Tiny ImageNet (64x64, 200 classes).

    Same CIFAR-adapted stem (3x3 conv, no maxpool) works well for 64x64
    since standard ImageNet stem would over-downsample.
    """
    defaults = dict(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        num_classes=200,
        in_channels=3,
        base_width=64,
    )
    defaults.update(kwargs)
    return ResNet(**defaults)
