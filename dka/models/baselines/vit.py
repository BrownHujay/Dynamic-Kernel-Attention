"""Minimal Vision Transformer (ViT-Small) baseline for DKA comparison.

Implements a from-scratch ViT with no external dependencies beyond PyTorch.
Supports both image classification (patch embedding) and text classification
(token embedding) modes.

Design choices for fair comparison with DKA:
- Pre-norm (LayerNorm before each sublayer)
- No CLS token; uses average pooling over tokens for classification
- Learned positional embeddings
- GELU activation in FFN
- Configurable d, H, L, patch_size, num_classes, image_size

Default config matches DKA-Small: d=256, H=8, L=8, ~8M params.

Reference: DKA Build Guide, Sections 7.1 and 10.7.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with pre-norm applied externally."""

    def __init__(self, d: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d % num_heads == 0, f"d ({d}) must be divisible by num_heads ({num_heads})"
        self.d = d
        self.num_heads = num_heads
        self.d_h = d // num_heads
        self.scale = self.d_h ** -0.5

        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n, d)
        Returns:
            (B, n, d)
        """
        B, n, d = x.shape
        qkv = self.qkv(x).reshape(B, n, 3, self.num_heads, self.d_h)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, n, d_h)
        q, k, v = qkv.unbind(0)  # each: (B, H, n, d_h)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, n, n)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, n, d)  # (B, n, d)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class FFN(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, d: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = d * expansion
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Pre-norm transformer block: LN -> MHSA -> residual, LN -> FFN -> residual."""

    def __init__(self, d: int, num_heads: int, dropout: float = 0.0, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = MultiHeadSelfAttention(d, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(d)
        self.ffn = FFN(d, expansion=4, dropout=dropout)
        self.drop_path_rate = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        """Stochastic depth: randomly drop entire residual branch during training."""
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep_prob = 1.0 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.rand(shape, device=x.device, dtype=x.dtype) < keep_prob
        return x * mask / keep_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self._drop_path(self.attn(self.norm1(x)))
        x = x + self._drop_path(self.ffn(self.norm2(x)))
        return x


class PatchEmbedding(nn.Module):
    """Split image into non-overlapping patches and project to d dimensions."""

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d: int):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"
        )
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_size = patch_size
        # Conv2d with kernel_size=stride=patch_size is equivalent to
        # flatten + linear but more efficient
        self.proj = nn.Conv2d(
            in_channels, d, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, num_patches, d)
        """
        x = self.proj(x)  # (B, d, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d)
        return x


class ViT(nn.Module):
    """Minimal Vision Transformer for image classification.

    Uses average pooling over tokens (no CLS token) for fair comparison
    with DKA which also uses AvgPool.

    Args:
        image_size: Input image spatial dimension (assumes square).
        patch_size: Patch size p; sequence length = (image_size/p)^2.
        in_channels: Number of input channels (3 for RGB).
        num_classes: Number of output classes.
        d: Model embedding dimension.
        num_heads: Number of attention heads (H).
        num_layers: Number of transformer blocks (L).
        dropout: Dropout rate for attention and FFN.
        drop_path: Maximum stochastic depth rate (linearly increases per layer).
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        d: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.d = d

        # Patch embedding
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d)
        num_patches = self.patch_embed.num_patches

        # Learned positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d))

        self.pos_drop = nn.Dropout(dropout)

        # Stochastic depth rates linearly increase from 0 to drop_path
        dp_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d, num_heads, dropout=dropout, drop_path=dp_rates[i])
            for i in range(num_layers)
        ])

        # Final norm + classification head
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard ViT conventions."""
        # Positional embeddings: Normal(0, 0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        # Patch projection: Normal(0, 0.02) for weight
        w = self.patch_embed.proj.weight
        nn.init.normal_(w.view(w.size(0), -1), std=0.02)
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)

        # Head
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Transformer blocks
        for block in self.blocks:
            # Linear layers: Kaiming uniform (default) is fine,
            # but ViT convention uses Xavier/truncated normal.
            # We use normal(0, 0.02) for QKV and projection layers.
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.
        Returns:
            (B, num_classes) logits.
        """
        x = self.patch_embed(x)         # (B, n, d)
        x = x + self.pos_embed          # add positional embeddings
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)               # global average pool over tokens
        x = self.head(x)                 # (B, num_classes)
        return x


class ViTForTextClassification(nn.Module):
    """ViT backbone adapted for text classification.

    Replaces patch embedding with a token embedding + positional embedding.
    Same transformer backbone, same AvgPool + Linear head.

    Args:
        vocab_size: Size of token vocabulary.
        max_seq_len: Maximum sequence length.
        num_classes: Number of output classes.
        d: Model embedding dimension.
        num_heads: Number of attention heads (H).
        num_layers: Number of transformer blocks (L).
        dropout: Dropout rate.
        drop_path: Maximum stochastic depth rate.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        max_seq_len: int = 128,
        num_classes: int = 4,
        d: int = 256,
        num_heads: int = 8,
        num_layers: int = 8,
        dropout: float = 0.1,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.d = d

        # Token embedding + positional embedding
        self.token_embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, d))

        self.pos_drop = nn.Dropout(dropout)

        dp_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]

        self.blocks = nn.ModuleList([
            TransformerBlock(d, num_heads, dropout=dropout, drop_path=dp_rates[i])
            for i in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        for block in self.blocks:
            for name, module in block.named_modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, n) integer token IDs.
        Returns:
            (B, num_classes) logits.
        """
        B, n = x.shape
        tok = self.token_embed(x)                      # (B, n, d)
        tok = tok + self.pos_embed[:, :n, :]           # add positional embeddings
        tok = self.pos_drop(tok)

        for block in self.blocks:
            tok = block(tok)

        tok = self.norm(tok)
        tok = tok.mean(dim=1)          # global average pool over tokens
        logits = self.head(tok)        # (B, num_classes)
        return logits


def vit_small_cifar10(**kwargs) -> ViT:
    """ViT-Small configured for CIFAR-10 to match DKA-Small (~8M params).

    Uses d=256, H=8 to match DKA-Small head dimensions. L=10 to reach
    ~7.9M params (standard MHSA has fewer params per layer than DKA, so
    more layers are needed to match the ~8M target).
    """
    defaults = dict(
        image_size=32,
        patch_size=4,
        in_channels=3,
        num_classes=10,
        d=256,
        num_heads=8,
        num_layers=10,
        dropout=0.1,
        drop_path=0.1,
    )
    defaults.update(kwargs)
    return ViT(**defaults)


def vit_small_tinyimagenet(**kwargs) -> ViT:
    """ViT-Small configured for Tiny ImageNet (64x64, 200 classes)."""
    defaults = dict(
        image_size=64,
        patch_size=4,
        in_channels=3,
        num_classes=200,
        d=256,
        num_heads=8,
        num_layers=10,
        dropout=0.1,
        drop_path=0.1,
    )
    defaults.update(kwargs)
    return ViT(**defaults)


def vit_small_agnews(**kwargs) -> ViTForTextClassification:
    """ViT-Small configured for AG News text classification."""
    defaults = dict(
        vocab_size=30522,
        max_seq_len=128,
        num_classes=4,
        d=256,
        num_heads=8,
        num_layers=10,
        dropout=0.1,
        drop_path=0.1,
    )
    defaults.update(kwargs)
    return ViTForTextClassification(**defaults)
