"""Full DKA Model architectures for image classification, text classification,
and language modeling.

Provides:
- DKAImageModel: Patch embedding + DKA blocks + classification head (no CLS token)
- DKATextModel: Token embedding + DKA blocks + classification or LM head

Factory class methods for standard configurations:
- DKA-Tiny:  d=128, H=4,  L=6,  R=2, ~1.5M params
- DKA-Small: d=256, H=8,  L=8,  R=4, ~8M params (PRIMARY)
- DKA-Base:  d=384, H=12, L=12, R=4, ~25M params

Reference: DKA Build Guide, Sections 4.1, 4.2, and 5.5.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from .dka_block import DKABlock


# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

# Image kernel size assignments per config
IMAGE_KERNEL_SIZES = {
    "tiny":  [3, 5, 7, 11],                                    # H=4
    "small": [3, 3, 5, 5, 7, 7, 11, 11],                       # H=8
    "base":  [3, 3, 3, 5, 5, 5, 7, 7, 7, 11, 11, 11],         # H=12
}

# Text kernel size assignments per config
TEXT_KERNEL_SIZES = {
    "tiny":  [3, 7, 11, 21],                                   # H=4
    "small": [3, 3, 7, 7, 11, 11, 21, 21],                     # H=8
    "base":  [3, 3, 3, 7, 7, 7, 11, 11, 11, 21, 21, 21],      # H=12
}

MODEL_CONFIGS = {
    "tiny":  {"d_model": 128, "num_heads": 4,  "num_layers": 6,  "rank": 2},
    "small": {"d_model": 256, "num_heads": 8,  "num_layers": 8,  "rank": 4},
    "base":  {"d_model": 384, "num_heads": 12, "num_layers": 12, "rank": 4},
}


# ---------------------------------------------------------------------------
# Patch Embedding for images
# ---------------------------------------------------------------------------

class PatchEmbedding(nn.Module):
    """Split an image into non-overlapping patches and linearly project.

    Input:  (B, 3, H_img, W_img)
    Output: (B, n, d_model)  where n = (H_img/p) * (W_img/p)

    Args:
        img_size: Spatial size of the input image (assumes square).
        patch_size: Side length of each square patch (p).
        d_model: Model embedding dimension.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int):
        super().__init__()
        assert img_size % patch_size == 0, (
            f"Image size {img_size} must be divisible by patch size {patch_size}"
        )
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 3

        self.proj = nn.Linear(patch_dim, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images of shape (B, 3, H, W).

        Returns:
            Patch embeddings of shape (B, n, d_model).
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, 3, H, W) -> (B, n, p*p*3)
        x = x.reshape(B, C, H // p, p, W // p, p)
        x = x.permute(0, 2, 4, 3, 5, 1)          # (B, H/p, W/p, p, p, 3)
        x = x.reshape(B, self.num_patches, -1)     # (B, n, p*p*3)

        # Linear projection to d_model
        x = self.proj(x)                            # (B, n, d_model)
        return x


# ---------------------------------------------------------------------------
# DKA Image Model
# ---------------------------------------------------------------------------

class DKAImageModel(nn.Module):
    """DKA Transformer for image classification.

    Architecture:
        1. Patch embedding: image -> non-overlapping p x p patches -> linear -> d
        2. Add learned positional embeddings P in R^(n x d)
        3. L stacked DKA Transformer Blocks
        4. Classification head: LayerNorm -> AvgPool across tokens -> Linear(d, num_classes)
        No CLS token.

    Args:
        img_size: Input image spatial size (square).
        patch_size: Patch side length.
        num_classes: Number of output classes.
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of DKA blocks.
        kernel_sizes: List of kernel sizes per head (length == num_heads).
        rank: Factored kernel rank.
        dropout: Dropout probability.
        drop_path_rate: Maximum stochastic depth rate (linearly increasing).
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        num_classes: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        kernel_sizes: list[int],
        rank: int = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, d_model)
        num_patches = self.patch_embed.num_patches

        # Learned positional embeddings P in R^(n x d)
        self.pos_embed = nn.Parameter(torch.empty(1, num_patches, d_model))

        # Stochastic depth: linearly increasing drop rate from 0 to drop_path_rate
        drop_rates = [
            drop_path_rate * i / max(num_layers - 1, 1)
            for i in range(num_layers)
        ]

        # Stacked DKA Transformer Blocks
        self.blocks = nn.ModuleList([
            DKABlock(
                d_model=d_model,
                num_heads=num_heads,
                kernel_sizes=kernel_sizes,
                rank=rank,
                dropout=dropout,
                drop_path=drop_rates[i],
                causal=False,
            )
            for i in range(num_layers)
        ])

        # Classification head: LayerNorm -> AvgPool -> Linear
        self.head_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights per build guide section 5.5."""
        # Patch embeddings: Normal(0, 0.02)
        nn.init.normal_(self.patch_embed.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.patch_embed.proj.bias)

        # Positional embeddings: Normal(0, 0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Output projection scaling by depth: Normal(0, 0.02 / sqrt(2*L))
        scale = 0.02 / math.sqrt(2 * self.num_layers)
        for block in self.blocks:
            if hasattr(block.dka, 'W_out'):
                nn.init.normal_(block.dka.W_out.weight, mean=0.0, std=scale)
            elif hasattr(block.dka, 'out_proj'):
                nn.init.normal_(block.dka.out_proj.weight, mean=0.0, std=scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images of shape (B, 3, H, W).

        Returns:
            Logits of shape (B, num_classes).
        """
        # Patch embed + positional embedding
        x = self.patch_embed(x)       # (B, n, d)
        x = x + self.pos_embed        # (B, n, d)

        # DKA Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification head: LayerNorm -> AvgPool -> Linear
        x = self.head_norm(x)         # (B, n, d)
        x = x.mean(dim=1)            # (B, d)  — average pool across tokens
        x = self.head(x)             # (B, num_classes)

        return x

    @classmethod
    def from_config(
        cls,
        config_name: str,
        img_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
    ) -> "DKAImageModel":
        """Create a DKAImageModel from a named configuration.

        Args:
            config_name: One of 'tiny', 'small', 'base'.
            img_size: Input image size (default 32 for CIFAR-10).
            patch_size: Patch size (default 4).
            num_classes: Number of output classes (default 10 for CIFAR-10).
            dropout: Dropout rate.
            drop_path_rate: Max stochastic depth rate.

        Returns:
            Configured DKAImageModel instance.
        """
        cfg = MODEL_CONFIGS[config_name]
        kernel_sizes = IMAGE_KERNEL_SIZES[config_name]

        return cls(
            img_size=img_size,
            patch_size=patch_size,
            num_classes=num_classes,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            kernel_sizes=kernel_sizes,
            rank=cfg["rank"],
            dropout=dropout,
            drop_path_rate=drop_path_rate,
        )


# ---------------------------------------------------------------------------
# DKA Text Model
# ---------------------------------------------------------------------------

class DKATextModel(nn.Module):
    """DKA Transformer for text classification and language modeling.

    Architecture:
        1. Token embedding table E in R^(V x d)
        2. Learned positional embeddings P in R^(n_max x d)
        3. L stacked DKA Transformer Blocks
        4. Task-specific head:
           - Classification (e.g. AG News): LayerNorm -> AvgPool -> Linear(d, num_classes)
           - Language modeling (e.g. WikiText-2): LayerNorm -> Linear(d, vocab_size) per position

    Args:
        vocab_size: Size of the token vocabulary (V).
        max_seq_len: Maximum sequence length (n_max).
        d_model: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of DKA blocks.
        kernel_sizes: List of kernel sizes per head.
        rank: Factored kernel rank.
        num_classes: Number of output classes for classification. If None, use LM head.
        dropout: Dropout probability.
        drop_path_rate: Maximum stochastic depth rate.
        causal: Whether to use causal masking (True for LM, False for classification).
    """

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        kernel_sizes: list[int],
        rank: int = 4,
        num_classes: Optional[int] = None,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.causal = causal
        self.is_lm = num_classes is None

        # Token embedding table E in R^(V x d)
        self.token_embed = nn.Embedding(vocab_size, d_model)

        # Learned positional embeddings P in R^(n_max x d)
        self.pos_embed = nn.Parameter(torch.empty(1, max_seq_len, d_model))

        # Embedding dropout
        self.embed_dropout = nn.Dropout(dropout)

        # Stochastic depth: linearly increasing drop rate
        drop_rates = [
            drop_path_rate * i / max(num_layers - 1, 1)
            for i in range(num_layers)
        ]

        # Stacked DKA Transformer Blocks
        # For LM tasks, use causal masking
        use_causal = causal or self.is_lm
        self.blocks = nn.ModuleList([
            DKABlock(
                d_model=d_model,
                num_heads=num_heads,
                kernel_sizes=kernel_sizes,
                rank=rank,
                dropout=dropout,
                drop_path=drop_rates[i],
                causal=use_causal,
            )
            for i in range(num_layers)
        ])

        # Task-specific head
        self.head_norm = nn.LayerNorm(d_model)

        if self.is_lm:
            # Language modeling head: Linear(d, V) at each position
            self.lm_head = nn.Linear(d_model, vocab_size)
        else:
            # Classification head: Linear(d, num_classes) after AvgPool
            assert num_classes is not None
            self.cls_head = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights per build guide section 5.5."""
        # Token embeddings: Normal(0, 0.02)
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)

        # Positional embeddings: Normal(0, 0.02)
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

        # Output projection scaling by depth: Normal(0, 0.02 / sqrt(2*L))
        scale = 0.02 / math.sqrt(2 * self.num_layers)
        for block in self.blocks:
            if hasattr(block.dka, 'W_out'):
                nn.init.normal_(block.dka.W_out.weight, mean=0.0, std=scale)
            elif hasattr(block.dka, 'out_proj'):
                nn.init.normal_(block.dka.out_proj.weight, mean=0.0, std=scale)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs of shape (B, n).
            attention_mask: Optional mask of shape (B, n), 1 for real tokens,
                0 for padding. Used for classification pooling.

        Returns:
            - Classification: logits of shape (B, num_classes).
            - Language modeling: logits of shape (B, n, vocab_size).
        """
        B, n = input_ids.shape

        # Token embedding + positional embedding
        x = self.token_embed(input_ids)            # (B, n, d)
        x = x + self.pos_embed[:, :n, :]           # (B, n, d)
        x = self.embed_dropout(x)

        # DKA Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Head
        x = self.head_norm(x)                      # (B, n, d)

        if self.is_lm:
            # Language modeling: predict next token at each position
            logits = self.lm_head(x)               # (B, n, V)
            return logits
        else:
            # Classification: AvgPool across tokens -> Linear
            if attention_mask is not None:
                # Mask out padding tokens before averaging
                mask = attention_mask.unsqueeze(-1).float()  # (B, n, 1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            else:
                x = x.mean(dim=1)                  # (B, d)

            logits = self.cls_head(x)              # (B, num_classes)
            return logits

    @classmethod
    def from_config(
        cls,
        config_name: str,
        vocab_size: int,
        max_seq_len: int = 128,
        num_classes: Optional[int] = 4,
        dropout: float = 0.1,
        drop_path_rate: float = 0.1,
        causal: bool = False,
    ) -> "DKATextModel":
        """Create a DKATextModel from a named configuration.

        Args:
            config_name: One of 'tiny', 'small', 'base'.
            vocab_size: Vocabulary size (V).
            max_seq_len: Maximum sequence length (default 128 for AG News).
            num_classes: Number of classes for classification. None for LM.
            dropout: Dropout rate.
            drop_path_rate: Max stochastic depth rate.
            causal: Whether to use causal masking.

        Returns:
            Configured DKATextModel instance.
        """
        cfg = MODEL_CONFIGS[config_name]
        kernel_sizes = TEXT_KERNEL_SIZES[config_name]

        # For LM tasks, enable causal masking
        if num_classes is None:
            causal = True

        return cls(
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            d_model=cfg["d_model"],
            num_heads=cfg["num_heads"],
            num_layers=cfg["num_layers"],
            kernel_sizes=kernel_sizes,
            rank=cfg["rank"],
            num_classes=num_classes,
            dropout=dropout,
            drop_path_rate=drop_path_rate,
            causal=causal,
        )
