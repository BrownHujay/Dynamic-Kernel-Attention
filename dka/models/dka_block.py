"""Full DKA Transformer Block.

Pre-norm formulation with Dynamic Kernel Attention and feed-forward network:
    Z = X + DropPath(DKA(LayerNorm(X)))
    Y = Z + DropPath(FFN(LayerNorm(Z)))

FFN uses GELU activation with 4x expansion:
    FFN(x) = W2 * GELU(W1 * x + b1) + b2

Reference: DKA Build Guide, Section 3.4.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dka_module import DKAModule


class DropPath(nn.Module):
    """Stochastic depth (drop path) for regularization.

    During training, randomly drops entire residual branches with probability
    `drop_prob`. At test time, all branches are kept (identity). Surviving
    samples are scaled by 1/(1-p) to maintain expected values.

    Args:
        drop_prob: Probability of dropping the path. 0.0 means no drop.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Shape: (B, 1, 1, ...) — drop entire samples, not individual elements
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        # Scale surviving paths to preserve expected value
        return x * random_tensor / keep_prob

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.3f}"


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation.

    FFN(x) = W2 * GELU(W1 * x + b1) + b2
    W1 in R^(d x 4d), W2 in R^(4d x d).

    Args:
        d_model: Model embedding dimension.
        expansion: FFN expansion factor (default 4).
        dropout: Dropout probability after the second linear layer.
    """

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden_dim = d_model * expansion
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n, d).

        Returns:
            Output tensor of shape (B, n, d).
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DKABlock(nn.Module):
    """Full DKA Transformer Block with pre-norm formulation.

    Architecture:
        Z = X + DropPath(Dropout(DKA(LayerNorm(X))))
        Y = Z + DropPath(FFN(LayerNorm(Z)))

    Supports causal masking for autoregressive language modeling.

    Args:
        d_model: Model embedding dimension.
        num_heads: Number of attention heads (H).
        kernel_sizes: List of kernel sizes, one per head. Length must equal num_heads.
        rank: Rank of factored kernel decomposition (R).
        dropout: Dropout probability after DKA output and FFN output.
        drop_path: Stochastic depth drop probability for this block.
        ffn_expansion: FFN hidden dimension multiplier (default 4).
        causal: If True, apply causal masking in DKA (for language modeling).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        kernel_sizes: list[int],
        rank: int = 4,
        dropout: float = 0.1,
        drop_path: float = 0.0,
        ffn_expansion: int = 4,
        causal: bool = False,
    ):
        super().__init__()

        # Pre-norm layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # DKA attention module
        self.dka = DKAModule(
            d_model=d_model,
            num_heads=num_heads,
            kernel_sizes=kernel_sizes,
            rank=rank,
            causal=causal,
        )

        # Dropout after DKA output (before residual addition)
        self.dka_dropout = nn.Dropout(dropout)

        # Feed-forward network (includes its own dropout)
        self.ffn = FeedForward(
            d_model=d_model,
            expansion=ffn_expansion,
            dropout=dropout,
        )

        # Stochastic depth (drop path) for both sublayers
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n, d).

        Returns:
            Output tensor of shape (B, n, d).
        """
        # Sublayer 1: DKA with pre-norm and residual
        z = x + self.drop_path(self.dka_dropout(self.dka(self.norm1(x))))

        # Sublayer 2: FFN with pre-norm and residual
        y = z + self.drop_path(self.ffn(self.norm2(z)))

        return y
