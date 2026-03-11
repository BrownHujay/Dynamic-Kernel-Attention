"""Factored Kernel Generation for Dynamic Kernel Attention (DKA).

Generates per-token convolutional kernels via factored rank decomposition.
Each token's kernel is the sum of R rank-1 outer products of spatial and
channel components, produced by lightweight MLPs from the token embedding.

Reference: DKA Build Guide, Sections 3.2 (Steps 2-3) and 5.5.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FactoredKernelGenerator(nn.Module):
    """Generates per-token convolutional kernels via factored rank decomposition.

    For each token, produces a kernel of shape (k_h, d_h) as:
        K_hat = K_base + alpha * (delta_K / (||delta_K||_F + eps))

    where delta_K = sum_{m=1}^{R} s_m (x) c_m^T, with s_m from a spatial MLP
    and c_m from a channel MLP.

    Args:
        d_h: Per-head embedding dimension.
        k_h: Kernel (window) size for this head.
        rank: Number of rank-1 components (R).
        eps: Epsilon for Frobenius norm stability.
    """

    def __init__(self, d_h: int, k_h: int, rank: int = 4, eps: float = 1e-6):
        super().__init__()
        self.d_h = d_h
        self.k_h = k_h
        self.rank = rank
        self.eps = eps

        # --- Rank-component MLPs ---
        # Each rank component m has a spatial MLP and a channel MLP.
        # We store them as parallel linear layers across all ranks for efficiency.

        # Spatial MLP: d_h -> d_h (hidden) -> k_h (output), per rank component
        self.spatial_fc1 = nn.Linear(d_h, d_h * rank)
        self.spatial_fc2 = nn.Linear(d_h * rank, k_h * rank)

        # Channel MLP: d_h -> d_h (hidden) -> d_h (output), per rank component
        self.channel_fc1 = nn.Linear(d_h, d_h * rank)
        self.channel_fc2 = nn.Linear(d_h * rank, d_h * rank)

        # --- Residual kernel structure ---
        # K_base: learned static kernel, shared across all tokens
        self.K_base = nn.Parameter(torch.empty(k_h, d_h))

        # alpha: learned scalar controlling dynamic contribution
        self.alpha = nn.Parameter(torch.tensor(0.01))

        self._init_weights()

    def _init_weights(self):
        """Initialize weights per build guide section 5.5."""
        # First layers: Kaiming uniform (default for nn.Linear, but be explicit)
        nn.init.kaiming_uniform_(self.spatial_fc1.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.channel_fc1.weight, a=math.sqrt(5))

        # Compute fan_in for bias init (matches PyTorch default Linear init)
        fan_in_s1 = self.spatial_fc1.weight.size(1)
        bound_s1 = 1 / math.sqrt(fan_in_s1)
        nn.init.uniform_(self.spatial_fc1.bias, -bound_s1, bound_s1)

        fan_in_c1 = self.channel_fc1.weight.size(1)
        bound_c1 = 1 / math.sqrt(fan_in_c1)
        nn.init.uniform_(self.channel_fc1.bias, -bound_c1, bound_c1)

        # Final layers (W_2s, W_2c): Normal(0, 0.001) so delta_K ~ 0 at init
        nn.init.normal_(self.spatial_fc2.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.spatial_fc2.bias)

        nn.init.normal_(self.channel_fc2.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.channel_fc2.bias)

        # K_base: Kaiming uniform, same as a standard depthwise conv layer.
        # fan_in = k_h (number of spatial positions summed over per channel).
        # bound = 1/sqrt(fan_in) for kaiming_uniform with a=sqrt(5).
        bound = 1.0 / math.sqrt(self.k_h)
        nn.init.uniform_(self.K_base, -bound, bound)

        # alpha: constant 0.01 (already set in Parameter init, but be explicit)
        nn.init.constant_(self.alpha, 0.01)

    def forward(self, x_h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate per-token kernels for one head.

        Args:
            x_h: Head embeddings, shape (B, n, d_h).

        Returns:
            K_hat: Generated kernels, shape (B, n, k_h, d_h).
            alpha: Current alpha_h scalar value.
        """
        B, n, d_h = x_h.shape
        R = self.rank

        # --- Spatial components: (B, n, d_h) -> (B, n, R, k_h) ---
        s = self.spatial_fc1(x_h)           # (B, n, d_h * R)
        s = F.gelu(s)
        s = self.spatial_fc2(s)             # (B, n, k_h * R)
        s = s.view(B, n, R, self.k_h)      # (B, n, R, k_h)

        # --- Channel components: (B, n, d_h) -> (B, n, R, d_h) ---
        c = self.channel_fc1(x_h)           # (B, n, d_h * R)
        c = F.gelu(c)
        c = self.channel_fc2(c)             # (B, n, d_h * R)
        c = c.view(B, n, R, d_h)           # (B, n, R, d_h)

        # --- Outer products summed over ranks ---
        # delta_K = sum_m s_m outer c_m -> (B, n, k_h, d_h)
        # s: (B, n, R, k_h) -> (B, n, R, k_h, 1)
        # c: (B, n, R, d_h) -> (B, n, R, 1, d_h)
        # outer product per rank then sum over R
        delta_K = torch.einsum('bnrk,bnrd->bnkd', s, c)  # (B, n, k_h, d_h)

        # --- Frobenius normalization ---
        # ||delta_K||_F computed per token: (B, n, 1, 1)
        frob_norm = torch.norm(
            delta_K.view(B, n, -1), dim=-1, keepdim=True
        ).unsqueeze(-1)  # (B, n, 1, 1)

        delta_K_normalized = delta_K / (frob_norm + self.eps)

        # --- Residual kernel structure ---
        # K_hat = K_base + alpha * normalized(delta_K)
        K_hat = self.K_base.unsqueeze(0).unsqueeze(0) + self.alpha * delta_K_normalized

        return K_hat, self.alpha
