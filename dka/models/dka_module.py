"""
Core Dynamic Kernel Attention (DKA) module.

Replaces standard multi-head self-attention with per-token dynamic kernel
generation and local window application. See DKA_BUILD_GUIDE.md sections 3.2-3.3
for the full specification.

Input:  X of shape (B, n, d)
Output: O of shape (B, n, d)
"""

import math
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_generator import FactoredKernelGenerator

# Default multi-scale kernel sizes per head (H=8)
IMAGE_KERNEL_SIZES = [3, 3, 5, 5, 7, 7, 11, 11]
TEXT_KERNEL_SIZES = [3, 3, 7, 7, 11, 11, 21, 21]


class DKAModule(nn.Module):
    """
    Dynamic Kernel Attention module -- drop-in replacement for multi-head
    self-attention.

    For each head:
      1. Project input into head subspace.
      2. Generate a per-token convolutional kernel via FactoredKernelGenerator.
      3. Extract local windows with unfold.
      4. Apply kernels to windows via einsum.
    Concatenate heads and apply output projection.

    Args:
        d_model: Model embedding dimension.
        num_heads: Number of attention heads (H).
        rank: Rank of factored kernel decomposition (R).
        kernel_sizes: List of kernel sizes, one per head. Length must equal
            num_heads. If None, uses IMAGE_KERNEL_SIZES for image mode or
            TEXT_KERNEL_SIZES for text mode.
        mode: "image" or "text". Only used when kernel_sizes is None.
        num_layers: Total number of transformer layers in the model. Used for
            output projection initialization scaling.
        dropout: Dropout probability after output projection.
        causal: If True, apply causal masking (zero out future positions in
            the kernel window). Required for autoregressive language modeling.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        rank: int = 4,
        kernel_sizes: Optional[List[int]] = None,
        mode: str = "image",
        num_layers: int = 8,
        dropout: float = 0.1,
        causal: bool = False,
    ):
        super().__init__()

        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.rank = rank
        self.num_layers = num_layers
        self.causal = causal

        # Resolve kernel sizes
        if kernel_sizes is not None:
            assert len(kernel_sizes) == num_heads, (
                f"kernel_sizes length ({len(kernel_sizes)}) must equal "
                f"num_heads ({num_heads})"
            )
            self.kernel_sizes = kernel_sizes
        elif mode == "text":
            assert num_heads == len(TEXT_KERNEL_SIZES), (
                f"Default text kernel sizes require num_heads={len(TEXT_KERNEL_SIZES)}, "
                f"got {num_heads}. Provide kernel_sizes explicitly."
            )
            self.kernel_sizes = TEXT_KERNEL_SIZES
        else:
            assert num_heads == len(IMAGE_KERNEL_SIZES), (
                f"Default image kernel sizes require num_heads={len(IMAGE_KERNEL_SIZES)}, "
                f"got {num_heads}. Provide kernel_sizes explicitly."
            )
            self.kernel_sizes = IMAGE_KERNEL_SIZES

        # --- Step 1: Head input projections ---
        # Each head gets its own linear projection: (d_model) -> (d_head)
        # Implemented as a single fused linear for efficiency, then reshaped.
        self.head_proj_in = nn.Linear(d_model, d_model, bias=True)

        # --- Step 2: Kernel generators (one per head, different kernel sizes) ---
        self.kernel_generators = nn.ModuleList([
            FactoredKernelGenerator(
                d_h=self.d_head,
                k_h=k,
                rank=rank,
            )
            for k in self.kernel_sizes
        ])

        # --- Step 5: Output projection ---
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout = nn.Dropout(dropout)

        # Pre-compute and register causal masks for each unique kernel size
        if self.causal:
            for k in set(self.kernel_sizes):
                mask = self._build_causal_mask(k)
                # Register as buffer so it moves to the right device
                self.register_buffer(f"causal_mask_{k}", mask, persistent=False)

        # --- Expose for logging/visualization ---
        # These are populated during forward and can be read externally.
        self._last_kernels: Optional[Dict[int, torch.Tensor]] = None
        self._last_alphas: Optional[Dict[int, torch.Tensor]] = None

        # --- Initialization ---
        self._init_weights()

    # ------------------------------------------------------------------
    # Initialization (section 5.5)
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        # Head input projection: Kaiming uniform (general linear layer)
        nn.init.kaiming_uniform_(self.head_proj_in.weight, a=math.sqrt(5))
        if self.head_proj_in.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.head_proj_in.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.head_proj_in.bias, -bound, bound)

        # Output projection: Normal(0, 0.02 / sqrt(2 * num_layers))
        std = 0.02 / math.sqrt(2.0 * self.num_layers)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Kernel generator weights are initialized inside FactoredKernelGenerator.

    # ------------------------------------------------------------------
    # Causal mask construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_causal_mask(kernel_size: int) -> torch.Tensor:
        """Build a binary mask that zeros out future positions in the window.

        For a window of size k centered on position i, position j in the
        window corresponds to offset j - floor(k/2) relative to i.  Offsets
        > 0 are future positions and must be masked to 0.

        Returns a 1-D tensor of shape (k,) with 1s for allowed (past/current)
        positions and 0s for future positions.
        """
        half = kernel_size // 2
        # offsets: [-half, ..., 0, ..., half]  (length = kernel_size)
        offsets = torch.arange(kernel_size) - half
        mask = (offsets <= 0).float()  # keep past and current, zero future
        return mask  # shape: (k,)

    # ------------------------------------------------------------------
    # Local window extraction via unfold (section 10.1)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_windows(
        x: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        """Pad and unfold to extract local windows.

        Args:
            x: (B, n, d_h)
            kernel_size: window size k

        Returns:
            windows: (B, n, k, d_h)
        """
        B, n, d_h = x.shape
        pad = kernel_size // 2
        # Zero-pad along sequence dimension: (B, n+k-1, d_h)
        x_padded = F.pad(x, (0, 0, pad, pad), mode="constant", value=0.0)
        # Unfold along dim=1 to get windows of size kernel_size
        # unfold(dim, size, step) -> (B, n, d_h, k)
        windows = x_padded.unfold(1, kernel_size, 1)  # (B, n, d_h, k)
        # Transpose last two dims to get (B, n, k, d_h)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        return windows

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, n, d).

        Returns:
            Output tensor of shape (B, n, d).
        """
        B, n, d = x.shape
        assert d == self.d_model, (
            f"Input dim {d} != d_model {self.d_model}"
        )

        # --- Step 1: Head projection (fused) ---
        # (B, n, d) -> (B, n, d) -> (B, n, H, d_h) -> (H, B, n, d_h)
        x_proj = self.head_proj_in(x)  # (B, n, d)
        x_heads = x_proj.view(B, n, self.num_heads, self.d_head)
        # Permute to (H, B, n, d_h) so we can iterate over heads via indexing
        x_heads = x_heads.permute(2, 0, 1, 3)  # (H, B, n, d_h)

        head_outputs = []
        kernels_for_logging: Dict[int, torch.Tensor] = {}
        alphas_for_logging: Dict[int, torch.Tensor] = {}

        for h in range(self.num_heads):
            x_h = x_heads[h]  # (B, n, d_h)
            k_h = self.kernel_sizes[h]

            # --- Step 2: Kernel generation ---
            # FactoredKernelGenerator returns:
            #   K_hat: (B, n, k_h, d_h) -- the final kernel (base + alpha * norm(delta))
            #   alpha: scalar tensor -- current alpha_h value
            K_hat, alpha_h = self.kernel_generators[h](x_h)

            # --- Step 3: Local window extraction ---
            windows = self._extract_windows(x_h, k_h)  # (B, n, k_h, d_h)

            # --- Optional causal masking ---
            if self.causal:
                causal_mask = getattr(self, f"causal_mask_{k_h}")
                # Broadcast mask: (k,) -> (1, 1, k, 1) to match (B, n, k, d_h)
                K_hat = K_hat * causal_mask.view(1, 1, k_h, 1)

            # --- Step 4: Kernel application ---
            # einsum('bnkd,bnkd->bnd', K_hat, windows)
            o_h = torch.einsum("bnkd,bnkd->bnd", K_hat, windows)  # (B, n, d_h)

            head_outputs.append(o_h)

            # Store for external logging/visualization
            kernels_for_logging[h] = K_hat.detach()
            alphas_for_logging[h] = alpha_h.detach()

        # --- Step 5: Concatenate heads and output projection ---
        # Concatenate along the last dimension: (B, n, d)
        concat = torch.cat(head_outputs, dim=-1)  # (B, n, H * d_h) = (B, n, d)
        output = self.out_proj(concat)  # (B, n, d)
        output = self.dropout(output)

        # Save for logging
        self._last_kernels = kernels_for_logging
        self._last_alphas = alphas_for_logging

        return output

    # ------------------------------------------------------------------
    # Convenience accessors for visualization / logging
    # ------------------------------------------------------------------

    def get_last_kernels(self) -> Optional[Dict[int, torch.Tensor]]:
        """Return the generated kernels from the most recent forward pass.

        Returns a dict mapping head index -> tensor of shape (B, n, k_h, d_h).
        Returns None if forward has not been called yet.
        """
        return self._last_kernels

    def get_last_alphas(self) -> Optional[Dict[int, torch.Tensor]]:
        """Return the alpha values from the most recent forward pass.

        Returns a dict mapping head index -> scalar tensor.
        Returns None if forward has not been called yet.
        """
        return self._last_alphas

    def get_kernel_sizes(self) -> List[int]:
        """Return the list of kernel sizes, one per head."""
        return list(self.kernel_sizes)
