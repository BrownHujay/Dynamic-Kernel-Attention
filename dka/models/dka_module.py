"""
Core Dynamic Kernel Attention (DKA) module.

Replaces standard multi-head self-attention with per-token dynamic kernel
generation and local window application. See DKA_BUILD_GUIDE.md sections 3.2-3.3
for the full specification.

Supports both 1D windows (text) and 2D windows (images). For images, kernel_sizes
specify the 2D side length (e.g., 3 means a 3x3 window = 9 spatial positions),
giving true CNN-style 2D locality.

Input:  X of shape (B, n, d)
Output: O of shape (B, n, d)
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel_generator import FactoredKernelGenerator

# Default multi-scale kernel sizes per head (H=8)
# For images: 2D side lengths (3 means 3x3=9 neighbors)
IMAGE_KERNEL_SIZES = [3, 3, 3, 3, 5, 5, 5, 5]
# For text: 1D widths (unchanged)
TEXT_KERNEL_SIZES = [3, 3, 7, 7, 11, 11, 21, 21]


class DKAModule(nn.Module):
    """
    Dynamic Kernel Attention module -- drop-in replacement for multi-head
    self-attention.

    For each head:
      1. Project input into head subspace.
      2. Generate a per-token convolutional kernel via FactoredKernelGenerator.
      3. Extract local windows with unfold (1D for text, 2D for images).
      4. Apply kernels to windows via einsum.
    Concatenate heads and apply output projection.

    Args:
        d_model: Model embedding dimension.
        num_heads: Number of attention heads (H).
        rank: Rank of factored kernel decomposition (R).
        kernel_sizes: List of kernel sizes, one per head. Length must equal
            num_heads. For images with grid_size set, these are 2D side lengths
            (e.g., 3 means 3x3 window). For text, these are 1D widths.
        mode: "image" or "text". Used for default kernel sizes when
            kernel_sizes is None.
        num_layers: Total number of transformer layers in the model. Used for
            output projection initialization scaling.
        causal: If True, apply causal masking (zero out future positions in
            the kernel window). Only for 1D/text mode.
        grid_size: (H_grid, W_grid) tuple for 2D window extraction on images.
            When set, kernel_sizes are interpreted as 2D side lengths and
            windows are extracted as 2D patches. None for 1D/text mode.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        rank: int = 4,
        kernel_sizes: Optional[List[int]] = None,
        mode: str = "image",
        num_layers: int = 8,
        causal: bool = False,
        grid_size: Optional[Tuple[int, int]] = None,
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
        self.grid_size = grid_size
        self.use_2d = grid_size is not None

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

        # For 2D mode, kernel_sizes are side lengths; compute total spatial
        # positions (k_total = k_side^2) for the kernel generator.
        if self.use_2d:
            self.kernel_sides = list(self.kernel_sizes)  # 2D side lengths
            self.k_totals = [k * k for k in self.kernel_sides]
        else:
            self.kernel_sides = None
            self.k_totals = list(self.kernel_sizes)  # 1D widths = total positions

        # --- Step 1: Head input projections ---
        self.head_proj_in = nn.Linear(d_model, d_model, bias=True)

        # --- Step 2: Kernel generators (one per head) ---
        # k_h for the generator = total spatial positions in the window
        self.kernel_generators = nn.ModuleList([
            FactoredKernelGenerator(
                d_h=self.d_head,
                k_h=k_total,
                rank=rank,
            )
            for k_total in self.k_totals
        ])

        # --- Step 5: Output projection ---
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        # Pre-compute and register causal masks (1D text only)
        if self.causal:
            assert not self.use_2d, "Causal masking is not supported with 2D windows"
            for k in set(self.k_totals):
                mask = self._build_causal_mask(k)
                self.register_buffer(f"causal_mask_{k}", mask, persistent=False)

        # --- Expose for logging/visualization ---
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
    # Causal mask construction (1D text only)
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
        offsets = torch.arange(kernel_size) - half
        mask = (offsets <= 0).float()
        return mask

    # ------------------------------------------------------------------
    # Local window extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_windows_1d(
        x: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        """Pad and unfold to extract 1D local windows (for text).

        Args:
            x: (B, n, d_h)
            kernel_size: window size k

        Returns:
            windows: (B, n, k, d_h)
        """
        pad = kernel_size // 2
        # Zero-pad along sequence dimension
        x_padded = F.pad(x, (0, 0, pad, pad), mode="constant", value=0.0)
        # Unfold along dim=1: (B, n, d_h, k)
        windows = x_padded.unfold(1, kernel_size, 1)
        # Transpose to (B, n, k, d_h)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        return windows

    @staticmethod
    def _extract_windows_2d(
        x: torch.Tensor, kernel_side: int, grid_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Extract 2D local windows from a flattened image grid via F.unfold.

        Uses PyTorch's native im2col (F.unfold) for efficient 2D patch extraction.

        Args:
            x: (B, n, d_h) where n = H_grid * W_grid
            kernel_side: side length of the 2D window (e.g., 3 for 3x3)
            grid_size: (H_grid, W_grid)

        Returns:
            windows: (B, n, k_total, d_h) where k_total = kernel_side^2
        """
        B, n, d_h = x.shape
        H, W = grid_size
        pad = kernel_side // 2
        k_total = kernel_side * kernel_side

        # Reshape to NCHW for F.unfold: (B, d_h, H, W)
        x_2d = x.view(B, H, W, d_h).permute(0, 3, 1, 2).contiguous()

        # F.unfold: native im2col -> (B, d_h * k_total, n)
        patches = F.unfold(x_2d, kernel_size=kernel_side, padding=pad)

        # Reshape to (B, n, k_total, d_h)
        patches = patches.view(B, d_h, k_total, n).permute(0, 3, 2, 1)
        return patches

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
        x_proj = self.head_proj_in(x)  # (B, n, d)
        x_heads = x_proj.view(B, n, self.num_heads, self.d_head)
        x_heads = x_heads.permute(2, 0, 1, 3)  # (H, B, n, d_h)

        head_outputs = []
        kernels_for_logging: Dict[int, torch.Tensor] = {}
        alphas_for_logging: Dict[int, torch.Tensor] = {}

        for h in range(self.num_heads):
            x_h = x_heads[h]  # (B, n, d_h)
            k_total = self.k_totals[h]

            # --- Step 2: Kernel generation ---
            # K_hat: (B, n, k_total, d_h)
            K_hat, alpha_h = self.kernel_generators[h](x_h)

            # --- Step 3: Local window extraction ---
            if self.use_2d and self.kernel_sides is not None and self.grid_size is not None:
                windows = self._extract_windows_2d(x_h, self.kernel_sides[h], self.grid_size)
            else:
                windows = self._extract_windows_1d(x_h, k_total)

            # --- Optional causal masking (1D text only) ---
            if self.causal:
                causal_mask = getattr(self, f"causal_mask_{k_total}")
                K_hat = K_hat * causal_mask.view(1, 1, k_total, 1)

            # --- Step 4: Kernel application ---
            o_h = torch.einsum("bnkd,bnkd->bnd", K_hat, windows)  # (B, n, d_h)

            head_outputs.append(o_h)

            # Store for external logging/visualization
            kernels_for_logging[h] = K_hat.detach()
            alphas_for_logging[h] = alpha_h.detach()

        # --- Step 5: Concatenate heads and output projection ---
        concat = torch.cat(head_outputs, dim=-1)  # (B, n, d)
        output = self.out_proj(concat)  # (B, n, d)

        # Save for logging
        self._last_kernels = kernels_for_logging
        self._last_alphas = alphas_for_logging

        return output

    # ------------------------------------------------------------------
    # Convenience accessors for visualization / logging
    # ------------------------------------------------------------------

    def get_last_kernels(self) -> Optional[Dict[int, torch.Tensor]]:
        """Return the generated kernels from the most recent forward pass.

        Returns a dict mapping head index -> tensor of shape (B, n, k_total, d_h).
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
        """Return the list of total kernel sizes (spatial positions), one per head."""
        return list(self.k_totals)
