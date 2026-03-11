"""DKA 'attention' maps vs real attention maps (Figure 12).

For DKA: the generated kernel implicitly defines local attention weights.
We sum kernel values across channels (d_h dimension) to produce a scalar
weight for each position in the local window.  For each token this gives a
1-D profile of shape (k_h,) showing how much it attends to each neighbor.

For ViT: we extract standard softmax attention maps from the same input.

The two are displayed side-by-side for visual comparison.

All figures saved as PNG + PDF.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Plotting defaults
# ---------------------------------------------------------------------------
_RC_PARAMS = {
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
}


def _apply_rc() -> None:
    plt.rcParams.update(_RC_PARAMS)


def _save(fig: plt.Figure, save_dir: str, name: str) -> None:
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{name}.png")
    fig.savefig(p / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# DKA attention-style map extraction
# ---------------------------------------------------------------------------

def _get_dka_modules(model: nn.Module) -> list:
    from dka.models.dka_module import DKAModule
    return [m for m in model.modules() if isinstance(m, DKAModule)]


def extract_dka_attention_maps(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    batch_idx: int = 0,
) -> Dict[int, np.ndarray]:
    """Extract pseudo-attention maps from a DKA layer.

    For each head, produces a (n, k_h) matrix where entry (i, j) is the
    summed absolute kernel weight that token *i* places on its *j*-th
    neighbor.

    Returns:
        dict mapping head_idx -> (n, k_h) numpy array.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    dka = dka_modules[layer_idx]
    kernels = dka.get_last_kernels()
    ksizes = dka.get_kernel_sizes()

    maps: Dict[int, np.ndarray] = {}
    for h, ksize in enumerate(ksizes):
        # kernels[h]: (B, n, k_h, d_h)
        k = kernels[h][batch_idx]  # (n, k_h, d_h)
        # Sum absolute values across channel dim -> (n, k_h)
        attn_style = k.abs().sum(dim=-1).cpu().numpy()
        maps[h] = attn_style
    return maps


def dka_attention_to_full_matrix(
    attn_map: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    """Expand a local (n, k) DKA attention map into a full (n, n) matrix.

    Entries outside the local window are zero.

    Args:
        attn_map: (n, k) array.
        kernel_size: kernel size k.

    Returns:
        (n, n) dense attention-style matrix.
    """
    n = attn_map.shape[0]
    half = kernel_size // 2
    full = np.zeros((n, n), dtype=attn_map.dtype)
    for i in range(n):
        for j_local in range(kernel_size):
            j_global = i - half + j_local
            if 0 <= j_global < n:
                full[i, j_global] = attn_map[i, j_local]
    return full


# ---------------------------------------------------------------------------
# ViT attention extraction (optional baseline)
# ---------------------------------------------------------------------------

def extract_vit_attention_maps(
    vit_model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    batch_idx: int = 0,
) -> Optional[np.ndarray]:
    """Extract attention maps from a standard ViT model.

    Attempts to find MultiheadAttention or similar modules and read their
    stored attention weights.  Returns (H, n, n) numpy array or None if
    extraction fails.

    This uses a forward hook approach that is compatible with both custom
    ViT implementations and timm models.
    """
    attn_weights_store: list = []

    def _hook_fn(module, input, output):
        # torch.nn.MultiheadAttention returns (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) >= 2:
            if output[1] is not None:
                attn_weights_store.append(output[1].detach())

    # Find attention modules
    attn_modules = []
    for m in vit_model.modules():
        if isinstance(m, nn.MultiheadAttention):
            attn_modules.append(m)

    if layer_idx >= len(attn_modules):
        return None

    handle = attn_modules[layer_idx].register_forward_hook(_hook_fn)

    vit_model.eval()
    with torch.no_grad():
        # Some ViT models need need_weights=True; we set it temporarily
        orig = getattr(attn_modules[layer_idx], "need_weights", True)
        attn_modules[layer_idx].need_weights = True
        try:
            _ = vit_model(x)
        finally:
            attn_modules[layer_idx].need_weights = orig
            handle.remove()

    if not attn_weights_store:
        return None

    # (B, H, n, n) or (B, n, n)
    w = attn_weights_store[0]
    if w.dim() == 3:
        return w[batch_idx].cpu().numpy()  # (n, n) or (H, n, n)
    elif w.dim() == 4:
        return w[batch_idx].cpu().numpy()  # (H, n, n)
    return w.cpu().numpy()


# ---------------------------------------------------------------------------
# Figure 12: Side-by-side comparison
# ---------------------------------------------------------------------------

def plot_attention_comparison(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    batch_idx: int = 0,
    vit_model: Optional[nn.Module] = None,
    vit_input: Optional[torch.Tensor] = None,
    vit_layer_idx: int = 0,
    vit_head_idx: int = 0,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 12 -- DKA pseudo-attention maps vs ViT attention maps.

    Left panel: DKA local attention pattern (expanded to n x n).
    Right panel: ViT global attention pattern (n x n).

    Args:
        model: Trained DKA model.
        x: DKA input tensor.
        layer_idx: DKA layer to visualize.
        head_idx: DKA head to visualize.
        batch_idx: Batch element.
        vit_model: Optional trained ViT model for comparison.
        vit_input: Input tensor for the ViT model.
        vit_layer_idx: ViT layer for attention extraction.
        vit_head_idx: ViT head for attention extraction.
        save_dir: Directory for saving figures.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    dka_maps = extract_dka_attention_maps(model, x, layer_idx, batch_idx)
    dka_modules = _get_dka_modules(model)
    ksizes = dka_modules[layer_idx].get_kernel_sizes()

    dka_local = dka_maps[head_idx]  # (n, k_h)
    dka_full = dka_attention_to_full_matrix(dka_local, ksizes[head_idx])

    has_vit = vit_model is not None and vit_input is not None
    ncols = 2 if has_vit else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5), squeeze=False)
    axes = axes[0]

    # DKA map
    sns.heatmap(
        dka_full, ax=axes[0], cmap="viridis", square=True,
        cbar_kws={"shrink": 0.7},
        xticklabels=False, yticklabels=False,
    )
    axes[0].set_title(
        f"DKA Pseudo-Attention\n(Layer {layer_idx}, Head {head_idx}, "
        f"k={ksizes[head_idx]})",
        fontsize=10,
    )
    axes[0].set_xlabel("Key position")
    axes[0].set_ylabel("Query position")

    # ViT map (if available)
    if has_vit:
        vit_attn = extract_vit_attention_maps(
            vit_model, vit_input, vit_layer_idx, batch_idx,
        )
        if vit_attn is not None:
            # Select one head if multi-head
            if vit_attn.ndim == 3:
                vit_map = vit_attn[vit_head_idx]
            else:
                vit_map = vit_attn
            sns.heatmap(
                vit_map, ax=axes[1], cmap="viridis", square=True,
                cbar_kws={"shrink": 0.7},
                xticklabels=False, yticklabels=False,
            )
            axes[1].set_title(
                f"ViT Attention\n(Layer {vit_layer_idx}, Head {vit_head_idx})",
                fontsize=10,
            )
            axes[1].set_xlabel("Key position")
            axes[1].set_ylabel("Query position")
        else:
            axes[1].text(
                0.5, 0.5, "ViT attention\nextraction failed",
                ha="center", va="center", fontsize=12,
                transform=axes[1].transAxes,
            )
            axes[1].set_title("ViT Attention (unavailable)")

    fig.suptitle("Figure 12: DKA vs Standard Attention Maps", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, "fig12_attention_comparison")
    return fig


# ---------------------------------------------------------------------------
# Multi-head overview
# ---------------------------------------------------------------------------

def plot_dka_attention_all_heads(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    batch_idx: int = 0,
    save_dir: str = "figures",
) -> plt.Figure:
    """Plot DKA pseudo-attention maps for all heads in one layer.

    Produces a grid with one subplot per head.
    """
    _apply_rc()
    dka_maps = extract_dka_attention_maps(model, x, layer_idx, batch_idx)
    dka_modules = _get_dka_modules(model)
    ksizes = dka_modules[layer_idx].get_kernel_sizes()

    num_heads = len(dka_maps)
    ncols = min(4, num_heads)
    nrows = math.ceil(num_heads / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False,
    )

    for h in range(num_heads):
        r, c = divmod(h, ncols)
        full = dka_attention_to_full_matrix(dka_maps[h], ksizes[h])
        sns.heatmap(
            full, ax=axes[r, c], cmap="viridis", square=True,
            cbar=False, xticklabels=False, yticklabels=False,
        )
        axes[r, c].set_title(f"Head {h} (k={ksizes[h]})", fontsize=9)

    # Hide unused subplots
    for idx in range(num_heads, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"DKA Pseudo-Attention Maps -- Layer {layer_idx}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig12_dka_all_heads")
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    model: nn.Module,
    data: torch.Tensor,
    save_dir: str = "figures",
    layer_idx: int = 0,
    vit_model: Optional[nn.Module] = None,
    vit_input: Optional[torch.Tensor] = None,
) -> None:
    """Generate all attention-map figures.

    Args:
        model: Trained DKA model.
        data: Input tensor.
        save_dir: Output directory.
        layer_idx: DKA layer to visualize.
        vit_model: Optional ViT model for comparison (Figure 12).
        vit_input: Input for ViT model.
    """
    plot_dka_attention_all_heads(model, data, layer_idx=layer_idx, save_dir=save_dir)
    plot_attention_comparison(
        model, data, layer_idx=layer_idx,
        vit_model=vit_model, vit_input=vit_input,
        save_dir=save_dir,
    )
