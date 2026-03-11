"""Kernel heatmap visualizations for DKA.

Generates:
  - Figure 7:  Generated kernels for different image regions
               (grid: rows = regions, cols = heads).
  - Figure 9:  Base kernel vs dynamic kernels comparison.
  - Figure 11: Kernel patterns across a sentence (strip of heatmaps,
               one per token, with words annotated below).

Supports both image and text models.

All figures are saved as PNG + PDF in *save_dir*.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_dka_modules(model: nn.Module) -> list:
    """Recursively collect all DKAModule instances from *model*."""
    from dka.models.dka_module import DKAModule
    modules = []
    for m in model.modules():
        if isinstance(m, DKAModule):
            modules.append(m)
    return modules


def _extract_kernels(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor], List[int]]:
    """Run a forward pass and return kernels from *layer_idx*.

    Returns:
        kernels: dict  head_idx -> (B, n, k_h, d_h)
        alphas:  dict  head_idx -> scalar tensor
        kernel_sizes: list[int]
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    assert layer_idx < len(dka_modules), (
        f"layer_idx {layer_idx} out of range (model has {len(dka_modules)} DKA layers)"
    )
    dka = dka_modules[layer_idx]
    kernels = dka.get_last_kernels()
    alphas = dka.get_last_alphas()
    ksizes = dka.get_kernel_sizes()
    return kernels, alphas, ksizes


def _save(fig: plt.Figure, save_dir: str, name: str) -> None:
    p = Path(save_dir)
    p.mkdir(parents=True, exist_ok=True)
    fig.savefig(p / f"{name}.png")
    fig.savefig(p / f"{name}.pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 7: Generated kernels for different image regions
# ---------------------------------------------------------------------------

def plot_kernels_for_regions(
    model: nn.Module,
    image_tensor: torch.Tensor,
    region_indices: List[List[int]],
    region_labels: List[str],
    layer_idx: int = 0,
    batch_idx: int = 0,
    save_dir: str = "figures",
    original_image: Optional[np.ndarray] = None,
) -> plt.Figure:
    """Figure 7 -- generated kernels for semantically distinct image regions.

    Args:
        model: Trained DKA model.
        image_tensor: Input image tensor of shape (1, n, d) or (B, n, d).
            If the model has a patch embedding stage, pass the raw image
            tensor (B, C, H, W) and the model should handle it.
        region_indices: List of lists; each inner list contains token indices
            belonging to one spatial region (e.g., sky, edge, object).
        region_labels: Human-readable label for each region.
        layer_idx: Which DKA layer to visualize.
        batch_idx: Which sample in the batch to use.
        save_dir: Directory for saving figures.
        original_image: Optional (H, W, 3) numpy array for display.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()
    kernels, alphas, ksizes = _extract_kernels(model, image_tensor, layer_idx)

    num_heads = len(kernels)
    num_regions = len(region_indices)

    extra_col = 1 if original_image is not None else 0
    ncols = num_heads + extra_col
    nrows = num_regions

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(2.0 * ncols, 2.0 * nrows),
        squeeze=False,
    )

    # Optional: show original image in first column
    if original_image is not None:
        for r in range(nrows):
            axes[r, 0].imshow(original_image)
            axes[r, 0].set_title(region_labels[r], fontsize=8)
            axes[r, 0].axis("off")

    for r, (indices, label) in enumerate(zip(region_indices, region_labels)):
        for h in range(num_heads):
            col = h + extra_col
            # Average the kernels over the tokens in this region
            k_vals = kernels[h][batch_idx, indices]  # (len(indices), k_h, d_h)
            avg_kernel = k_vals.mean(dim=0).cpu().numpy()  # (k_h, d_h)

            ax = axes[r, col]
            sns.heatmap(
                avg_kernel, ax=ax, cmap="RdBu_r", center=0,
                cbar=False, xticklabels=False, yticklabels=False,
            )
            if r == 0:
                ax.set_title(f"H{h} (k={ksizes[h]})", fontsize=8)
            if h == 0 and original_image is None:
                ax.set_ylabel(label, fontsize=8)

    fig.suptitle(
        f"Figure 7: Generated Kernels by Region (Layer {layer_idx})",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig07_kernels_by_region")
    return fig


# ---------------------------------------------------------------------------
# Figure 9: Base kernel vs dynamic kernels comparison
# ---------------------------------------------------------------------------

def plot_base_vs_dynamic(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    token_indices: Optional[List[int]] = None,
    num_tokens: int = 6,
    batch_idx: int = 0,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 9 -- base kernel vs dynamic kernels for selected tokens.

    Shows K_base as the first heatmap, then K_hat_i for several tokens,
    annotated with the alpha_h value.

    Args:
        model: Trained DKA model.
        x: Input tensor (B, n, d) or raw input for the model.
        layer_idx: Which DKA layer.
        head_idx: Which head to visualize.
        token_indices: Specific token positions to show. If None, evenly
            spaced tokens are chosen.
        num_tokens: Number of dynamic kernels to display (ignored if
            token_indices is given).
        batch_idx: Which batch element.
        save_dir: Directory for saving figures.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()
    kernels, alphas, ksizes = _extract_kernels(model, x, layer_idx)

    K_hat = kernels[head_idx][batch_idx]  # (n, k_h, d_h)
    n = K_hat.shape[0]

    # Get base kernel from the kernel generator
    dka_modules = _get_dka_modules(model)
    gen = dka_modules[layer_idx].kernel_generators[head_idx]
    K_base = gen.K_base.detach().cpu().numpy()  # (k_h, d_h)
    alpha_val = gen.alpha.detach().cpu().item()

    if token_indices is None:
        token_indices = np.linspace(0, n - 1, num_tokens, dtype=int).tolist()

    ncols = 1 + len(token_indices)
    fig, axes = plt.subplots(1, ncols, figsize=(2.2 * ncols, 2.5), squeeze=False)
    axes = axes[0]

    # Base kernel
    sns.heatmap(
        K_base, ax=axes[0], cmap="RdBu_r", center=0,
        cbar=False, xticklabels=False, yticklabels=False,
    )
    axes[0].set_title("$K_{base}$", fontsize=9)

    vmin = K_hat.cpu().numpy().min()
    vmax = K_hat.cpu().numpy().max()

    for i, tidx in enumerate(token_indices):
        k_i = K_hat[tidx].cpu().numpy()  # (k_h, d_h)
        ax = axes[i + 1]
        sns.heatmap(
            k_i, ax=ax, cmap="RdBu_r", center=0,
            vmin=vmin, vmax=vmax,
            cbar=(i == len(token_indices) - 1),
            xticklabels=False, yticklabels=False,
        )
        ax.set_title(f"$\\hat{{K}}_{{{tidx}}}$", fontsize=9)

    fig.suptitle(
        f"Figure 9: Base vs Dynamic Kernels  "
        f"(Layer {layer_idx}, Head {head_idx}, "
        f"$\\alpha_h$={alpha_val:.4f})",
        fontsize=11, y=1.04,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig09_base_vs_dynamic")
    return fig


# ---------------------------------------------------------------------------
# Figure 11: Kernel patterns across a sentence
# ---------------------------------------------------------------------------

def plot_kernel_strip(
    model: nn.Module,
    x: torch.Tensor,
    tokens: List[str],
    layer_idx: int = 0,
    head_idx: int = 0,
    batch_idx: int = 0,
    max_tokens: int = 20,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 11 -- kernel heatmap strip for every token in a sentence.

    Args:
        model: Trained DKA text model.
        x: Input tensor (B, n, d) or raw input.
        tokens: List of string tokens corresponding to positions.
        layer_idx: Which DKA layer.
        head_idx: Which head.
        batch_idx: Which batch element.
        max_tokens: Truncate display to at most this many tokens.
        save_dir: Directory for saving figures.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()
    kernels, alphas, ksizes = _extract_kernels(model, x, layer_idx)
    K_hat = kernels[head_idx][batch_idx]  # (n, k_h, d_h)

    display_len = min(K_hat.shape[0], len(tokens), max_tokens)
    tokens_disp = tokens[:display_len]

    fig, axes = plt.subplots(
        1, display_len,
        figsize=(1.5 * display_len, 2.5),
        squeeze=False,
    )
    axes = axes[0]

    for i in range(display_len):
        k_i = K_hat[i].cpu().numpy()
        ax = axes[i]
        sns.heatmap(
            k_i, ax=ax, cmap="RdBu_r", center=0,
            cbar=False, xticklabels=False, yticklabels=False,
        )
        ax.set_xlabel(tokens_disp[i], fontsize=7, rotation=45, ha="right")
        ax.set_title("", fontsize=1)

    fig.suptitle(
        f"Figure 11: Kernel Patterns Across Sentence  "
        f"(Layer {layer_idx}, Head {head_idx}, k={ksizes[head_idx]})",
        fontsize=11, y=1.05,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig11_kernel_strip")
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    model: nn.Module,
    data: torch.Tensor,
    save_dir: str = "figures",
    layer_idx: int = 0,
    region_indices: Optional[List[List[int]]] = None,
    region_labels: Optional[List[str]] = None,
    original_image: Optional[np.ndarray] = None,
    tokens: Optional[List[str]] = None,
) -> None:
    """Generate all kernel visualization figures.

    For image models, provide *region_indices* / *region_labels* /
    *original_image*.  For text models, provide *tokens*.

    Args:
        model: Trained DKA model.
        data: Input tensor (B, n, d) or raw model input.
        save_dir: Output directory for figures.
        layer_idx: Which DKA layer.
        region_indices: Token indices per region (image).
        region_labels: Labels per region (image).
        original_image: Original image as numpy array (image).
        tokens: Token strings (text).
    """
    # Figure 9 -- always applicable
    plot_base_vs_dynamic(
        model, data, layer_idx=layer_idx, save_dir=save_dir,
    )

    # Figure 7 -- image regions
    if region_indices is not None and region_labels is not None:
        plot_kernels_for_regions(
            model, data,
            region_indices=region_indices,
            region_labels=region_labels,
            layer_idx=layer_idx,
            save_dir=save_dir,
            original_image=original_image,
        )

    # Figure 11 -- text token strip
    if tokens is not None:
        plot_kernel_strip(
            model, data,
            tokens=tokens,
            layer_idx=layer_idx,
            save_dir=save_dir,
        )
