"""Kernel similarity matrix and diversity metrics (Figure 8).

Computes pairwise cosine similarity between generated kernels across all
tokens for a given head, displayed as an n x n heatmap.  Block structure
in the heatmap indicates that similar input patches generate similar
kernels -- the model has learned semantic grouping.

Also provides scalar diversity statistics useful for monitoring during
training (mean pairwise cosine similarity, collapse detection).

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
import torch.nn.functional as F

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


def _get_dka_modules(model: nn.Module) -> list:
    from dka.models.dka_module import DKAModule
    return [m for m in model.modules() if isinstance(m, DKAModule)]


# ---------------------------------------------------------------------------
# Cosine similarity computation
# ---------------------------------------------------------------------------

def compute_kernel_similarity_matrix(
    kernels: torch.Tensor,
) -> np.ndarray:
    """Compute pairwise cosine similarity between all token kernels.

    Args:
        kernels: (n, k_h, d_h) tensor of generated kernels for one head
            and one batch element.

    Returns:
        (n, n) numpy array of pairwise cosine similarities in [-1, 1].
    """
    n = kernels.shape[0]
    # Flatten each kernel to a vector: (n, k_h * d_h)
    flat = kernels.reshape(n, -1).float()
    # Normalize
    flat_norm = F.normalize(flat, p=2, dim=-1)
    # Pairwise cosine similarity: (n, n)
    sim = torch.mm(flat_norm, flat_norm.t())
    return sim.cpu().numpy()


def compute_diversity_stats(
    kernels: torch.Tensor,
) -> Dict[str, float]:
    """Compute scalar diversity statistics for one head's kernels.

    Args:
        kernels: (n, k_h, d_h) tensor.

    Returns:
        dict with keys:
            - mean_cosine_sim: mean of upper-triangle pairwise cosine sims.
            - std_cosine_sim: std of upper-triangle pairwise cosine sims.
            - max_cosine_sim: max pairwise cosine sim (excluding self).
            - min_cosine_sim: min pairwise cosine sim.
    """
    sim = compute_kernel_similarity_matrix(kernels)
    n = sim.shape[0]
    # Extract upper triangle (excluding diagonal)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    upper = sim[mask]

    return {
        "mean_cosine_sim": float(upper.mean()),
        "std_cosine_sim": float(upper.std()),
        "max_cosine_sim": float(upper.max()),
        "min_cosine_sim": float(upper.min()),
    }


def compute_all_diversity_stats(
    model: nn.Module,
    x: torch.Tensor,
    batch_idx: int = 0,
) -> Dict[int, Dict[int, Dict[str, float]]]:
    """Compute diversity stats for all layers and heads.

    Returns:
        {layer_idx: {head_idx: {stat_name: value}}}.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    results: Dict[int, Dict[int, Dict[str, float]]] = {}

    for li, dka in enumerate(dka_modules):
        kernels = dka.get_last_kernels()
        results[li] = {}
        for h, k_tensor in kernels.items():
            k = k_tensor[batch_idx]  # (n, k_h, d_h)
            results[li][h] = compute_diversity_stats(k)

    return results


# ---------------------------------------------------------------------------
# Figure 8: Kernel similarity matrix
# ---------------------------------------------------------------------------

def plot_kernel_similarity(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    head_idx: int = 0,
    batch_idx: int = 0,
    original_image: Optional[np.ndarray] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 8 -- kernel similarity matrix with optional image patches.

    Left: n x n heatmap of pairwise cosine similarity.
    Right (optional): original image with patch grid overlay.

    Args:
        model: Trained DKA model.
        x: Input tensor.
        layer_idx: DKA layer.
        head_idx: Head to visualize.
        batch_idx: Batch element.
        original_image: Optional (H, W, 3) numpy array of the original image.
        patch_grid: Optional (rows, cols) indicating the patch layout, e.g.
            (8, 8) for CIFAR-10 with patch_size=4.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    dka = dka_modules[layer_idx]
    kernels = dka.get_last_kernels()
    ksizes = dka.get_kernel_sizes()

    k_head = kernels[head_idx][batch_idx]  # (n, k_h, d_h)
    sim_matrix = compute_kernel_similarity_matrix(k_head)
    stats = compute_diversity_stats(k_head)

    has_image = original_image is not None
    ncols = 2 if has_image else 1

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5.5), squeeze=False)
    axes = axes[0]

    # Similarity heatmap
    sns.heatmap(
        sim_matrix, ax=axes[0], cmap="RdYlBu_r",
        vmin=-1, vmax=1, square=True,
        cbar_kws={"shrink": 0.7, "label": "Cosine Similarity"},
        xticklabels=False, yticklabels=False,
    )
    axes[0].set_title(
        f"Kernel Similarity (Layer {layer_idx}, Head {head_idx}, "
        f"k={ksizes[head_idx]})\n"
        f"mean={stats['mean_cosine_sim']:.3f}, "
        f"std={stats['std_cosine_sim']:.3f}",
        fontsize=10,
    )
    axes[0].set_xlabel("Token index")
    axes[0].set_ylabel("Token index")

    # Original image with patch grid
    if has_image:
        axes[1].imshow(original_image)
        if patch_grid is not None:
            pr, pc = patch_grid
            h_img, w_img = original_image.shape[:2]
            ph, pw = h_img / pr, w_img / pc
            for i in range(1, pr):
                axes[1].axhline(i * ph, color="white", linewidth=0.5, alpha=0.7)
            for j in range(1, pc):
                axes[1].axvline(j * pw, color="white", linewidth=0.5, alpha=0.7)
        axes[1].set_title("Original Image (patch grid)", fontsize=10)
        axes[1].axis("off")

    fig.suptitle("Figure 8: Kernel Similarity Matrix", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, save_dir, "fig08_kernel_similarity")
    return fig


# ---------------------------------------------------------------------------
# Multi-head similarity overview
# ---------------------------------------------------------------------------

def plot_similarity_all_heads(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    batch_idx: int = 0,
    save_dir: str = "figures",
) -> plt.Figure:
    """Plot kernel similarity matrices for all heads in a single layer."""
    _apply_rc()

    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    dka = dka_modules[layer_idx]
    kernels = dka.get_last_kernels()
    ksizes = dka.get_kernel_sizes()
    num_heads = len(kernels)

    ncols = min(4, num_heads)
    nrows = math.ceil(num_heads / ncols)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.8 * nrows), squeeze=False,
    )

    for h in range(num_heads):
        r, c = divmod(h, ncols)
        k_head = kernels[h][batch_idx]
        sim = compute_kernel_similarity_matrix(k_head)
        stats = compute_diversity_stats(k_head)

        sns.heatmap(
            sim, ax=axes[r, c], cmap="RdYlBu_r",
            vmin=-1, vmax=1, square=True, cbar=False,
            xticklabels=False, yticklabels=False,
        )
        axes[r, c].set_title(
            f"H{h} k={ksizes[h]} "
            f"(mean={stats['mean_cosine_sim']:.2f})",
            fontsize=8,
        )

    for idx in range(num_heads, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        f"Kernel Similarity Matrices -- Layer {layer_idx}",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig08_similarity_all_heads")
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    model: nn.Module,
    data: torch.Tensor,
    save_dir: str = "figures",
    layer_idx: int = 0,
    original_image: Optional[np.ndarray] = None,
    patch_grid: Optional[Tuple[int, int]] = None,
) -> None:
    """Generate all diversity/similarity figures.

    Args:
        model: Trained DKA model.
        data: Input tensor.
        save_dir: Output directory.
        layer_idx: DKA layer.
        original_image: Optional image for Figure 8.
        patch_grid: Optional patch grid dimensions.
    """
    plot_kernel_similarity(
        model, data, layer_idx=layer_idx,
        original_image=original_image, patch_grid=patch_grid,
        save_dir=save_dir,
    )
    plot_similarity_all_heads(
        model, data, layer_idx=layer_idx, save_dir=save_dir,
    )

    # Print summary stats
    stats = compute_all_diversity_stats(model, data)
    print("\n=== Kernel Diversity Statistics ===")
    for li in sorted(stats.keys()):
        for hi in sorted(stats[li].keys()):
            s = stats[li][hi]
            print(
                f"  Layer {li} Head {hi}: "
                f"mean_sim={s['mean_cosine_sim']:.4f}  "
                f"std_sim={s['std_cosine_sim']:.4f}  "
                f"max_sim={s['max_cosine_sim']:.4f}"
            )
