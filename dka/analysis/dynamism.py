"""Kernel dynamism analysis (Figures 19 and 20).

Figure 19: Kernel variance per layer -- bar chart showing how "dynamic"
    each layer is (average kernel variance across tokens in a batch).
Figure 20: Kernel variance per head within layer -- heatmap with rows =
    layers, columns = heads, color = kernel variance.

Higher variance = more diverse per-token kernels = more dynamic behavior.

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


def _get_dka_modules(model: nn.Module) -> list:
    from dka.models.dka_module import DKAModule
    return [m for m in model.modules() if isinstance(m, DKAModule)]


# ---------------------------------------------------------------------------
# Variance computation
# ---------------------------------------------------------------------------

def compute_kernel_variance(
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[int, Dict[int, float]]:
    """Compute kernel variance across tokens for all layers and heads.

    For each (layer, head), the generated kernel tensor has shape
    (B, n, k_h, d_h).  We flatten each kernel to (B*n, k_h*d_h) and
    compute the mean element-wise variance across the token dimension.

    This measures how much the kernels vary from token to token -- higher
    variance = more dynamic content-dependent behavior.

    Args:
        model: DKA model (will run a forward pass).
        x: Input tensor.

    Returns:
        {layer_idx: {head_idx: variance_scalar}}.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    result: Dict[int, Dict[int, float]] = {}

    for li, dka in enumerate(dka_modules):
        kernels = dka.get_last_kernels()
        result[li] = {}
        for h, k_tensor in kernels.items():
            # k_tensor: (B, n, k_h, d_h)
            B, n, k_h, d_h = k_tensor.shape
            # Flatten spatial and channel dims: (B, n, k_h * d_h)
            flat = k_tensor.reshape(B, n, -1).float()
            # Variance across token dimension (dim=1), averaged over
            # batch and feature dims
            var = flat.var(dim=1).mean().item()
            result[li][h] = var

    return result


def compute_kernel_variance_per_batch(
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[int, Dict[int, np.ndarray]]:
    """Like compute_kernel_variance but returns per-batch-element values.

    Returns:
        {layer_idx: {head_idx: (B,) numpy array of per-sample variances}}.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    result: Dict[int, Dict[int, np.ndarray]] = {}

    for li, dka in enumerate(dka_modules):
        kernels = dka.get_last_kernels()
        result[li] = {}
        for h, k_tensor in kernels.items():
            B, n, k_h, d_h = k_tensor.shape
            flat = k_tensor.reshape(B, n, -1).float()
            # Variance over tokens, mean over features, keep batch: (B,)
            var_per_sample = flat.var(dim=1).mean(dim=-1).cpu().numpy()
            result[li][h] = var_per_sample

    return result


# ---------------------------------------------------------------------------
# Figure 19: Kernel variance per layer (bar chart)
# ---------------------------------------------------------------------------

def plot_variance_per_layer(
    model: nn.Module,
    x: torch.Tensor,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 19 -- bar chart of average kernel variance per layer.

    Args:
        model: Trained DKA model.
        x: Input tensor.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    var_data = compute_kernel_variance(model, x)
    num_layers = len(var_data)
    layer_indices = sorted(var_data.keys())

    # Average variance across heads for each layer
    layer_means = []
    layer_stds = []
    for li in layer_indices:
        vals = list(var_data[li].values())
        layer_means.append(np.mean(vals))
        layer_stds.append(np.std(vals))

    fig, ax = plt.subplots(figsize=(max(6, num_layers * 0.8), 4))

    bars = ax.bar(
        range(num_layers), layer_means,
        yerr=layer_stds, capsize=4,
        color=plt.cm.viridis(np.linspace(0.2, 0.8, num_layers)),
        edgecolor="black", linewidth=0.5,
    )

    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Kernel Variance (across tokens)")
    ax.set_title("Figure 19: Kernel Variance Per Layer")
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels([f"L{li}" for li in layer_indices])
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir, "fig19_variance_per_layer")
    return fig


# ---------------------------------------------------------------------------
# Figure 20: Kernel variance per head (heatmap)
# ---------------------------------------------------------------------------

def plot_variance_heatmap(
    model: nn.Module,
    x: torch.Tensor,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 20 -- heatmap of kernel variance (rows=layers, cols=heads).

    Args:
        model: Trained DKA model.
        x: Input tensor.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    var_data = compute_kernel_variance(model, x)
    layer_indices = sorted(var_data.keys())
    num_layers = len(layer_indices)

    # Determine max heads across layers
    max_heads = max(len(var_data[li]) for li in layer_indices)

    # Build matrix
    matrix = np.full((num_layers, max_heads), np.nan)
    for row, li in enumerate(layer_indices):
        for hi in sorted(var_data[li].keys()):
            matrix[row, hi] = var_data[li][hi]

    # Get kernel sizes for column labels
    dka_modules = _get_dka_modules(model)
    if dka_modules:
        ksizes = dka_modules[0].get_kernel_sizes()
        col_labels = [f"H{h}\n(k={ksizes[h]})" for h in range(max_heads)]
    else:
        col_labels = [f"H{h}" for h in range(max_heads)]

    row_labels = [f"Layer {li}" for li in layer_indices]

    fig, ax = plt.subplots(figsize=(max(6, max_heads * 1.2), max(4, num_layers * 0.6)))

    sns.heatmap(
        matrix, ax=ax,
        annot=True, fmt=".4f",
        cmap="YlOrRd",
        xticklabels=col_labels,
        yticklabels=row_labels,
        cbar_kws={"label": "Kernel Variance"},
        linewidths=0.5, linecolor="white",
    )
    ax.set_title("Figure 20: Kernel Variance Per Head Within Layer")

    fig.tight_layout()
    _save(fig, save_dir, "fig20_variance_heatmap")
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    model: nn.Module,
    data: torch.Tensor,
    save_dir: str = "figures",
) -> None:
    """Generate all dynamism analysis figures (19 and 20).

    Args:
        model: Trained DKA model.
        data: Input tensor.
        save_dir: Output directory.
    """
    plot_variance_per_layer(model, data, save_dir=save_dir)
    plot_variance_heatmap(model, data, save_dir=save_dir)

    # Print summary
    var_data = compute_kernel_variance(model, data)
    print("\n=== Kernel Dynamism (Variance) Summary ===")
    for li in sorted(var_data.keys()):
        vals = list(var_data[li].values())
        print(
            f"  Layer {li}: mean_var={np.mean(vals):.6f}  "
            f"max_var={np.max(vals):.6f}  min_var={np.min(vals):.6f}"
        )
