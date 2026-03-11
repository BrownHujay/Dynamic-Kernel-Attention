"""Spectral (FFT) analysis of spatial components of generated kernels (Figure 13).

For each head (grouped by kernel size), computes the FFT of the spatial
component s_i across all tokens in a batch and plots the average power
spectrum.

Hypothesis: small-kernel heads produce high-frequency kernels (edge
detectors), large-kernel heads produce low-frequency kernels (smooth,
averaging).

All figures saved as PNG + PDF.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
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
# Spatial component extraction
# ---------------------------------------------------------------------------

def extract_spatial_components(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
) -> Dict[int, np.ndarray]:
    """Extract the spatial profile of each kernel by summing across channels.

    For each head, the spatial component of the generated kernel at token *i*
    is defined as sum_{d} |K_hat[i, :, d]|, yielding a vector of length k_h.

    We use absolute values so that the magnitude reflects the importance of
    each spatial position regardless of sign.

    Returns:
        dict mapping head_idx -> (B * n, k_h) numpy array, one spatial
        profile per token across the entire batch.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    dka = dka_modules[layer_idx]
    kernels = dka.get_last_kernels()

    spatial: Dict[int, np.ndarray] = {}
    for h, k_tensor in kernels.items():
        # k_tensor: (B, n, k_h, d_h)
        # Sum absolute values across d_h -> (B, n, k_h)
        s = k_tensor.abs().sum(dim=-1)  # (B, n, k_h)
        B, n, k_h = s.shape
        spatial[h] = s.reshape(B * n, k_h).cpu().numpy()

    return spatial


# ---------------------------------------------------------------------------
# FFT computation
# ---------------------------------------------------------------------------

def compute_power_spectra(
    spatial_components: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """Compute average power spectrum for each head's spatial profiles.

    For each head with spatial profiles of shape (N_tokens, k_h):
      1. Apply 1-D FFT along the k_h axis for every token.
      2. Compute |FFT|^2 (power spectrum).
      3. Average across all tokens.

    Returns:
        dict mapping head_idx -> (k_h // 2 + 1,) average power spectrum
        (one-sided, since the spatial signals are real-valued).
    """
    spectra: Dict[int, np.ndarray] = {}
    for h, s in spatial_components.items():
        # s: (N_tokens, k_h)
        fft_vals = np.fft.rfft(s, axis=-1)  # (N_tokens, k_h // 2 + 1)
        power = np.abs(fft_vals) ** 2
        avg_power = power.mean(axis=0)  # (k_h // 2 + 1,)
        spectra[h] = avg_power
    return spectra


# ---------------------------------------------------------------------------
# Figure 13: FFT analysis
# ---------------------------------------------------------------------------

def plot_spectral_analysis(
    model: nn.Module,
    x: torch.Tensor,
    layer_idx: int = 0,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 13 -- FFT analysis of spatial components, grouped by kernel size.

    One subplot per unique kernel size.  Within each subplot, one line per
    head that uses that kernel size.  X-axis = frequency index, Y-axis =
    average power (log scale).

    Args:
        model: Trained DKA model.
        x: Input tensor.
        layer_idx: Which DKA layer.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    spatial = extract_spatial_components(model, x, layer_idx)
    spectra = compute_power_spectra(spatial)

    dka_modules = _get_dka_modules(model)
    ksizes = dka_modules[layer_idx].get_kernel_sizes()

    # Group heads by kernel size
    groups: Dict[int, List[int]] = defaultdict(list)
    for h, k in enumerate(ksizes):
        groups[k].append(h)

    sorted_sizes = sorted(groups.keys())
    ncols = len(sorted_sizes)
    fig, axes = plt.subplots(
        1, ncols, figsize=(4 * ncols, 3.5), squeeze=False,
    )
    axes = axes[0]

    colors = plt.cm.tab10(np.linspace(0, 1, len(ksizes)))

    for col, ksize in enumerate(sorted_sizes):
        ax = axes[col]
        for h in groups[ksize]:
            freq_idx = np.arange(len(spectra[h]))
            ax.semilogy(
                freq_idx, spectra[h],
                marker="o", markersize=3,
                label=f"Head {h}",
                color=colors[h],
            )
        ax.set_title(f"k = {ksize}", fontsize=10)
        ax.set_xlabel("Frequency index")
        if col == 0:
            ax.set_ylabel("Average power (log)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Figure 13: Spectral Analysis of Spatial Components (Layer {layer_idx})",
        fontsize=12, y=1.04,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig13_spectral_analysis")
    return fig


# ---------------------------------------------------------------------------
# Aggregate across layers
# ---------------------------------------------------------------------------

def plot_spectral_all_layers(
    model: nn.Module,
    x: torch.Tensor,
    save_dir: str = "figures",
) -> plt.Figure:
    """Plot spectral analysis for every DKA layer in a grid.

    Rows = layers, columns = unique kernel sizes.
    """
    _apply_rc()

    dka_modules = _get_dka_modules(model)
    num_layers = len(dka_modules)

    # Get unique kernel sizes from first layer (assumed same across layers)
    ksizes_0 = dka_modules[0].get_kernel_sizes()
    unique_sizes = sorted(set(ksizes_0))
    ncols = len(unique_sizes)

    fig, axes = plt.subplots(
        num_layers, ncols,
        figsize=(4 * ncols, 2.5 * num_layers),
        squeeze=False,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, len(ksizes_0)))

    for layer_idx in range(num_layers):
        spatial = extract_spatial_components(model, x, layer_idx)
        spectra = compute_power_spectra(spatial)
        ksizes = dka_modules[layer_idx].get_kernel_sizes()

        groups: Dict[int, List[int]] = defaultdict(list)
        for h, k in enumerate(ksizes):
            groups[k].append(h)

        for col, ksize in enumerate(unique_sizes):
            ax = axes[layer_idx, col]
            if ksize in groups:
                for h in groups[ksize]:
                    freq_idx = np.arange(len(spectra[h]))
                    ax.semilogy(
                        freq_idx, spectra[h],
                        marker="o", markersize=2,
                        label=f"H{h}", color=colors[h],
                    )
            if layer_idx == 0:
                ax.set_title(f"k={ksize}", fontsize=9)
            if col == 0:
                ax.set_ylabel(f"L{layer_idx}", fontsize=9)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            if layer_idx == 0:
                ax.legend(fontsize=5)

    fig.suptitle(
        "Spectral Analysis: All Layers x Kernel Sizes", fontsize=12, y=1.02,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig13_spectral_all_layers")
    return fig


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    model: nn.Module,
    data: torch.Tensor,
    save_dir: str = "figures",
    layer_idx: int = 0,
) -> None:
    """Generate all spectral analysis figures.

    Args:
        model: Trained DKA model.
        data: Input tensor.
        save_dir: Output directory.
        layer_idx: DKA layer for single-layer analysis.
    """
    plot_spectral_analysis(model, data, layer_idx=layer_idx, save_dir=save_dir)
    plot_spectral_all_layers(model, data, save_dir=save_dir)
