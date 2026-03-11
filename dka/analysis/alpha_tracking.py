"""Alpha trajectory logging and plotting (Figure 6).

Provides:
  - AlphaTracker: lightweight logger that records alpha_h values for every
    head in every layer at each epoch (or any user-defined frequency).
  - plot_alpha_trajectories: generates the Figure 6 plot with one subplot
    per layer, 8 lines per subplot (one per head), colored by kernel size.

All figures saved as PNG + PDF.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# Kernel-size -> color mapping
# ---------------------------------------------------------------------------
_KSIZE_COLORS = {
    3: "#1f77b4",   # blue
    5: "#2ca02c",   # green
    7: "#ff7f0e",   # orange
    11: "#d62728",  # red
    15: "#9467bd",  # purple
    21: "#8c564b",  # brown
}


def _color_for_ksize(ksize: int) -> str:
    return _KSIZE_COLORS.get(ksize, "#7f7f7f")


# ---------------------------------------------------------------------------
# AlphaTracker
# ---------------------------------------------------------------------------

class AlphaTracker:
    """Lightweight logger for alpha_h values during training.

    Usage in training loop::

        tracker = AlphaTracker()

        for epoch in range(num_epochs):
            train_one_epoch(model, ...)
            tracker.log(model, epoch)

        tracker.save("logs/alpha_log.json")
        tracker.plot(save_dir="figures")

    Internal storage layout::

        self.history[layer_idx][head_idx] = [(epoch, alpha_value), ...]
    """

    def __init__(self) -> None:
        # history[layer_idx][head_idx] = list of (epoch, alpha_value)
        self.history: Dict[int, Dict[int, List[tuple]]] = defaultdict(
            lambda: defaultdict(list)
        )
        # Cache kernel sizes per layer so we can color by size in plots
        self._kernel_sizes: Dict[int, List[int]] = {}

    # ---------------------------------------------------------------
    # Logging
    # ---------------------------------------------------------------

    def log(self, model: nn.Module, epoch: int) -> None:
        """Record current alpha values from all DKA layers.

        Call once per epoch (or at any desired frequency).

        Args:
            model: The DKA model (can be wrapped in DataParallel / DDP).
            epoch: Current epoch number (or step number).
        """
        dka_modules = _get_dka_modules(model)
        for layer_idx, dka in enumerate(dka_modules):
            if layer_idx not in self._kernel_sizes:
                self._kernel_sizes[layer_idx] = dka.get_kernel_sizes()
            for h, gen in enumerate(dka.kernel_generators):
                alpha_val = gen.alpha.detach().cpu().item()
                self.history[layer_idx][h].append((epoch, alpha_val))

    def log_from_dict(
        self,
        alpha_dict: Dict[int, Dict[int, float]],
        epoch: int,
        kernel_sizes: Optional[Dict[int, List[int]]] = None,
    ) -> None:
        """Record alpha values from a pre-extracted dictionary.

        Useful when alpha values are logged via W&B or other frameworks
        and reconstructed later.

        Args:
            alpha_dict: {layer_idx: {head_idx: alpha_value}}.
            epoch: Epoch number.
            kernel_sizes: Optional {layer_idx: [k_h per head]}.
        """
        for layer_idx, heads in alpha_dict.items():
            for head_idx, alpha_val in heads.items():
                self.history[layer_idx][head_idx].append((epoch, alpha_val))
        if kernel_sizes is not None:
            self._kernel_sizes.update(kernel_sizes)

    # ---------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save alpha history to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "history": {
                str(li): {
                    str(hi): records
                    for hi, records in heads.items()
                }
                for li, heads in self.history.items()
            },
            "kernel_sizes": {
                str(li): ks for li, ks in self._kernel_sizes.items()
            },
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, path: str) -> None:
        """Load alpha history from a JSON file."""
        with open(path) as f:
            data = json.load(f)

        self.history = defaultdict(lambda: defaultdict(list))
        for li_str, heads in data["history"].items():
            li = int(li_str)
            for hi_str, records in heads.items():
                hi = int(hi_str)
                self.history[li][hi] = [tuple(r) for r in records]

        self._kernel_sizes = {}
        for li_str, ks in data.get("kernel_sizes", {}).items():
            self._kernel_sizes[int(li_str)] = ks

    # ---------------------------------------------------------------
    # Plotting (Figure 6)
    # ---------------------------------------------------------------

    def plot(
        self,
        save_dir: str = "figures",
        max_cols: int = 4,
    ) -> plt.Figure:
        """Figure 6 -- alpha trajectories.

        One subplot per layer, one line per head colored by kernel size.
        X-axis = epoch, Y-axis = alpha_h.

        Args:
            save_dir: Directory for saving figures.
            max_cols: Maximum columns in the subplot grid.

        Returns:
            matplotlib Figure.
        """
        return plot_alpha_trajectories(
            self.history, self._kernel_sizes,
            save_dir=save_dir, max_cols=max_cols,
        )


# ---------------------------------------------------------------------------
# Plot function (usable without the tracker class)
# ---------------------------------------------------------------------------

def plot_alpha_trajectories(
    history: Dict[int, Dict[int, List[tuple]]],
    kernel_sizes: Optional[Dict[int, List[int]]] = None,
    save_dir: str = "figures",
    max_cols: int = 4,
) -> plt.Figure:
    """Figure 6 -- alpha trajectory plot.

    Args:
        history: {layer_idx: {head_idx: [(epoch, alpha), ...]}}.
        kernel_sizes: Optional {layer_idx: [k_h, ...]}. Used for
            coloring lines by kernel size.
        save_dir: Output directory.
        max_cols: Max columns in grid.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    num_layers = len(history)
    if num_layers == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        ax.text(0.5, 0.5, "No alpha data logged", ha="center", va="center")
        _save(fig, save_dir, "fig06_alpha_trajectories")
        return fig

    ncols = min(max_cols, num_layers)
    nrows = int(np.ceil(num_layers / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.5 * ncols, 3.0 * nrows),
        squeeze=False,
        sharex=True,
    )

    layer_indices = sorted(history.keys())

    for i, layer_idx in enumerate(layer_indices):
        r, c = divmod(i, ncols)
        ax = axes[r, c]
        heads = history[layer_idx]

        for head_idx in sorted(heads.keys()):
            records = heads[head_idx]
            epochs = [r[0] for r in records]
            alphas = [r[1] for r in records]

            # Determine color from kernel size
            ksize = None
            if kernel_sizes and layer_idx in kernel_sizes:
                ks = kernel_sizes[layer_idx]
                if head_idx < len(ks):
                    ksize = ks[head_idx]

            color = _color_for_ksize(ksize) if ksize else None
            label = f"H{head_idx}" + (f" (k={ksize})" if ksize else "")

            ax.plot(epochs, alphas, label=label, color=color, linewidth=1.2)

        ax.set_title(f"Layer {layer_idx}", fontsize=10)
        ax.set_ylabel("$\\alpha_h$")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6, loc="upper left", ncol=2)

    # Label x-axis on bottom row
    for c in range(ncols):
        axes[nrows - 1, c].set_xlabel("Epoch")

    # Hide unused subplots
    for idx in range(num_layers, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r, c].axis("off")

    fig.suptitle(
        "Figure 6: $\\alpha_h$ Trajectories During Training",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig06_alpha_trajectories")
    return fig


# ---------------------------------------------------------------------------
# Convenience: extract current alphas from a model snapshot
# ---------------------------------------------------------------------------

def get_current_alphas(model: nn.Module) -> Dict[int, Dict[int, float]]:
    """Return current alpha values as {layer: {head: value}}.

    Useful for logging to W&B or TensorBoard independently of AlphaTracker.
    """
    dka_modules = _get_dka_modules(model)
    result: Dict[int, Dict[int, float]] = {}
    for li, dka in enumerate(dka_modules):
        result[li] = {}
        for hi, gen in enumerate(dka.kernel_generators):
            result[li][hi] = gen.alpha.detach().cpu().item()
    return result


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    model: nn.Module,
    data: Optional[torch.Tensor] = None,
    log_path: Optional[str] = None,
    save_dir: str = "figures",
) -> None:
    """Generate alpha trajectory figure.

    Either load a pre-saved log file (*log_path*), or display the current
    snapshot from the model (single point, useful for sanity checks).

    Args:
        model: Trained DKA model.
        data: Not used directly (alpha values come from model parameters).
        log_path: Path to a saved AlphaTracker JSON log.
        save_dir: Output directory.
    """
    tracker = AlphaTracker()

    if log_path is not None:
        tracker.load(log_path)
    else:
        # Single-point snapshot
        tracker.log(model, epoch=0)

    tracker.plot(save_dir=save_dir)
