"""Ablation result plots (Figures 14 and 15).

Figure 14: Grouped bar chart of ablation results.
    Groups: Static vs Dynamic, Rank 1/2/4/8, Kernel size variants,
    Diversity loss variants, Residual variants, Generator capacity.
    Y-axis: accuracy.  Error bars for multi-seed runs.

Figure 15: Rank vs Accuracy vs Parameters scatter plot.
    X-axis = total params, Y-axis = accuracy, points labeled by rank R.
    Shows the efficiency-expressiveness tradeoff.

Both accept results as dictionaries and produce publication-quality plots.

All figures saved as PNG + PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
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


# ===================================================================
# Figure 14: Grouped Bar Chart of Ablation Results
# ===================================================================

def plot_ablation_bar_chart(
    ablation_groups: Dict[str, Dict[str, Dict[str, float]]],
    metric: str = "accuracy",
    save_dir: str = "figures",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Figure 14 -- grouped bar chart for ablation results.

    Args:
        ablation_groups: Nested dict with structure::

            {
                "group_name": {
                    "variant_name": {
                        "accuracy": 95.2,
                        "accuracy_std": 0.3,   # optional, for error bars
                    },
                    ...
                },
                ...
            }

            Example groups: "Static vs Dynamic", "Rank", "Kernel Size",
            "Diversity Loss", "Residual Structure", "Generator Capacity".

        metric: Which key to plot from each variant dict (default "accuracy").
        save_dir: Output directory.
        figsize: Optional (width, height) in inches.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    group_names = list(ablation_groups.keys())
    num_groups = len(group_names)

    # Determine the max number of variants in any group
    max_variants = max(len(v) for v in ablation_groups.values())

    # Use distinct colors for variants within each group
    cmap = plt.cm.Set2
    colors = [cmap(i / max(1, max_variants - 1)) for i in range(max_variants)]

    if figsize is None:
        figsize = (max(10, num_groups * 3), 5)

    fig, ax = plt.subplots(figsize=figsize)

    group_width = 0.8
    total_width = num_groups + (num_groups - 1) * 0.5  # spacing between groups

    x_offset = 0.0
    xtick_positions = []
    xtick_labels = []

    # Legend entries (to avoid duplicates)
    legend_handles = []
    legend_labels_seen = set()

    for g_idx, group_name in enumerate(group_names):
        variants = ablation_groups[group_name]
        variant_names = list(variants.keys())
        n_variants = len(variant_names)

        bar_width = group_width / n_variants
        group_center = x_offset + group_width / 2

        for v_idx, var_name in enumerate(variant_names):
            val = variants[var_name].get(metric, 0)
            err = variants[var_name].get(f"{metric}_std", 0)

            x_pos = x_offset + v_idx * bar_width + bar_width / 2

            bar = ax.bar(
                x_pos, val,
                width=bar_width * 0.85,
                yerr=err if err > 0 else None,
                capsize=3,
                color=colors[v_idx % len(colors)],
                edgecolor="black", linewidth=0.5,
                label=var_name if var_name not in legend_labels_seen else None,
            )

            # Value label on bar
            ax.text(
                x_pos, val + (err if err > 0 else 0) + 0.2,
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=6.5,
            )

            xtick_positions.append(x_pos)
            xtick_labels.append(var_name)

        # Group label below
        ax.text(
            group_center, ax.get_ylim()[0] - 1.5,
            group_name,
            ha="center", va="top", fontsize=8, fontweight="bold",
            style="italic",
        )

        x_offset += group_width + 0.6  # gap between groups

    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{metric.replace('_', ' ').title()} (%)")
    ax.set_title("Figure 14: Ablation Study Results")
    ax.grid(axis="y", alpha=0.3)

    # Add group separator lines
    x_sep = 0.0
    for g_idx, group_name in enumerate(group_names[:-1]):
        n_v = len(ablation_groups[group_name])
        x_sep += group_width + 0.3  # middle of gap
        ax.axvline(x_sep, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        x_sep += 0.3

    fig.tight_layout()
    _save(fig, save_dir, "fig14_ablation_bars")
    return fig


# ===================================================================
# Figure 15: Rank vs Accuracy vs Parameters Scatter Plot
# ===================================================================

def plot_rank_accuracy_params(
    rank_results: Dict[int, Dict[str, float]],
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 15 -- scatter plot: params vs accuracy, labeled by rank R.

    Args:
        rank_results: Dict mapping rank R -> results dict::

            {
                1: {"accuracy": 93.5, "params": 4.2e6},
                2: {"accuracy": 94.1, "params": 5.1e6},
                4: {"accuracy": 95.2, "params": 8.0e6},
                8: {"accuracy": 95.4, "params": 14.0e6},
            }

        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    fig, ax = plt.subplots(figsize=(7, 5))

    ranks = sorted(rank_results.keys())
    params_list = [rank_results[r]["params"] for r in ranks]
    acc_list = [rank_results[r]["accuracy"] for r in ranks]

    # Scatter
    scatter = ax.scatter(
        params_list, acc_list,
        s=120, c=ranks, cmap="viridis",
        edgecolors="black", linewidths=0.8, zorder=5,
    )

    # Connect with a line
    ax.plot(params_list, acc_list, "k--", alpha=0.4, linewidth=1, zorder=4)

    # Annotate each point with rank
    for r, p, a in zip(ranks, params_list, acc_list):
        ax.annotate(
            f"R={r}",
            (p, a),
            textcoords="offset points",
            xytext=(8, 8),
            fontsize=9,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5),
        )

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, label="Rank $R$", shrink=0.8)

    ax.set_xlabel("Total Parameters")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Figure 15: Rank vs Accuracy vs Parameters")
    ax.grid(True, alpha=0.3)

    # Format x-axis as millions
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
    )

    fig.tight_layout()
    _save(fig, save_dir, "fig15_rank_accuracy_params")
    return fig


# ===================================================================
# Convenience: build ablation groups from flat results
# ===================================================================

def build_ablation_groups_from_results(
    results: Dict[str, Dict[str, float]],
    group_assignments: Dict[str, List[str]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Organize flat results into grouped structure for plot_ablation_bar_chart.

    Args:
        results: {variant_name: {metric_name: value}}.
        group_assignments: {group_name: [variant_name, ...]}.

    Returns:
        Nested dict suitable for plot_ablation_bar_chart.
    """
    groups: Dict[str, Dict[str, Dict[str, float]]] = {}
    for group_name, variant_names in group_assignments.items():
        groups[group_name] = {}
        for vn in variant_names:
            if vn in results:
                groups[group_name][vn] = results[vn]
    return groups


# ===================================================================
# Example ablation data (for testing / demonstration)
# ===================================================================

def get_example_ablation_data() -> Dict[str, Dict[str, Dict[str, float]]]:
    """Return example ablation results for demonstration purposes."""
    return {
        "Static vs Dynamic": {
            "DKA-Static": {"accuracy": 91.5, "accuracy_std": 0.3},
            "DKA-Full": {"accuracy": 95.2, "accuracy_std": 0.2},
        },
        "Rank": {
            "R=1": {"accuracy": 93.5, "accuracy_std": 0.4},
            "R=2": {"accuracy": 94.1, "accuracy_std": 0.3},
            "R=4": {"accuracy": 95.2, "accuracy_std": 0.2},
            "R=8": {"accuracy": 95.4, "accuracy_std": 0.3},
        },
        "Kernel Size": {
            "All k=3": {"accuracy": 93.8, "accuracy_std": 0.3},
            "All k=7": {"accuracy": 94.5, "accuracy_std": 0.2},
            "All k=11": {"accuracy": 94.2, "accuracy_std": 0.3},
            "Multi-scale": {"accuracy": 95.2, "accuracy_std": 0.2},
        },
        "Diversity Loss": {
            "$\\lambda$=0": {"accuracy": 94.6, "accuracy_std": 0.3},
            "$\\lambda$=0.1": {"accuracy": 95.2, "accuracy_std": 0.2},
            "$\\lambda$=0.5": {"accuracy": 94.9, "accuracy_std": 0.3},
        },
        "Residual Structure": {
            "No residual": {"accuracy": 93.1, "accuracy_std": 0.5},
            "Residual": {"accuracy": 95.2, "accuracy_std": 0.2},
            "Fixed $\\alpha$=1": {"accuracy": 94.3, "accuracy_std": 0.3},
        },
        "Generator": {
            "Linear": {"accuracy": 92.8, "accuracy_std": 0.4},
            "MLP (2-layer)": {"accuracy": 95.2, "accuracy_std": 0.2},
            "MLP (3-layer)": {"accuracy": 95.1, "accuracy_std": 0.3},
        },
    }


def get_example_rank_data() -> Dict[int, Dict[str, float]]:
    """Return example rank-vs-params data for demonstration purposes."""
    return {
        1: {"accuracy": 93.5, "params": 4.2e6},
        2: {"accuracy": 94.1, "params": 5.1e6},
        4: {"accuracy": 95.2, "params": 8.0e6},
        8: {"accuracy": 95.4, "params": 14.0e6},
    }


# ===================================================================
# Standalone entry point
# ===================================================================

def main(
    model: nn.Module = None,
    data: Any = None,
    save_dir: str = "figures",
    ablation_results: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
    rank_results: Optional[Dict[int, Dict[str, float]]] = None,
) -> None:
    """Generate ablation figures.

    If *ablation_results* and *rank_results* are None, uses example data
    for demonstration.

    Args:
        model: Not used (results are passed directly).
        data: Not used.
        save_dir: Output directory.
        ablation_results: Grouped ablation results for Figure 14.
        rank_results: Rank-vs-params results for Figure 15.
    """
    import torch.nn as nn  # local import for type hint

    if ablation_results is None:
        print("No ablation results provided; using example data for Figure 14.")
        ablation_results = get_example_ablation_data()

    if rank_results is None:
        print("No rank results provided; using example data for Figure 15.")
        rank_results = get_example_rank_data()

    plot_ablation_bar_chart(ablation_results, save_dir=save_dir)
    plot_rank_accuracy_params(rank_results, save_dir=save_dir)
