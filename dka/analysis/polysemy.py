"""Polysemous word kernel comparison -- the 'Bank' figure (Figure 10).

Given two sentences containing the same word in different semantic
contexts, extract the DKA-generated kernels for that word and display
them as side-by-side heatmaps.  If the kernels differ meaningfully, the
model has learned word-sense disambiguation at the kernel level.

Supports multiple word pairs in a single figure.

All figures saved as PNG + PDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
# Kernel extraction for a specific token position
# ---------------------------------------------------------------------------

def extract_kernel_at_position(
    model: nn.Module,
    x: torch.Tensor,
    token_pos: int,
    layer_idx: int = 0,
    head_idx: int = 0,
    batch_idx: int = 0,
) -> np.ndarray:
    """Run a forward pass and extract the kernel for one token position.

    Args:
        model: DKA text model.
        x: Input tensor (B, n, d) or raw model input.
        token_pos: Index of the target token in the sequence.
        layer_idx: DKA layer to extract from.
        head_idx: Head to extract from.
        batch_idx: Batch element.

    Returns:
        (k_h, d_h) numpy array -- the generated kernel for that token.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    dka = dka_modules[layer_idx]
    kernels = dka.get_last_kernels()
    # kernels[head_idx]: (B, n, k_h, d_h)
    return kernels[head_idx][batch_idx, token_pos].cpu().numpy()


def extract_kernels_all_heads(
    model: nn.Module,
    x: torch.Tensor,
    token_pos: int,
    layer_idx: int = 0,
    batch_idx: int = 0,
) -> Dict[int, np.ndarray]:
    """Extract kernels at *token_pos* for all heads.

    Returns:
        {head_idx: (k_h, d_h) numpy array}.
    """
    model.eval()
    with torch.no_grad():
        _ = model(x)

    dka_modules = _get_dka_modules(model)
    dka = dka_modules[layer_idx]
    kernels = dka.get_last_kernels()

    result: Dict[int, np.ndarray] = {}
    for h, k_tensor in kernels.items():
        result[h] = k_tensor[batch_idx, token_pos].cpu().numpy()
    return result


# ---------------------------------------------------------------------------
# Tokenization helpers
# ---------------------------------------------------------------------------

def find_token_position(
    tokens: List[str],
    target_word: str,
) -> Optional[int]:
    """Find the index of *target_word* in *tokens* (case-insensitive).

    If the word appears multiple times, returns the first occurrence.
    Returns None if not found.
    """
    target_lower = target_word.lower()
    for i, t in enumerate(tokens):
        # Strip common subword prefixes/suffixes
        clean = t.strip().lower().strip(".,!?;:'\"")
        if clean == target_lower:
            return i
    return None


# ---------------------------------------------------------------------------
# Figure 10: Polysemy comparison
# ---------------------------------------------------------------------------

def plot_polysemy_comparison(
    model: nn.Module,
    sentence_pairs: List[Tuple[torch.Tensor, torch.Tensor, List[str], List[str], str]],
    layer_idx: int = 0,
    head_idx: int = 0,
    batch_idx: int = 0,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 10 -- compare kernels for polysemous words in different contexts.

    Args:
        model: Trained DKA text model.
        sentence_pairs: List of tuples, each containing:
            - x1: input tensor for sentence 1 (B, n, d)
            - x2: input tensor for sentence 2 (B, n, d)
            - tokens1: list of string tokens for sentence 1
            - tokens2: list of string tokens for sentence 2
            - target_word: the polysemous word to compare
        layer_idx: DKA layer.
        head_idx: Head to visualize.
        batch_idx: Batch element.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    num_pairs = len(sentence_pairs)
    # 3 columns per pair: kernel1, kernel2, difference
    ncols = 3
    nrows = num_pairs

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(3.5 * ncols, 2.5 * nrows),
        squeeze=False,
    )

    for row, (x1, x2, tokens1, tokens2, word) in enumerate(sentence_pairs):
        pos1 = find_token_position(tokens1, word)
        pos2 = find_token_position(tokens2, word)

        if pos1 is None or pos2 is None:
            for c in range(ncols):
                axes[row, c].text(
                    0.5, 0.5,
                    f"'{word}' not found\nin tokens",
                    ha="center", va="center", fontsize=10,
                    transform=axes[row, c].transAxes,
                )
                axes[row, c].axis("off")
            continue

        k1 = extract_kernel_at_position(
            model, x1, pos1, layer_idx, head_idx, batch_idx,
        )
        k2 = extract_kernel_at_position(
            model, x2, pos2, layer_idx, head_idx, batch_idx,
        )
        diff = k1 - k2

        vmax = max(np.abs(k1).max(), np.abs(k2).max())

        # Sentence 1 kernel
        sns.heatmap(
            k1, ax=axes[row, 0], cmap="RdBu_r", center=0,
            vmin=-vmax, vmax=vmax,
            cbar=False, xticklabels=False, yticklabels=False,
        )
        s1_short = " ".join(tokens1[:8]) + ("..." if len(tokens1) > 8 else "")
        axes[row, 0].set_title(f'"{word}" in:\n{s1_short}', fontsize=7)
        if row == 0:
            axes[row, 0].text(
                0.5, 1.25, "Context A",
                ha="center", fontsize=9, fontweight="bold",
                transform=axes[row, 0].transAxes,
            )

        # Sentence 2 kernel
        sns.heatmap(
            k2, ax=axes[row, 1], cmap="RdBu_r", center=0,
            vmin=-vmax, vmax=vmax,
            cbar=False, xticklabels=False, yticklabels=False,
        )
        s2_short = " ".join(tokens2[:8]) + ("..." if len(tokens2) > 8 else "")
        axes[row, 1].set_title(f'"{word}" in:\n{s2_short}', fontsize=7)
        if row == 0:
            axes[row, 1].text(
                0.5, 1.25, "Context B",
                ha="center", fontsize=9, fontweight="bold",
                transform=axes[row, 1].transAxes,
            )

        # Difference
        diff_max = np.abs(diff).max()
        sns.heatmap(
            diff, ax=axes[row, 2], cmap="PiYG", center=0,
            vmin=-diff_max if diff_max > 0 else -1,
            vmax=diff_max if diff_max > 0 else 1,
            cbar=True, xticklabels=False, yticklabels=False,
            cbar_kws={"shrink": 0.8},
        )
        # Compute cosine similarity
        flat1 = k1.flatten()
        flat2 = k2.flatten()
        norm1 = np.linalg.norm(flat1)
        norm2 = np.linalg.norm(flat2)
        if norm1 > 0 and norm2 > 0:
            cos_sim = np.dot(flat1, flat2) / (norm1 * norm2)
        else:
            cos_sim = 1.0
        axes[row, 2].set_title(
            f"Difference\ncos_sim={cos_sim:.3f}", fontsize=8,
        )
        if row == 0:
            axes[row, 2].text(
                0.5, 1.25, "A - B",
                ha="center", fontsize=9, fontweight="bold",
                transform=axes[row, 2].transAxes,
            )

    fig.suptitle(
        f"Figure 10: Polysemous Word Kernels "
        f"(Layer {layer_idx}, Head {head_idx})",
        fontsize=12, y=1.05,
    )
    fig.tight_layout()
    _save(fig, save_dir, "fig10_polysemy")
    return fig


# ---------------------------------------------------------------------------
# Multi-head polysemy view
# ---------------------------------------------------------------------------

def plot_polysemy_all_heads(
    model: nn.Module,
    x1: torch.Tensor,
    x2: torch.Tensor,
    tokens1: List[str],
    tokens2: List[str],
    target_word: str,
    layer_idx: int = 0,
    batch_idx: int = 0,
    save_dir: str = "figures",
) -> Optional[plt.Figure]:
    """Compare kernels for one polysemous word across all heads.

    Shows a row of heatmaps: head 0..H-1, with sentence 1 on top row
    and sentence 2 on bottom row.
    """
    _apply_rc()

    pos1 = find_token_position(tokens1, target_word)
    pos2 = find_token_position(tokens2, target_word)
    if pos1 is None or pos2 is None:
        print(f"'{target_word}' not found in one or both token lists.")
        return None

    kernels1 = extract_kernels_all_heads(model, x1, pos1, layer_idx, batch_idx)
    kernels2 = extract_kernels_all_heads(model, x2, pos2, layer_idx, batch_idx)

    num_heads = len(kernels1)

    fig, axes = plt.subplots(
        2, num_heads,
        figsize=(2.2 * num_heads, 4.5),
        squeeze=False,
    )

    dka_modules = _get_dka_modules(model)
    ksizes = dka_modules[layer_idx].get_kernel_sizes()

    for h in range(num_heads):
        vmax = max(np.abs(kernels1[h]).max(), np.abs(kernels2[h]).max())

        sns.heatmap(
            kernels1[h], ax=axes[0, h], cmap="RdBu_r", center=0,
            vmin=-vmax, vmax=vmax,
            cbar=False, xticklabels=False, yticklabels=False,
        )
        axes[0, h].set_title(f"H{h} (k={ksizes[h]})", fontsize=8)

        sns.heatmap(
            kernels2[h], ax=axes[1, h], cmap="RdBu_r", center=0,
            vmin=-vmax, vmax=vmax,
            cbar=False, xticklabels=False, yticklabels=False,
        )

    axes[0, 0].set_ylabel("Context A", fontsize=9)
    axes[1, 0].set_ylabel("Context B", fontsize=9)

    s1_short = " ".join(tokens1[:6]) + "..."
    s2_short = " ".join(tokens2[:6]) + "..."
    fig.suptitle(
        f'"{target_word}" -- Layer {layer_idx}\n'
        f"A: {s1_short}\nB: {s2_short}",
        fontsize=10, y=1.08,
    )
    fig.tight_layout()
    _save(fig, save_dir, f"fig10_polysemy_{target_word}_all_heads")
    return fig


# ---------------------------------------------------------------------------
# Convenience: default word pairs
# ---------------------------------------------------------------------------

DEFAULT_WORD_PAIRS = [
    (
        "I went to the bank to deposit money",
        "I sat on the river bank and watched the water",
        "bank",
    ),
    (
        "The bat flew out of the cave at dusk",
        "He swung the bat and hit a home run",
        "bat",
    ),
    (
        "Spring is the best season for flowers",
        "The spring in the mattress broke",
        "spring",
    ),
    (
        "The crane lifted the heavy steel beam",
        "A crane stood by the pond on one leg",
        "crane",
    ),
    (
        "Turn left at the next intersection",
        "He left the room without saying goodbye",
        "left",
    ),
]


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def main(
    model: nn.Module,
    data: torch.Tensor = None,
    tokenize_fn=None,
    layer_idx: int = 0,
    head_idx: int = 0,
    save_dir: str = "figures",
    word_pairs: Optional[List[Tuple[str, str, str]]] = None,
) -> None:
    """Generate the polysemy comparison figure.

    Args:
        model: Trained DKA text model.
        data: Not directly used (sentences are tokenized via *tokenize_fn*).
        tokenize_fn: Callable(sentence: str) -> (tensor, list[str]).
            Must return (input_tensor of shape (1, n, d), list of token
            strings).  If None, only a placeholder message is printed.
        layer_idx: DKA layer.
        head_idx: Head for the main Figure 10.
        save_dir: Output directory.
        word_pairs: List of (sentence1, sentence2, target_word). If None,
            uses DEFAULT_WORD_PAIRS.
    """
    if tokenize_fn is None:
        print(
            "polysemy.main requires a tokenize_fn that converts a sentence "
            "to (input_tensor, token_list). Skipping."
        )
        return

    if word_pairs is None:
        word_pairs = DEFAULT_WORD_PAIRS

    sentence_pair_data = []
    for s1, s2, word in word_pairs:
        x1, tok1 = tokenize_fn(s1)
        x2, tok2 = tokenize_fn(s2)
        sentence_pair_data.append((x1, x2, tok1, tok2, word))

    plot_polysemy_comparison(
        model, sentence_pair_data,
        layer_idx=layer_idx, head_idx=head_idx,
        save_dir=save_dir,
    )

    # Also do per-word all-heads views
    for x1, x2, tok1, tok2, word in sentence_pair_data:
        plot_polysemy_all_heads(
            model, x1, x2, tok1, tok2, word,
            layer_idx=layer_idx, save_dir=save_dir,
        )
