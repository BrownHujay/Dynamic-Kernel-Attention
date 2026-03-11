"""Efficiency analysis: FLOPs, throughput, and memory (Figures 16-18).

Figure 16: FLOPs vs sequence length -- theoretical curves from the
    formulas in Section 6.1 of the build guide.
Figure 17: Throughput comparison -- actual benchmarking of models.
Figure 18: Memory comparison -- peak GPU memory during forward/backward.

All figures saved as PNG + PDF.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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


# ===================================================================
# Theoretical FLOP counting (Section 6.1)
# ===================================================================

def flops_dka_per_layer(
    n: int,
    d: int = 256,
    H: int = 8,
    R: int = 4,
    kernel_sizes: Optional[List[int]] = None,
) -> float:
    """Compute theoretical FLOPs for one DKA layer.

    Formula from build guide section 6.1:
        FLOPs_DKA = n * sum_h [ R*(6*d_h^2 + 3*k_h*d_h) + k_h*d_h ] + 2*n*d^2

    Args:
        n: Sequence length.
        d: Model dimension.
        H: Number of heads.
        R: Rank.
        kernel_sizes: List of kernel sizes per head.

    Returns:
        Total multiply-add operations (FLOPs).
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 5, 5, 7, 7, 11, 11]
    d_h = d // H
    total = 0.0
    for k_h in kernel_sizes:
        per_head = R * (6 * d_h**2 + 3 * k_h * d_h) + k_h * d_h
        total += per_head
    total = n * total + 2 * n * d**2
    return total


def flops_attention_per_layer(
    n: int,
    d: int = 256,
) -> float:
    """Compute theoretical FLOPs for one standard attention layer.

    Formula: FLOPs_Attn = 2*n^2*d + 4*n*d^2

    Args:
        n: Sequence length.
        d: Model dimension.

    Returns:
        Total multiply-add operations.
    """
    return 2.0 * n**2 * d + 4.0 * n * d**2


def flops_linear_attention_per_layer(
    n: int,
    d: int = 256,
) -> float:
    """Approximate FLOPs for linear attention (for reference).

    Linear attention avoids the n^2 term:
        FLOPs ~ 4*n*d^2 + 2*n*d  (feature map + aggregation)
    """
    return 4.0 * n * d**2 + 2.0 * n * d


# ===================================================================
# Memory estimates
# ===================================================================

def memory_dka_activations(
    n: int,
    d: int = 256,
    H: int = 8,
    kernel_sizes: Optional[List[int]] = None,
    bytes_per_float: int = 4,
) -> float:
    """Estimate DKA activation memory (generated kernels + windows).

    Memory = n * sum_h(k_h * d_h) floats for kernels,
             plus n * sum_h(k_h * d_h) floats for windows.

    Returns bytes.
    """
    if kernel_sizes is None:
        kernel_sizes = [3, 3, 5, 5, 7, 7, 11, 11]
    d_h = d // H
    total_floats = 0.0
    for k_h in kernel_sizes:
        total_floats += 2.0 * n * k_h * d_h  # kernels + windows
    return total_floats * bytes_per_float


def memory_attention_activations(
    n: int,
    d: int = 256,
    H: int = 8,
    bytes_per_float: int = 4,
) -> float:
    """Estimate standard attention activation memory.

    Stores H attention matrices of size n x n, plus QKV tensors.
    Memory ~ H * n^2 + 3*n*d  (attention maps + QKV).

    Returns bytes.
    """
    total_floats = float(H) * n**2 + 3.0 * n * d
    return total_floats * bytes_per_float


# ===================================================================
# Figure 16: FLOPs vs Sequence Length
# ===================================================================

def plot_flops_vs_seqlen(
    d: int = 256,
    H: int = 8,
    R: int = 4,
    kernel_sizes: Optional[List[int]] = None,
    seq_lengths: Optional[List[int]] = None,
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 16 -- theoretical FLOPs vs sequence length (log-log).

    Lines for: Standard Attention, DKA (ours), Linear Attention.

    Args:
        d: Model dimension.
        H: Number of heads.
        R: DKA rank.
        kernel_sizes: Kernel sizes per head.
        seq_lengths: Sequence lengths to plot.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]

    ns = np.array(seq_lengths, dtype=float)
    flops_attn = np.array([flops_attention_per_layer(int(n), d) for n in ns])
    flops_dka = np.array([
        flops_dka_per_layer(int(n), d, H, R, kernel_sizes) for n in ns
    ])
    flops_lin = np.array([
        flops_linear_attention_per_layer(int(n), d) for n in ns
    ])

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(ns, flops_attn, "o-", label="Standard Attention $O(n^2 d)$",
              color="#d62728", linewidth=2, markersize=5)
    ax.loglog(ns, flops_dka, "s-", label="DKA (ours) $O(nkd)$",
              color="#1f77b4", linewidth=2, markersize=5)
    ax.loglog(ns, flops_lin, "^--", label="Linear Attention $O(nd^2)$",
              color="#2ca02c", linewidth=1.5, markersize=5, alpha=0.7)

    ax.set_xlabel("Sequence Length $n$")
    ax.set_ylabel("FLOPs per Layer")
    ax.set_title("Figure 16: FLOPs vs Sequence Length")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    # Annotate crossover
    for i in range(len(ns) - 1):
        if flops_dka[i] >= flops_attn[i] and flops_dka[i + 1] < flops_attn[i + 1]:
            ax.axvline(
                ns[i], color="gray", linestyle=":", alpha=0.5,
                label=f"Crossover ~n={int(ns[i])}",
            )
            break

    fig.tight_layout()
    _save(fig, save_dir, "fig16_flops_vs_seqlen")
    return fig


# ===================================================================
# Figure 17: Throughput Comparison (actual benchmarking)
# ===================================================================

def _warmup_and_benchmark(
    model: nn.Module,
    input_fn: Callable[[], torch.Tensor],
    num_warmup: int = 10,
    num_iters: int = 50,
    device: str = "cuda",
) -> Tuple[float, float]:
    """Benchmark forward-pass throughput.

    Returns:
        (mean_time_seconds, std_time_seconds) per forward pass.
    """
    model.eval()
    model.to(device)

    times = []

    # Warmup
    for _ in range(num_warmup):
        x = input_fn().to(device)
        with torch.no_grad():
            _ = model(x)

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()

    # Benchmark
    for _ in range(num_iters):
        x = input_fn().to(device)

        if device == "cuda" and torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = model(x)
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0  # ms -> s
        else:
            t0 = time.perf_counter()
            with torch.no_grad():
                _ = model(x)
            elapsed = time.perf_counter() - t0

        times.append(elapsed)

    return float(np.mean(times)), float(np.std(times))


def benchmark_throughput(
    models: Dict[str, nn.Module],
    input_fn: Callable[[], torch.Tensor],
    batch_size: int,
    num_warmup: int = 10,
    num_iters: int = 50,
    device: str = "cuda",
) -> Dict[str, float]:
    """Benchmark multiple models and return throughput (samples/sec).

    Args:
        models: {name: model} dict.
        input_fn: Callable that returns one batch of input.
        batch_size: Batch size used by input_fn (for throughput calc).
        num_warmup: Warmup iterations.
        num_iters: Benchmark iterations.
        device: Device string.

    Returns:
        {model_name: throughput_samples_per_sec}.
    """
    results: Dict[str, float] = {}
    for name, model in models.items():
        mean_t, std_t = _warmup_and_benchmark(
            model, input_fn, num_warmup, num_iters, device,
        )
        throughput = batch_size / mean_t if mean_t > 0 else 0
        results[name] = throughput
        print(f"  {name}: {throughput:.1f} samples/sec "
              f"(mean={mean_t*1000:.2f}ms, std={std_t*1000:.2f}ms)")
    return results


def plot_throughput(
    throughput_results: Dict[str, float],
    save_dir: str = "figures",
    title_suffix: str = "",
) -> plt.Figure:
    """Figure 17 -- throughput bar chart.

    Args:
        throughput_results: {model_name: samples_per_sec}.
        save_dir: Output directory.
        title_suffix: Extra text for the title (e.g., "batch_size=128").

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    names = list(throughput_results.keys())
    values = list(throughput_results.values())

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4.5))

    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    bars = ax.bar(
        range(len(names)), values,
        color=colors, edgecolor="black", linewidth=0.5,
    )

    # Add value labels on bars
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.0f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Throughput (samples/sec)")
    ax.set_title(f"Figure 17: Throughput Comparison{title_suffix}")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir, "fig17_throughput")
    return fig


# ===================================================================
# Figure 18: Memory Comparison
# ===================================================================

def measure_peak_memory(
    model: nn.Module,
    input_fn: Callable[[], torch.Tensor],
    device: str = "cuda",
    include_backward: bool = True,
) -> float:
    """Measure peak GPU memory during forward (+ optional backward) pass.

    Args:
        model: Model to measure.
        input_fn: Returns one batch of input.
        device: Device string.
        include_backward: If True, also run backward pass.

    Returns:
        Peak memory in MB. Returns 0 if CUDA is not available.
    """
    if device != "cuda" or not torch.cuda.is_available():
        return 0.0

    model.to(device)
    model.train()

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    x = input_fn().to(device)
    out = model(x)

    if include_backward:
        if out.dim() > 1:
            loss = out.sum()
        else:
            loss = out
        loss.backward()

    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_mb = peak_bytes / (1024 ** 2)

    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    return peak_mb


def plot_memory_comparison(
    memory_results: Dict[str, float],
    save_dir: str = "figures",
) -> plt.Figure:
    """Figure 18 -- peak GPU memory bar chart.

    Args:
        memory_results: {model_name: peak_memory_MB}.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    names = list(memory_results.keys())
    values = list(memory_results.values())

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.5), 4.5))

    colors = plt.cm.Pastel1(np.linspace(0, 1, len(names)))
    bars = ax.bar(
        range(len(names)), values,
        color=colors, edgecolor="black", linewidth=0.5,
    )

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.0f} MB",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Peak GPU Memory (MB)")
    ax.set_title("Figure 18: Peak Memory Comparison")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    _save(fig, save_dir, "fig18_memory")
    return fig


def plot_memory_vs_seqlen(
    d: int = 256,
    H: int = 8,
    kernel_sizes: Optional[List[int]] = None,
    seq_lengths: Optional[List[int]] = None,
    save_dir: str = "figures",
) -> plt.Figure:
    """Plot theoretical memory vs sequence length (DKA vs Attention).

    Args:
        d: Model dimension.
        H: Number of heads.
        kernel_sizes: Kernel sizes per head.
        seq_lengths: Sequence lengths to plot.
        save_dir: Output directory.

    Returns:
        matplotlib Figure.
    """
    _apply_rc()

    if seq_lengths is None:
        seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096]

    ns = np.array(seq_lengths, dtype=float)
    mem_attn = np.array([
        memory_attention_activations(int(n), d, H) for n in ns
    ]) / (1024 ** 2)  # bytes -> MB
    mem_dka = np.array([
        memory_dka_activations(int(n), d, H, kernel_sizes) for n in ns
    ]) / (1024 ** 2)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(ns, mem_attn, "o-", label="Standard Attention",
              color="#d62728", linewidth=2, markersize=5)
    ax.loglog(ns, mem_dka, "s-", label="DKA (ours)",
              color="#1f77b4", linewidth=2, markersize=5)

    ax.set_xlabel("Sequence Length $n$")
    ax.set_ylabel("Activation Memory (MB)")
    ax.set_title("Figure 18b: Memory vs Sequence Length (Theoretical)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    _save(fig, save_dir, "fig18b_memory_vs_seqlen")
    return fig


# ===================================================================
# Standalone entry point
# ===================================================================

def main(
    model: nn.Module = None,
    data: torch.Tensor = None,
    save_dir: str = "figures",
    models_dict: Optional[Dict[str, nn.Module]] = None,
    input_fn: Optional[Callable[[], torch.Tensor]] = None,
    batch_size: int = 128,
    d: int = 256,
    H: int = 8,
    R: int = 4,
    kernel_sizes: Optional[List[int]] = None,
    device: str = "cuda",
) -> None:
    """Generate all efficiency figures.

    Figure 16 is always generated (theoretical, no model needed).
    Figures 17 and 18 require *models_dict* and *input_fn* for actual
    benchmarking; otherwise they are skipped.

    Args:
        model: Primary DKA model (optional, used if models_dict is None).
        data: Sample input tensor (optional).
        save_dir: Output directory.
        models_dict: {name: model} for benchmarking.
        input_fn: Callable returning one batch of input.
        batch_size: Batch size used by input_fn.
        d: Model dimension for theoretical curves.
        H: Number of heads for theoretical curves.
        R: Rank for theoretical curves.
        kernel_sizes: Kernel sizes for theoretical curves.
        device: Device for benchmarking.
    """
    # Figure 16: theoretical FLOP curves (always)
    plot_flops_vs_seqlen(d=d, H=H, R=R, kernel_sizes=kernel_sizes, save_dir=save_dir)

    # Figure 18b: theoretical memory curves (always)
    plot_memory_vs_seqlen(d=d, H=H, kernel_sizes=kernel_sizes, save_dir=save_dir)

    # Figures 17 & 18: actual benchmarking (if models provided)
    if models_dict is not None and input_fn is not None:
        print("\n=== Throughput Benchmark ===")
        throughput = benchmark_throughput(
            models_dict, input_fn, batch_size,
            device=device,
        )
        plot_throughput(throughput, save_dir=save_dir)

        if device == "cuda" and torch.cuda.is_available():
            print("\n=== Memory Benchmark ===")
            memory_results: Dict[str, float] = {}
            for name, m in models_dict.items():
                peak = measure_peak_memory(m, input_fn, device=device)
                memory_results[name] = peak
                print(f"  {name}: {peak:.1f} MB peak")
            plot_memory_comparison(memory_results, save_dir=save_dir)
        else:
            print("CUDA not available -- skipping memory benchmarks.")
    elif model is not None:
        print(
            "Only one model provided. Pass models_dict={name: model} "
            "and input_fn for full throughput/memory benchmarks."
        )
