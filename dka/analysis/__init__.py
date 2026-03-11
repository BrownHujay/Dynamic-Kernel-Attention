"""Analysis and visualization modules for Dynamic Kernel Attention (DKA).

Provides tools for visualizing generated kernels, computing spectral analysis,
tracking alpha trajectories, measuring kernel diversity and dynamism, running
efficiency benchmarks, and generating publication-quality ablation plots.

Usage:
    from dka.analysis import (
        kernel_viz,
        attention_maps,
        spectral,
        alpha_tracking,
        diversity_metrics,
        dynamism,
        polysemy,
        efficiency,
        ablation_plots,
    )
"""

from . import kernel_viz
from . import attention_maps
from . import spectral
from . import alpha_tracking
from . import diversity_metrics
from . import dynamism
from . import polysemy
from . import efficiency
from . import ablation_plots

__all__ = [
    "kernel_viz",
    "attention_maps",
    "spectral",
    "alpha_tracking",
    "diversity_metrics",
    "dynamism",
    "polysemy",
    "efficiency",
    "ablation_plots",
]
