"""Optimizer configuration for Dynamic Kernel Attention training.

Creates an AdamW optimizer with three parameter groups and differentiated
learning rates, weight decay, and gradient clipping:

  1. Main parameters (embeddings, FFN, LayerNorm, head projections, K_base):
     lr=3e-4, weight_decay=0.05 (biases/LN: wd=0.0)
  2. Kernel generators (W_1s, W_2s, W_1c, W_2c and biases):
     lr=3e-5, weight_decay=0.05 (biases: wd=0.0)
  3. Alpha scalars: lr=1e-3, weight_decay=0.0

Reference: DKA Build Guide, Sections 5.2 and 10.3.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Parameter classification helpers
# ---------------------------------------------------------------------------

def _is_kernel_generator_param(name: str) -> bool:
    """Return True if the parameter belongs to a kernel generator MLP.

    Matches names containing kernel generator weight/bias identifiers:
    spatial_fc1, spatial_fc2, channel_fc1, channel_fc2.
    """
    kg_keywords = ("spatial_fc1", "spatial_fc2", "channel_fc1", "channel_fc2")
    return any(kw in name for kw in kg_keywords)


def _is_alpha_param(name: str) -> bool:
    """Return True if the parameter is an alpha scalar."""
    # Match 'alpha' as a leaf parameter name in kernel generators.
    # Handles both nested (e.g. 'blocks.0.dka.kernel_generators.0.alpha')
    # and top-level (e.g. 'alpha') parameter names.
    return name.endswith(".alpha") or name == "alpha"


def _is_bias_or_layernorm(name: str, param: torch.Tensor) -> bool:
    """Return True if the parameter is a bias or belongs to LayerNorm.

    LayerNorm parameters (weight and bias) and all bias vectors should
    receive zero weight decay.
    """
    if name.endswith(".bias"):
        return True
    # LayerNorm weight (gamma) -- identified by 'norm' in the name
    # and being a 1-D parameter
    if ("norm" in name.lower()) and param.ndim == 1:
        return True
    return False


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    lr_main: float = 3e-4,
    lr_kernel_gen: float = 3e-5,
    lr_alpha: float = 1e-3,
    weight_decay: float = 0.05,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with DKA-specific parameter groups.

    Args:
        model: The DKA model (any nn.Module).
        lr_main: Learning rate for main parameters.
        lr_kernel_gen: Learning rate for kernel generator parameters.
        lr_alpha: Learning rate for alpha scalars.
        weight_decay: Default weight decay (applied to non-bias, non-LN params).
        betas: Adam beta coefficients.
        eps: Adam epsilon.

    Returns:
        Configured AdamW optimizer with three parameter groups.
    """
    # Separate parameters into groups
    main_decay: List[torch.Tensor] = []
    main_no_decay: List[torch.Tensor] = []
    kg_decay: List[torch.Tensor] = []
    kg_no_decay: List[torch.Tensor] = []
    alpha_params: List[torch.Tensor] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if _is_alpha_param(name):
            alpha_params.append(param)
        elif _is_kernel_generator_param(name):
            if _is_bias_or_layernorm(name, param):
                kg_no_decay.append(param)
            else:
                kg_decay.append(param)
        else:
            if _is_bias_or_layernorm(name, param):
                main_no_decay.append(param)
            else:
                main_decay.append(param)

    param_groups = [
        {
            "params": main_decay,
            "lr": lr_main,
            "weight_decay": weight_decay,
            "group_name": "main_decay",
        },
        {
            "params": main_no_decay,
            "lr": lr_main,
            "weight_decay": 0.0,
            "group_name": "main_no_decay",
        },
        {
            "params": kg_decay,
            "lr": lr_kernel_gen,
            "weight_decay": weight_decay,
            "group_name": "kernel_gen_decay",
        },
        {
            "params": kg_no_decay,
            "lr": lr_kernel_gen,
            "weight_decay": 0.0,
            "group_name": "kernel_gen_no_decay",
        },
        {
            "params": alpha_params,
            "lr": lr_alpha,
            "weight_decay": 0.0,
            "group_name": "alpha",
        },
    ]

    # Filter out empty groups
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    return optimizer


# ---------------------------------------------------------------------------
# Per-group gradient clipping
# ---------------------------------------------------------------------------

def clip_grad_norms(
    model: nn.Module,
    max_norm_global: float = 1.0,
    max_norm_kernel_gen: float = 0.5,
) -> Dict[str, float]:
    """Apply per-group gradient clipping.

    1. Clip kernel generator parameters at max_norm_kernel_gen (tighter).
    2. Clip all parameters globally at max_norm_global.

    This order ensures kernel generator params get their own tighter clip
    before the global clip is applied.

    Args:
        model: The DKA model.
        max_norm_global: Global gradient norm clipping threshold.
        max_norm_kernel_gen: Gradient norm clipping threshold for kernel
            generator parameters specifically.

    Returns:
        Dict with gradient norm info for logging:
            'grad_norm_global': Total gradient norm of all parameters.
            'grad_norm_kernel_gen': Total gradient norm of kernel gen params
                (before clipping).
            'grad_norm_main': Total gradient norm of non-kernel-gen params
                (before clipping).
    """
    kg_params = []
    other_params = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            if _is_kernel_generator_param(name):
                kg_params.append(param)
            else:
                other_params.append(param)

    # Compute norms before clipping (for logging)
    kg_norm = _compute_grad_norm(kg_params)
    other_norm = _compute_grad_norm(other_params)

    # Step 1: Clip kernel generator params at tighter threshold
    if kg_params:
        torch.nn.utils.clip_grad_norm_(kg_params, max_norm_kernel_gen)

    # Step 2: Global clip on all params
    all_params = [p for p in model.parameters() if p.grad is not None]
    if all_params:
        torch.nn.utils.clip_grad_norm_(all_params, max_norm_global)

    return {
        "grad_norm_global": kg_norm + other_norm,
        "grad_norm_kernel_gen": kg_norm,
        "grad_norm_main": other_norm,
    }


def _compute_grad_norm(params: List[torch.Tensor]) -> float:
    """Compute the total L2 gradient norm for a list of parameters."""
    if not params:
        return 0.0
    total_norm_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_norm_sq += p.grad.data.norm(2).item() ** 2
    return total_norm_sq ** 0.5
