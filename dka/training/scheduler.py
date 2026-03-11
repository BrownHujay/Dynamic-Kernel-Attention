"""Learning rate scheduler for Dynamic Kernel Attention training.

Linear warmup for the first max(10 epochs, 5% of total steps), then cosine
annealing to eta_min=1e-5 for the remainder.

Compatible with PyTorch's LR scheduler interface (step-based).

Reference: DKA Build Guide, Section 5.2.
"""

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_epochs: int = 10,
    steps_per_epoch: int = 1,
    eta_min: float = 1e-5,
) -> LambdaLR:
    """Build a warmup + cosine annealing LR scheduler.

    The warmup phase covers the first max(warmup_epochs * steps_per_epoch,
    0.05 * total_steps) steps with linear warmup from 0 to the base LR.
    The remaining steps use cosine annealing down to eta_min.

    Args:
        optimizer: The optimizer whose LR groups will be scheduled.
        total_steps: Total number of training steps (epochs * steps_per_epoch).
        warmup_epochs: Number of warmup epochs (default 10).
        steps_per_epoch: Number of optimizer steps per epoch.
        eta_min: Minimum learning rate at end of cosine annealing.

    Returns:
        A LambdaLR scheduler that should be stepped once per optimizer step.
    """
    warmup_from_epochs = warmup_epochs * steps_per_epoch
    warmup_from_fraction = int(0.05 * total_steps)
    warmup_steps = max(warmup_from_epochs, warmup_from_fraction)

    # Ensure warmup_steps doesn't exceed total steps
    warmup_steps = min(warmup_steps, total_steps)
    cosine_steps = total_steps - warmup_steps

    # Extract base LRs from each parameter group
    base_lrs = [group["lr"] for group in optimizer.param_groups]

    def lr_lambda_fn(current_step: int) -> float:
        """Compute LR multiplier for the given step.

        Returns a multiplier in [0, 1] that is applied to the base LR
        of each parameter group. Since different groups have different
        base LRs, the lambda returns a fraction of the base LR.
        """
        if current_step < warmup_steps:
            # Linear warmup: scale from 0 to 1
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing from 1.0 down to eta_min / base_lr
            # We return the multiplier; eta_min is handled per-group
            # by using the minimum ratio across groups
            progress = float(current_step - warmup_steps) / float(
                max(1, cosine_steps)
            )
            # Cosine schedule: starts at 1.0, ends near eta_min/base_lr
            # Since each group has a different base_lr, we use a generic
            # multiplier that cosine-decays to a small value.
            # The final LR = base_lr * multiplier, and we want the minimum
            # to be eta_min. We compute: mult = eta_ratio + (1 - eta_ratio) * cosine
            # But eta_min / base_lr differs per group, so we use the overall
            # cosine multiplier and handle eta_min as a floor.
            cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
            return cosine_mult

    # Use a per-group lambda to correctly handle eta_min for each group
    def make_lambda(base_lr: float):
        eta_ratio = eta_min / base_lr if base_lr > 0 else 0.0

        def _fn(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(
                    max(1, cosine_steps)
                )
                cosine_mult = 0.5 * (1.0 + math.cos(math.pi * progress))
                # Interpolate between eta_ratio and 1.0
                return eta_ratio + (1.0 - eta_ratio) * cosine_mult

        return _fn

    lambdas = [make_lambda(blr) for blr in base_lrs]
    scheduler = LambdaLR(optimizer, lr_lambda=lambdas)
    return scheduler


class WarmupCosineScheduler:
    """Alternative class-based scheduler with the same behavior.

    This wraps the lambda-based scheduler and provides convenience methods
    for inspecting the current state.

    Args:
        optimizer: Optimizer to schedule.
        total_steps: Total training steps.
        warmup_epochs: Number of warmup epochs.
        steps_per_epoch: Steps per epoch.
        eta_min: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        warmup_epochs: int = 10,
        steps_per_epoch: int = 1,
        eta_min: float = 1e-5,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.eta_min = eta_min

        self._scheduler = build_scheduler(
            optimizer=optimizer,
            total_steps=total_steps,
            warmup_epochs=warmup_epochs,
            steps_per_epoch=steps_per_epoch,
            eta_min=eta_min,
        )
        self._step_count = 0

    def step(self) -> None:
        """Advance the scheduler by one step."""
        self._scheduler.step()
        self._step_count += 1

    def get_last_lr(self):
        """Return the last computed learning rates."""
        return self._scheduler.get_last_lr()

    def state_dict(self):
        """Return scheduler state for checkpointing."""
        return {
            "scheduler": self._scheduler.state_dict(),
            "step_count": self._step_count,
        }

    def load_state_dict(self, state_dict):
        """Load scheduler state from checkpoint."""
        self._scheduler.load_state_dict(state_dict["scheduler"])
        self._step_count = state_dict["step_count"]
