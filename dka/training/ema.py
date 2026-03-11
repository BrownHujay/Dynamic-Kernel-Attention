"""Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of model parameters that is updated each training step
as: ema_param = decay * ema_param + (1 - decay) * model_param.

During evaluation, the EMA weights can be swapped into the model via apply(),
and restored afterward via restore().

Reference: DKA Build Guide, Section 5.4.
"""

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn


class EMA:
    """Exponential Moving Average of model parameters.

    Usage:
        ema = EMA(model, decay=0.9999)

        # During training, after each optimizer step:
        ema.update()

        # For evaluation:
        ema.apply()       # swap EMA weights into the model
        evaluate(model)
        ema.restore()     # swap original weights back

    Args:
        model: The model whose parameters to track.
        decay: EMA decay rate (default 0.9999).
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay

        # Shadow parameters: detached copies of model parameters
        self.shadow: Dict[str, torch.Tensor] = {}
        # Backup of original model parameters (populated by apply())
        self.backup: Dict[str, torch.Tensor] = {}

        # Initialize shadow parameters from current model state
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self) -> None:
        """Update EMA parameters using current model parameters.

        ema_param = decay * ema_param + (1 - decay) * model_param

        Should be called once after each optimizer step during training.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(
                    param.data, alpha=1.0 - self.decay
                )

    def apply(self) -> None:
        """Swap EMA weights into the model for evaluation.

        Backs up the current model weights so they can be restored later.
        Call restore() after evaluation to swap back.
        """
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self) -> None:
        """Restore original model weights after EMA evaluation.

        Must be called after apply() to swap the training weights back.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return EMA shadow parameters for checkpointing."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load EMA shadow parameters from a checkpoint."""
        for name, tensor in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(tensor)
