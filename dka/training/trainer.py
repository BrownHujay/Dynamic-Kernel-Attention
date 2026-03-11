"""Main training loop for Dynamic Kernel Attention.

Provides a Trainer class that handles:
  - Mixed precision training (CUDA / MPS / CPU)
  - Per-group gradient clipping
  - EMA model evaluation
  - Checkpointing (best model + periodic)
  - Optional wandb logging
  - Mixup/CutMix application
  - Classification (top-1 accuracy) and language modeling (perplexity) evaluation

Reference: DKA Build Guide, Sections 5.2, 5.4, 10.4, 10.5.
"""

import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ema import EMA
from .optimizer import clip_grad_norms

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def _get_amp_device_type(device: torch.device) -> str:
    """Return the device type string for torch.amp.

    MPS and CPU both use 'cpu' for autocast; CUDA uses 'cuda'.
    """
    if device.type == "cuda":
        return "cuda"
    return "cpu"


def _get_device(device_str: Optional[str] = None) -> torch.device:
    """Resolve device string to torch.device, with automatic detection."""
    if device_str is not None:
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class Trainer:
    """Main training loop for DKA models.

    Supports both classification (image/text) and language modeling tasks.

    Args:
        model: The DKA model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        optimizer: Configured optimizer (from build_optimizer).
        scheduler: LR scheduler (from build_scheduler).
        loss_fn: Loss function (from DKALoss). Must accept (logits, targets,
            kernel_list) and return a dict with 'loss', 'ce_loss', 'div_loss'.
        config: Configuration dict with training hyperparameters:
            - task: 'classification' or 'lm' (language modeling)
            - epochs: Total number of training epochs
            - device: Device string ('cuda', 'mps', 'cpu', or None for auto)
            - use_amp: Whether to use mixed precision (default True)
            - ema_decay: EMA decay rate (0 to disable, default 0.9999)
            - save_dir: Directory for checkpoints (default 'checkpoints')
            - save_every: Save checkpoint every N epochs (default 10)
            - log_every: Log metrics every N steps (default 50)
            - grad_clip_global: Global gradient norm clip (default 1.0)
            - grad_clip_kernel_gen: Kernel gen gradient norm clip (default 0.5)
            - use_wandb: Whether to log to wandb (default False)
            - wandb_project: Wandb project name (default 'dka')
            - wandb_run_name: Wandb run name (optional)
            - num_classes: Number of classes (classification only)
            - vocab_size: Vocabulary size (LM only)
        mixup_fn: Optional mixup/cutmix function. Called as
            mixup_fn(images, targets) -> (mixed_images, mixed_targets).
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        loss_fn: nn.Module,
        config: Dict[str, Any],
        mixup_fn: Optional[Callable] = None,
    ):
        self.config = config
        self.task = config.get("task", "classification")
        self.epochs = config.get("epochs", 300)
        self.device = _get_device(config.get("device", None))
        self.use_amp = config.get("use_amp", True)
        self.save_dir = config.get("save_dir", "checkpoints")
        self.save_every = config.get("save_every", 10)
        self.log_every = config.get("log_every", 50)
        self.div_loss_every = config.get("div_loss_every", 10)  # Only compute diversity loss every N steps
        self.grad_clip_global = config.get("grad_clip_global", 1.0)
        self.grad_clip_kernel_gen = config.get("grad_clip_kernel_gen", 0.5)
        self.use_wandb = config.get("use_wandb", False) and WANDB_AVAILABLE

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.mixup_fn = mixup_fn

        # Mixed precision setup
        self.amp_device_type = _get_amp_device_type(self.device)
        # GradScaler is only needed for CUDA; for CPU/MPS autocast we skip it
        self.use_grad_scaler = self.use_amp and self.device.type == "cuda"
        self.scaler = (
            torch.amp.GradScaler("cuda") if self.use_grad_scaler else None
        )

        # EMA
        ema_decay = config.get("ema_decay", 0.9999)
        if ema_decay > 0:
            self.ema = EMA(self.model, decay=ema_decay)
        else:
            self.ema = None

        # Checkpointing
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_metric = float("-inf")  # accuracy or -perplexity
        self.global_step = 0

        # Wandb init
        if self.use_wandb:
            wandb.init(
                project=config.get("wandb_project", "dka"),
                name=config.get("wandb_run_name", None),
                config=config,
            )

    # ------------------------------------------------------------------
    # Kernel collection helpers
    # ------------------------------------------------------------------

    def _collect_kernels(self) -> List[torch.Tensor]:
        """Collect generated kernels from all DKA modules in the model.

        Walks the module tree looking for DKAModule instances and retrieves
        their cached kernels from the most recent forward pass.

        Returns:
            List of kernel tensors, each of shape (B, n, k_h, d_h).
        """
        kernels = []
        for module in self.model.modules():
            if hasattr(module, "get_last_kernels"):
                last_k = module.get_last_kernels()
                if last_k is not None:
                    for head_idx, k_tensor in last_k.items():
                        kernels.append(k_tensor)
        return kernels

    def _collect_alphas(self) -> List[float]:
        """Collect alpha values from all DKA modules.

        Returns:
            List of alpha scalar values.
        """
        alphas = []
        for module in self.model.modules():
            if hasattr(module, "get_last_alphas"):
                last_a = module.get_last_alphas()
                if last_a is not None:
                    for head_idx, a_tensor in last_a.items():
                        alphas.append(a_tensor.item())
        return alphas

    def _compute_kernel_stats(
        self, kernels: List[torch.Tensor]
    ) -> Dict[str, float]:
        """Compute kernel statistics for logging.

        Returns dict with:
            - mean_cosine_sim: Mean pairwise cosine similarity across all heads
            - mean_frob_norm_delta: Mean Frobenius norm of delta_K
        """
        stats = {"mean_cosine_sim": 0.0, "mean_frob_norm_delta": 0.0}

        if not kernels:
            return stats

        total_cos = 0.0
        total_frob = 0.0
        count = 0

        for k_tensor in kernels:
            B, n, k_h, d_h = k_tensor.shape
            flat = k_tensor.reshape(B, n, -1)  # (B, n, k_h*d_h)

            # Frobenius norm of each kernel
            frob_norms = torch.norm(flat, dim=-1)  # (B, n)
            total_frob += frob_norms.mean().item()

            # Pairwise cosine similarity: sample a few pairs for efficiency
            if n >= 2:
                num_samples = min(32, n)
                idx = torch.randperm(n, device=k_tensor.device)[:num_samples]
                sampled = flat[:, idx, :]  # (B, num_samples, k_h*d_h)
                # Normalize
                normed = F.normalize(sampled, dim=-1)
                # Pairwise cosine sim: (B, num_samples, num_samples)
                sim_matrix = torch.bmm(normed, normed.transpose(1, 2))
                # Extract upper triangle (exclude diagonal)
                mask = torch.triu(
                    torch.ones(num_samples, num_samples, device=k_tensor.device),
                    diagonal=1,
                ).bool()
                cos_vals = sim_matrix[:, mask]
                total_cos += cos_vals.mean().item()

            count += 1

        if count > 0:
            stats["mean_cosine_sim"] = total_cos / count
            stats["mean_frob_norm_delta"] = total_frob / count

        return stats

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_one_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch.

        Returns:
            Dict of aggregated metrics for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        running_ce = 0.0
        running_div = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            metrics = self._train_step(batch, epoch, batch_idx)

            running_loss += metrics["loss"]
            running_ce += metrics["ce_loss"]
            running_div += metrics["div_loss"]
            num_batches += 1

            # Step-level logging
            if self.global_step % self.log_every == 0:
                self._log_step(metrics, epoch, batch_idx)

            self.global_step += 1

        return {
            "train_loss": running_loss / max(num_batches, 1),
            "train_ce_loss": running_ce / max(num_batches, 1),
            "train_div_loss": running_div / max(num_batches, 1),
        }

    def _train_step(
        self, batch: Any, epoch: int, batch_idx: int
    ) -> Dict[str, float]:
        """Execute a single training step."""
        # Unpack batch
        if self.task == "lm":
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
        else:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

        # Apply mixup/cutmix (classification only, typically images)
        if self.mixup_fn is not None and self.task == "classification":
            inputs, targets = self.mixup_fn(inputs, targets)

        self.optimizer.zero_grad()

        # Only compute diversity loss every N steps (expensive on MPS)
        compute_div = (self.global_step % self.div_loss_every == 0)

        # Forward pass with mixed precision
        if self.use_amp:
            with torch.amp.autocast(device_type=self.amp_device_type):
                logits = self.model(inputs)
                # Reshape for LM: (B, T, V) -> (B*T, V)
                if self.task == "lm" and logits.ndim == 3:
                    B, T, V = logits.shape
                    logits = logits.reshape(B * T, V)
                    targets = targets.reshape(B * T)

                kernel_list = self._collect_kernels() if compute_div else []
                loss_dict = self.loss_fn(logits, targets, kernel_list)
                loss = loss_dict["loss"]
        else:
            logits = self.model(inputs)
            if self.task == "lm" and logits.ndim == 3:
                B, T, V = logits.shape
                logits = logits.reshape(B * T, V)
                targets = targets.reshape(B * T)

            kernel_list = self._collect_kernels() if compute_div else []
            loss_dict = self.loss_fn(logits, targets, kernel_list)
            loss = loss_dict["loss"]

        # Backward pass
        if self.use_grad_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_info = clip_grad_norms(
                self.model,
                max_norm_global=self.grad_clip_global,
                max_norm_kernel_gen=self.grad_clip_kernel_gen,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            grad_info = clip_grad_norms(
                self.model,
                max_norm_global=self.grad_clip_global,
                max_norm_kernel_gen=self.grad_clip_kernel_gen,
            )
            self.optimizer.step()

        # Scheduler step (per optimizer step)
        if self.scheduler is not None:
            self.scheduler.step()

        # EMA update
        if self.ema is not None:
            self.ema.update()

        return {
            "loss": loss.item(),
            "ce_loss": loss_dict["ce_loss"].item(),
            "div_loss": loss_dict["div_loss"].item(),
            "grad_norm_global": grad_info["grad_norm_global"],
            "grad_norm_kernel_gen": grad_info["grad_norm_kernel_gen"],
            "grad_norm_main": grad_info["grad_norm_main"],
        }

    def _log_step(
        self, metrics: Dict[str, float], epoch: int, batch_idx: int
    ) -> None:
        """Log step-level metrics to console and optionally wandb."""
        alphas = self._collect_alphas()
        kernels = self._collect_kernels()
        kernel_stats = self._compute_kernel_stats(kernels)

        alpha_mean = sum(alphas) / len(alphas) if alphas else 0.0
        alpha_std = (
            (sum((a - alpha_mean) ** 2 for a in alphas) / len(alphas)) ** 0.5
            if len(alphas) > 1
            else 0.0
        )

        lr_current = self.optimizer.param_groups[0]["lr"]

        log_data = {
            "epoch": epoch,
            "step": self.global_step,
            "total_loss": metrics["loss"],
            "ce_loss": metrics["ce_loss"],
            "div_loss": metrics["div_loss"],
            "alpha_mean": alpha_mean,
            "alpha_std": alpha_std,
            "mean_cosine_sim": kernel_stats["mean_cosine_sim"],
            "mean_frob_norm_delta": kernel_stats["mean_frob_norm_delta"],
            "grad_norm_global": metrics.get("grad_norm_global", 0.0),
            "grad_norm_kernel_gen": metrics.get("grad_norm_kernel_gen", 0.0),
            "grad_norm_main": metrics.get("grad_norm_main", 0.0),
            "lr": lr_current,
        }

        # Console output
        print(
            f"  [Epoch {epoch} | Step {self.global_step}] "
            f"loss={metrics['loss']:.4f} "
            f"ce={metrics['ce_loss']:.4f} "
            f"div={metrics['div_loss']:.4f} "
            f"alpha={alpha_mean:.4f}+/-{alpha_std:.4f} "
            f"cos_sim={kernel_stats['mean_cosine_sim']:.3f} "
            f"lr={lr_current:.2e}"
        )

        if self.use_wandb:
            wandb.log(log_data, step=self.global_step)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, use_ema: bool = True) -> Dict[str, float]:
        """Run evaluation on the validation set.

        Args:
            use_ema: If True and EMA is available, evaluate with EMA weights.

        Returns:
            Dict with evaluation metrics:
                - For classification: 'val_loss', 'val_accuracy'
                - For LM: 'val_loss', 'val_perplexity'
        """
        # Swap to EMA weights if available
        if use_ema and self.ema is not None:
            self.ema.apply()

        self.model.eval()

        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        total_ce_for_ppl = 0.0
        num_batches = 0

        for batch in self.val_loader:
            inputs, targets = batch
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            if self.use_amp:
                with torch.amp.autocast(device_type=self.amp_device_type):
                    logits = self.model(inputs)
            else:
                logits = self.model(inputs)

            if self.task == "lm":
                if logits.ndim == 3:
                    B, T, V = logits.shape
                    logits_flat = logits.reshape(B * T, V)
                    targets_flat = targets.reshape(B * T)
                else:
                    logits_flat = logits
                    targets_flat = targets

                loss = F.cross_entropy(logits_flat, targets_flat)
                total_loss += loss.item()
                total_ce_for_ppl += loss.item() * targets_flat.numel()
                total_tokens += targets_flat.numel()
            else:
                loss = F.cross_entropy(logits, targets)
                total_loss += loss.item()

                preds = logits.argmax(dim=-1)
                total_correct += (preds == targets).sum().item()
                total_tokens += targets.size(0)

            num_batches += 1

        # Restore original weights
        if use_ema and self.ema is not None:
            self.ema.restore()

        self.model.train()

        metrics: Dict[str, float] = {
            "val_loss": total_loss / max(num_batches, 1),
        }

        if self.task == "lm":
            avg_ce = total_ce_for_ppl / max(total_tokens, 1)
            metrics["val_perplexity"] = math.exp(min(avg_ce, 100))  # cap for stability
        else:
            metrics["val_accuracy"] = (
                total_correct / max(total_tokens, 1) * 100.0
            )

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """Save a training checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Current evaluation metrics.
            is_best: If True, also save as 'best_model.pt'.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "metrics": metrics,
            "global_step": self.global_step,
            "config": self.config,
        }

        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)

        if is_best:
            best_path = os.path.join(self.save_dir, "best_model.pt")
            torch.save(checkpoint, best_path)
            print(f"  Saved best model checkpoint at epoch {epoch}")

    def load_checkpoint(self, path: str) -> int:
        """Load a training checkpoint.

        Args:
            path: Path to the checkpoint file.

        Returns:
            The epoch number to resume from.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler is not None and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

        if self.scaler is not None and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint.get("global_step", 0)
        return checkpoint.get("epoch", 0)

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(self, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Run the full training loop.

        Args:
            resume_from: Optional path to a checkpoint to resume from.

        Returns:
            Dict with training history (losses, metrics per epoch).
        """
        start_epoch = 0
        if resume_from is not None:
            start_epoch = self.load_checkpoint(resume_from) + 1
            print(f"Resuming training from epoch {start_epoch}")

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        if self.task == "classification":
            history["val_accuracy"] = []
        else:
            history["val_perplexity"] = []

        print(f"Training on device: {self.device}")
        print(f"Mixed precision: {self.use_amp} (amp device type: {self.amp_device_type})")
        print(f"EMA: {'enabled' if self.ema else 'disabled'}")
        print(f"Total epochs: {self.epochs}")
        print(f"Logging every {self.log_every} steps")
        print("-" * 60)

        for epoch in range(start_epoch, self.epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_one_epoch(epoch)

            # Evaluate
            val_metrics = self.evaluate(use_ema=(self.ema is not None))

            epoch_time = time.time() - epoch_start

            # Track best metric
            if self.task == "classification":
                current_metric = val_metrics.get("val_accuracy", 0.0)
                metric_name = "accuracy"
            else:
                # For LM, lower perplexity is better; negate for comparison
                current_metric = -val_metrics.get("val_perplexity", float("inf"))
                metric_name = "perplexity"

            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric

            # Save checkpoints
            if is_best:
                self.save_checkpoint(epoch, val_metrics, is_best=True)
            if (epoch + 1) % self.save_every == 0:
                self.save_checkpoint(epoch, val_metrics, is_best=False)

            # Record history
            history["train_loss"].append(train_metrics["train_loss"])
            history["val_loss"].append(val_metrics["val_loss"])
            if self.task == "classification":
                history["val_accuracy"].append(
                    val_metrics.get("val_accuracy", 0.0)
                )
            else:
                history["val_perplexity"].append(
                    val_metrics.get("val_perplexity", 0.0)
                )

            # Epoch-level logging
            self._log_epoch(epoch, train_metrics, val_metrics, epoch_time)

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch,
                        **train_metrics,
                        **val_metrics,
                        "epoch_time": epoch_time,
                    },
                    step=self.global_step,
                )

        # Save final checkpoint
        final_metrics = self.evaluate(use_ema=(self.ema is not None))
        self.save_checkpoint(self.epochs - 1, final_metrics, is_best=False)

        if self.use_wandb:
            wandb.finish()

        return history

    def _log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        epoch_time: float,
    ) -> None:
        """Print epoch-level summary."""
        msg = (
            f"Epoch {epoch}/{self.epochs} ({epoch_time:.1f}s) | "
            f"Train loss: {train_metrics['train_loss']:.4f} "
            f"(CE: {train_metrics['train_ce_loss']:.4f}, "
            f"Div: {train_metrics['train_div_loss']:.4f})"
        )

        if self.task == "classification":
            msg += f" | Val loss: {val_metrics['val_loss']:.4f}"
            msg += f" | Val acc: {val_metrics.get('val_accuracy', 0):.2f}%"
        else:
            msg += f" | Val loss: {val_metrics['val_loss']:.4f}"
            msg += f" | Val ppl: {val_metrics.get('val_perplexity', 0):.2f}"

        best_str = (
            f" [BEST]"
            if (
                self.task == "classification"
                and val_metrics.get("val_accuracy", 0) >= self.best_metric
            )
            or (
                self.task == "lm"
                and -val_metrics.get("val_perplexity", float("inf"))
                >= self.best_metric
            )
            else ""
        )
        print(msg + best_str)
