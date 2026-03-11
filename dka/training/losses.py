"""Loss functions for Dynamic Kernel Attention training.

Provides:
  - Label-smoothed cross-entropy loss (classification and language modeling).
  - Kernel diversity loss that penalizes high cosine similarity between
    generated kernels of random token pairs within each head/layer.
  - Combined DKA loss: L_CE + lambda_div * L_div.

Reference: DKA Build Guide, Section 5.1.
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross-entropy loss with label smoothing.

    For classification: standard cross-entropy with optional label smoothing.
    For language modeling: next-token prediction cross-entropy.

    Args:
        smoothing: Label smoothing factor (default 0.1).
        reduction: Reduction mode -- 'mean', 'sum', or 'none'.
    """

    def __init__(self, smoothing: float = 0.1, reduction: str = "mean"):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute label-smoothed cross-entropy.

        Args:
            logits: Predictions of shape (B, C) for classification or
                    (B*T, V) for language modeling.
            targets: Ground truth of shape (B,) or (B*T,) with class indices.

        Returns:
            Scalar loss tensor.
        """
        C = logits.size(-1)

        if self.smoothing > 0.0:
            log_probs = F.log_softmax(logits, dim=-1)

            # True class contribution: (1 - smoothing) * log_prob of correct class
            nll_loss = F.nll_loss(log_probs, targets, reduction="none")

            # Uniform distribution contribution: -smoothing * mean of all log_probs
            smooth_loss = -log_probs.mean(dim=-1)

            loss = (1.0 - self.smoothing) * nll_loss + self.smoothing * smooth_loss

            if self.reduction == "mean":
                return loss.mean()
            elif self.reduction == "sum":
                return loss.sum()
            return loss
        else:
            return F.cross_entropy(logits, targets, reduction=self.reduction)


class KernelDiversityLoss(nn.Module):
    """Kernel diversity loss to prevent kernel collapse across tokens.

    For each head h in each layer l, samples M random token pairs (i, j) and
    penalizes high cosine similarity between their flattened generated kernels:

        L_div^{l,h} = (1/M) * sum max(0, cos_sim(vec(K_i), vec(K_j)) - tau)

    Total diversity loss is averaged over all heads and layers.

    Args:
        num_pairs: Number of random token pairs to sample per head (M).
        tau: Cosine similarity threshold. Pairs below tau incur no penalty.
    """

    def __init__(self, num_pairs: int = 64, tau: float = 0.5):
        super().__init__()
        self.num_pairs = num_pairs
        self.tau = tau

    def forward(self, kernel_list: List[torch.Tensor]) -> torch.Tensor:
        """Compute kernel diversity loss.

        Args:
            kernel_list: List of kernel tensors, one per (layer, head).
                Each tensor has shape (B, n, k_h, d_h). The list contains
                all heads across all layers -- length = L * H.

        Returns:
            Scalar diversity loss averaged over all heads, layers, and pairs.
        """
        if len(kernel_list) == 0:
            return torch.tensor(0.0)

        device = kernel_list[0].device
        total_loss = torch.tensor(0.0, device=device)

        for kernels in kernel_list:
            # kernels: (B, n, k_h, d_h)
            B, n, k_h, d_h = kernels.shape

            if n < 2:
                continue

            # Flatten kernels to vectors: (B, n, k_h * d_h)
            flat = kernels.reshape(B, n, -1)

            # Sample M random token pairs (indices into n)
            M = min(self.num_pairs, n * (n - 1) // 2)
            idx_i = torch.randint(0, n, (M,), device=device)
            idx_j = torch.randint(0, n, (M,), device=device)

            # Ensure i != j by resampling collisions
            same_mask = idx_i == idx_j
            idx_j[same_mask] = (idx_j[same_mask] + 1) % n

            # Gather vectors for the sampled pairs: (B, M, k_h*d_h)
            vecs_i = flat[:, idx_i, :]  # (B, M, k_h*d_h)
            vecs_j = flat[:, idx_j, :]  # (B, M, k_h*d_h)

            # Cosine similarity: (B, M)
            cos_sim = F.cosine_similarity(vecs_i, vecs_j, dim=-1)

            # Hinge loss: max(0, cos_sim - tau)
            pair_loss = F.relu(cos_sim - self.tau)

            # Mean over pairs and batch
            total_loss = total_loss + pair_loss.mean()

        # Average over all heads/layers
        num_entries = len(kernel_list)
        if num_entries > 0:
            total_loss = total_loss / num_entries

        return total_loss


class DKALoss(nn.Module):
    """Combined DKA training loss.

    L = L_CE + lambda_div * L_div

    Args:
        smoothing: Label smoothing factor for cross-entropy.
        lambda_div: Weight for the kernel diversity loss.
        num_pairs: Number of random pairs for diversity loss (M).
        tau: Cosine similarity threshold for diversity loss.
    """

    def __init__(
        self,
        smoothing: float = 0.1,
        lambda_div: float = 0.1,
        num_pairs: int = 64,
        tau: float = 0.5,
    ):
        super().__init__()
        self.ce_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.div_loss = KernelDiversityLoss(num_pairs=num_pairs, tau=tau)
        self.lambda_div = lambda_div

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        kernel_list: Optional[List[torch.Tensor]] = None,
    ) -> dict:
        """Compute combined loss.

        Args:
            logits: Model predictions -- (B, C) or (B*T, V).
            targets: Ground truth labels -- (B,) or (B*T,).
            kernel_list: List of kernel tensors from all layers/heads.
                Each has shape (B, n, k_h, d_h). If None or empty, diversity
                loss is zero.

        Returns:
            Dict with keys:
                'loss': Total combined loss (scalar).
                'ce_loss': Cross-entropy component (scalar).
                'div_loss': Diversity loss component (scalar).
        """
        ce = self.ce_loss(logits, targets)

        if kernel_list is not None and len(kernel_list) > 0 and self.lambda_div > 0:
            div = self.div_loss(kernel_list)
        else:
            div = torch.tensor(0.0, device=logits.device)

        total = ce + self.lambda_div * div

        return {
            "loss": total,
            "ce_loss": ce.detach(),
            "div_loss": div.detach(),
        }
