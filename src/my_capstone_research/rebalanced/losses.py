from __future__ import annotations

import torch
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float,
    alpha: float | None = None,
) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        reduction="none",
    )
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    modulating_factor = (1.0 - p_t).pow(gamma)
    loss = ce_loss * modulating_factor
    if alpha is not None:
        alpha_factor = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_factor * loss
    return loss.mean()


def weighted_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    positive_weight: float,
    negative_weight: float,
) -> torch.Tensor:
    weights = torch.where(
        targets > 0,
        torch.full_like(targets.float(), positive_weight),
        torch.full_like(targets.float(), negative_weight),
    )
    return F.binary_cross_entropy_with_logits(
        logits,
        targets.float(),
        weight=weights,
    )
