"""
对比损失函数

包含：
- infonce_loss: InfoNCE（支持单正例与多正例策略）
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def infonce_loss(
    scores: torch.Tensor,
    positive_indices: torch.Tensor | None = None,
    pos_mask: torch.Tensor | None = None,
    *,
    scale: float = 1.0,
    reduction: str = "mean",
    mode: str = "single",
) -> torch.Tensor:
    """
    InfoNCE loss with optional multi-positive support.

    Modes:
      - "single": 单正例（使用 positive_indices 或 pos_mask 中唯一正例）
      - "posset": 多正例正例集（positives 不互相竞争）
      - "avgpos": 多正例逐正例对比（每个正例 vs 所有负例）
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.ndim}D")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    mode = mode.lower()
    if mode in ("single", "one"):
        if positive_indices is None:
            if pos_mask is None:
                raise ValueError("positive_indices or pos_mask is required for mode='single'")
            if pos_mask.ndim != 2 or pos_mask.shape != scores.shape:
                raise ValueError(f"pos_mask shape {tuple(pos_mask.shape)} != scores shape {tuple(scores.shape)}")
            pos_mask = pos_mask.to(dtype=torch.bool, device=scores.device)
            pos_count = pos_mask.sum(dim=-1)
            if (pos_count == 0).any():
                raise ValueError("InfoNCE requires at least 1 positive per row")
            positive_indices = pos_mask.float().argmax(dim=-1)
        else:
            if positive_indices.ndim != 1:
                raise ValueError(f"positive_indices must be 1D, got {positive_indices.ndim}D")
            if scores.shape[0] != positive_indices.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: scores {scores.shape[0]} vs positive_indices {positive_indices.shape[0]}"
                )
            if (positive_indices < 0).any() or (positive_indices >= scores.shape[1]).any():
                raise ValueError("positive_indices contains invalid index")

        scaled_scores = scores * float(scale)
        positive_indices = positive_indices.to(dtype=torch.long, device=scores.device)
        loss = F.cross_entropy(scaled_scores, positive_indices, reduction=reduction)
        return loss

    if mode in ("posset", "set", "avgpos", "average"):
        if pos_mask is None:
            if positive_indices is None:
                raise ValueError("pos_mask or positive_indices is required for multi-positive InfoNCE")
            if positive_indices.ndim != 1:
                raise ValueError(f"positive_indices must be 1D, got {positive_indices.ndim}D")
            if scores.shape[0] != positive_indices.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: scores {scores.shape[0]} vs positive_indices {positive_indices.shape[0]}"
                )
            if (positive_indices < 0).any() or (positive_indices >= scores.shape[1]).any():
                raise ValueError("positive_indices contains invalid index")
            pos_mask = torch.zeros_like(scores, dtype=torch.bool)
            pos_mask.scatter_(1, positive_indices.view(-1, 1), True)
        else:
            if pos_mask.ndim != 2 or pos_mask.shape != scores.shape:
                raise ValueError(f"pos_mask shape {tuple(pos_mask.shape)} != scores shape {tuple(scores.shape)}")
            pos_mask = pos_mask.to(dtype=torch.bool, device=scores.device)

        pos_count = pos_mask.sum(dim=-1)
        if (pos_count == 0).any():
            raise ValueError("InfoNCE requires at least 1 positive per row")

        scaled = scores * float(scale)
        if mode in ("posset", "set"):
            log_z = torch.logsumexp(scaled, dim=-1)
            pos_only = scaled.masked_fill(~pos_mask, float("-inf"))
            log_pos = torch.logsumexp(pos_only, dim=-1)
            loss = log_z - log_pos
        else:
            neg_only = scaled.masked_fill(pos_mask, float("-inf"))
            neg_lse = torch.logsumexp(neg_only, dim=-1)
            pos_only = scaled.masked_fill(~pos_mask, float("-inf"))
            denom = torch.logaddexp(pos_only, neg_lse.unsqueeze(-1))
            loss_mat = denom - pos_only
            loss_mat = torch.where(pos_mask, loss_mat, torch.zeros_like(loss_mat))
            loss = loss_mat.sum(dim=-1) / pos_count.to(loss_mat.dtype)

        if reduction == "none":
            return loss
        if reduction == "mean":
            return loss.mean()
        if reduction == "sum":
            return loss.sum()
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    raise ValueError(f"Unsupported mode for infonce_loss: {mode!r}")

