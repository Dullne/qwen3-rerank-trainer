"""
Pairwise 损失函数

包含：
- pairwise_posrank_loss: 正例间排序损失
- ranknet_loss: RankNet 损失
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def pairwise_posrank_loss(
    strong_scores: torch.Tensor,
    weak_scores: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Pairwise positive-positive ranking loss (RankNet-style).

    Goal:
      For the same query, enforce strong positive > weak positive.

    For each pair:
      delta = strong - weak
      loss = -log sigmoid(delta) = softplus(-delta)

    Notes:
      - This requires a notion of "strong vs weak" among positives (graded labels,
        teacher scores, or an explicit ordering convention in your data).
      - Do NOT train with only this loss (no negatives), otherwise scores can collapse.
        Use it as an auxiliary loss alongside a pos-vs-neg loss (e.g., BCE/CE).
    """
    if strong_scores.shape != weak_scores.shape:
        raise ValueError(
            f"strong_scores shape {tuple(strong_scores.shape)} != weak_scores shape {tuple(weak_scores.shape)}"
        )
    delta = strong_scores - weak_scores
    loss = F.softplus(-delta)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def ranknet_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    sigma: float = 1.0,
    max_pairs_per_batch: int = 2000000,
) -> torch.Tensor:
    """
    RankNet pairwise loss (listwise format).

    对于每个文档对 (i, j)，如果 label_i > label_j:
      loss = -log(sigmoid(sigma * (s_i - s_j))) = softplus(-sigma * (s_i - s_j))

    Args:
        scores: [B, L] 每个 query 的文档分数
        labels: [B, L] 文档标签（1=正例，0=负例，-inf=padding）
        sigma: 缩放因子
        max_pairs_per_batch: 最大 pair 数，超出后分块计算（控制显存）

    Returns:
        Average pairwise loss
    """
    # 只对有效标签构造正负对（-inf 视为 padding）
    valid_mask = labels > float('-inf')
    pos_mask = (labels > 0.5) & valid_mask
    neg_mask = (labels < 0.5) & valid_mask
    pair_mask = pos_mask.unsqueeze(2) & neg_mask.unsqueeze(1)  # [B, L, L]

    if not pair_mask.any():
        return torch.tensor(0.0, device=scores.device)

    # 估算 pair 数，超限则分块计算
    total_pairs = pair_mask.sum().item()
    if total_pairs <= max_pairs_per_batch:
        score_diffs = scores.unsqueeze(2) - scores.unsqueeze(1)
        pair_losses = F.softplus(-sigma * score_diffs)
        return pair_losses[pair_mask].mean()

    total_loss = 0.0
    num_pairs = 0
    B, L = scores.shape
    for b in range(B):
        pos_idx = pos_mask[b].nonzero(as_tuple=True)[0]
        neg_idx = neg_mask[b].nonzero(as_tuple=True)[0]
        if pos_idx.numel() == 0 or neg_idx.numel() == 0:
            continue
        pos_scores = scores[b, pos_idx]  # [P]
        neg_scores = scores[b, neg_idx]  # [N]
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
        pair_loss = F.softplus(-sigma * diff)
        total_loss += pair_loss.sum()
        num_pairs += pair_loss.numel()

    if num_pairs == 0:
        return torch.tensor(0.0, device=scores.device)

    return total_loss / num_pairs
