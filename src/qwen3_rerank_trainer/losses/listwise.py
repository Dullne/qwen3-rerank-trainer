"""
Listwise 损失函数

包含：
- ListNet (listwise_softmax_ce)
- ListMLE (list_mle)
- Position-Aware ListMLE (p_list_mle)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _tie_aware_pl_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
    position_aware: bool = False,
    eps: float = 1e-10,
) -> torch.Tensor:
    """Plackett-Luce loss over relevance groups, without ordering ties."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")

    scaled_scores = scores / float(temperature)
    row_losses = []

    for row_scores, row_labels in zip(scaled_scores, labels):
        valid_mask = torch.isfinite(row_labels)
        row_scores = row_scores[valid_mask]
        row_labels = row_labels[valid_mask]

        if row_scores.numel() <= 1:
            row_losses.append(scaled_scores.new_zeros(()))
            continue

        levels = torch.unique(row_labels)
        levels, _ = levels.sort(descending=True)
        if levels.numel() <= 1:
            row_losses.append(scaled_scores.new_zeros(()))
            continue

        remaining = torch.ones(row_scores.numel(), dtype=torch.bool, device=row_scores.device)
        row_loss = scaled_scores.new_zeros(())
        weight_offset = 0

        if position_aware:
            positions = torch.arange(row_scores.numel(), device=row_scores.device, dtype=row_scores.dtype)
            n_valid = row_scores.new_tensor(float(row_scores.numel()))
            weights = torch.pow(row_scores.new_tensor(2.0), n_valid - positions) - 1.0
            weights = weights / weights.sum().clamp(min=eps)

        for level in levels[:-1]:
            group_mask = remaining & (row_labels == level)
            group_size = int(group_mask.sum().item())
            if group_size == 0:
                continue

            denom = torch.logsumexp(row_scores[remaining], dim=0)
            group_terms = -(row_scores[group_mask] - denom)
            term = group_terms.mean()

            if position_aware:
                group_weight = weights[weight_offset:weight_offset + group_size].sum()
                term = term * group_weight
                weight_offset += group_size

            row_loss = row_loss + term
            remaining = remaining & ~group_mask

        row_losses.append(row_loss)

    return torch.stack(row_losses)


def listwise_softmax_ce(
    scores: torch.Tensor,
    targets: torch.Tensor,
    *,
    score_temperature: float = 1.0,
    target_temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Listwise softmax cross-entropy (ListNet-style).

    This is useful when you have *graded* relevance labels or a teacher score
    distribution for candidates under the same query.

    Inputs:
      - scores: shape [B, M], model scores for M candidates per query.
      - targets: shape [B, M], either graded labels (e.g. 0/1/2/3) or teacher scores.

    Definition:
      p_target = softmax(targets / target_temperature)
      log p_model = log_softmax(scores / score_temperature)
      loss_i = - sum_j p_target[i,j] * log p_model[i,j]
    """
    if scores.ndim != 2 or targets.ndim != 2:
        raise ValueError(f"scores/targets must be 2D, got scores={scores.ndim}D targets={targets.ndim}D")
    if scores.shape != targets.shape:
        raise ValueError(f"scores shape {tuple(scores.shape)} != targets shape {tuple(targets.shape)}")
    if score_temperature <= 0 or target_temperature <= 0:
        raise ValueError("score_temperature and target_temperature must be > 0")

    scores = scores / float(score_temperature)
    targets = targets / float(target_temperature)

    # Support padding with -inf in targets (ignored in softmax).
    valid_mask = torch.isfinite(targets)
    has_valid = valid_mask.any(dim=-1, keepdim=True)
    scores = scores.masked_fill(~valid_mask, float("-inf"))
    targets = targets.masked_fill(~valid_mask, float("-inf"))

    # All-padding rows would make softmax/log_softmax return NaNs.
    scores = torch.where(has_valid, scores, torch.zeros_like(scores))
    targets = torch.where(has_valid, targets, torch.zeros_like(targets))

    p_target = torch.softmax(targets, dim=-1)
    log_p_model = torch.log_softmax(scores, dim=-1)
    terms = torch.where(valid_mask, p_target * log_p_model, torch.zeros_like(log_p_model))
    loss = -terms.sum(dim=-1)

    # Rows with all padding -> zero loss
    loss = torch.where(has_valid.squeeze(-1), loss, torch.zeros_like(loss))

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def list_mle(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1e-10,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    ListMLE: Listwise approach to learning to rank based on maximum likelihood estimation.

    最大化由真实标签诱导的排列的似然概率。使用数值稳定的 log-sum-exp 技巧。

    Inputs:
      - scores: shape [B, M], 模型对 M 个候选的预测分数
      - labels: shape [B, M], 相关性标签（graded 或 binary）
                padding 位置应设为 -inf，会被自动排除
      - temperature: 分数缩放因子

    Definition:
      给定按标签降序排列 y(1), y(2), ..., y(n)，计算：
      P(y|s) = ∏_{i=1}^n exp(s_{y(i)}/τ) / ∑_{k=i}^n exp(s_{y(k)}/τ)
      loss = -log P(y|s)

    Reference:
      Xia et al., "Listwise Approach to Learning to Rank: Theory and Algorithm", ICML 2008
    """
    if scores.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"scores/labels must be 2D, got scores={scores.ndim}D labels={labels.ndim}D")
    if scores.shape != labels.shape:
        raise ValueError(f"scores shape {tuple(scores.shape)} != labels shape {tuple(labels.shape)}")

    loss = _tie_aware_pl_loss(
        scores,
        labels,
        temperature=temperature,
        position_aware=False,
        eps=eps,
    )

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def p_list_mle(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    eps: float = 1e-10,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Position-Aware ListMLE (p-ListMLE): 带位置权重的 ListMLE。

    在 ListMLE 基础上，对不同位置的 log-likelihood 赋予不同权重，
    top 位置的权重更大，使模型更关注 top 排序的准确性。

    Inputs:
      - scores: shape [B, M], 模型对 M 个候选的预测分数
      - labels: shape [B, M], 相关性标签，padding 位置应设为 -inf
      - temperature: 分数缩放因子

    Definition:
      weight[i] = 2^(n-i) - 1，其中 n 是有效文档数，i 是位置（从 0 开始）
      loss = -∑_{i=1}^n weight[i] × [s_{y(i)} - log(∑_{k=i}^n exp(s_{y(k)}))]

    Reference:
      Lan et al., "Position-Aware ListMLE: A Sequential Learning Process for Ranking", UAI 2014
    """
    if scores.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"scores/labels must be 2D, got scores={scores.ndim}D labels={labels.ndim}D")
    if scores.shape != labels.shape:
        raise ValueError(f"scores shape {tuple(scores.shape)} != labels shape {tuple(labels.shape)}")

    loss = _tie_aware_pl_loss(
        scores,
        labels,
        temperature=temperature,
        position_aware=True,
        eps=eps,
    )

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")
