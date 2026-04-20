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

    # Support padding with -inf in targets (ignored in softmax)
    valid_mask = torch.isfinite(targets)
    scores = scores.masked_fill(~valid_mask, float("-inf"))
    targets = targets.masked_fill(~valid_mask, float("-inf"))

    p_target = torch.softmax(targets, dim=-1)
    log_p_model = torch.log_softmax(scores, dim=-1)
    loss = -(p_target * log_p_model).sum(dim=-1)

    # Rows with all padding -> zero loss
    has_valid = valid_mask.any(dim=-1)
    loss = torch.where(has_valid, loss, torch.zeros_like(loss))

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

    B, M = scores.shape

    valid_mask = labels > float('-inf')
    num_valid = valid_mask.sum(dim=-1)

    scores = scores / temperature

    _, sorted_indices = labels.sort(descending=True, dim=-1)

    sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices)
    sorted_mask = torch.gather(valid_mask, dim=-1, index=sorted_indices)

    sorted_scores = sorted_scores.masked_fill(~sorted_mask, float('-inf'))

    flipped = torch.flip(sorted_scores, dims=[-1])
    log_cumsum_flipped = torch.logcumsumexp(flipped, dim=-1)
    log_cumsum = torch.flip(log_cumsum_flipped, dims=[-1])

    log_probs = sorted_scores - log_cumsum

    log_probs = log_probs.masked_fill(~sorted_mask, 0.0)

    loss = -log_probs.sum(dim=-1)

    num_valid = num_valid.clamp(min=1)

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

    B, M = scores.shape
    device = scores.device

    valid_mask = labels > float('-inf')
    num_valid = valid_mask.sum(dim=-1)

    scores = scores / temperature

    _, sorted_indices = labels.sort(descending=True, dim=-1)
    sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices)
    sorted_mask = torch.gather(valid_mask, dim=-1, index=sorted_indices)

    sorted_scores = sorted_scores.masked_fill(~sorted_mask, float('-inf'))

    flipped = torch.flip(sorted_scores, dims=[-1])
    log_cumsum_flipped = torch.logcumsumexp(flipped, dim=-1)
    log_cumsum = torch.flip(log_cumsum_flipped, dims=[-1])

    log_probs = sorted_scores - log_cumsum

    positions = torch.arange(M, device=device).float().unsqueeze(0)
    weights = torch.pow(2.0, num_valid.unsqueeze(1) - positions) - 1.0
    weights = weights.clamp(min=0.0)

    weights_sum = weights.sum(dim=-1, keepdim=True).clamp(min=eps)
    weights = weights / weights_sum

    log_probs = log_probs.masked_fill(~sorted_mask, 0.0)
    weights = weights.masked_fill(~sorted_mask, 0.0)

    weighted_log_probs = log_probs * weights
    loss = -weighted_log_probs.sum(dim=-1)

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f"Unsupported reduction: {reduction!r}")
