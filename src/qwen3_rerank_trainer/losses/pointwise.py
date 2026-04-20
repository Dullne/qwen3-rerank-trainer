"""
Pointwise 损失函数

包含：
- yes_no_to_score: Qwen 格式转换
- pointwise_ce_from_yes_no_logits: 二分类 CE
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def yes_no_to_score(yes_logits: torch.Tensor, no_logits: torch.Tensor) -> torch.Tensor:
    """
    Convert Qwen-style (yes_logit, no_logit) into a single scalar score.

    For binary softmax over [no, yes]:
      P(yes) = softmax([no, yes])[1] = sigmoid(yes - no)
    so `score = yes - no` is the natural logit for P(yes).
    """
    return yes_logits - no_logits


def pointwise_ce_from_yes_no_logits(
    yes_logits: torch.Tensor,
    no_logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Point-wise classification loss for reranking.

    Inputs:
      - yes_logits/no_logits: shape [N]
      - labels: shape [N], values in {0,1} where 1 means "yes"/relevant.

    Loss:
      CrossEntropy over 2 classes [no, yes].
    """
    if yes_logits.shape != no_logits.shape:
        raise ValueError(f"yes_logits shape {tuple(yes_logits.shape)} != no_logits shape {tuple(no_logits.shape)}")
    if labels.ndim != 1:
        labels = labels.view(-1)

    logits_2 = torch.stack([no_logits, yes_logits], dim=1)
    labels = labels.to(dtype=torch.long, device=logits_2.device)
    return F.cross_entropy(logits_2, labels)
