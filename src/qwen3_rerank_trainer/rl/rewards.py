"""
RL 奖励函数模块

提供 doc-level 排序奖励计算，用于 REINFORCE 训练。
"""

from __future__ import annotations

from typing import List, Union
import torch


def compute_ndcg_based_rewards(
    scores: Union[torch.Tensor, List[float]],
    labels: Union[torch.Tensor, List[int]],
    k: int = 10,
) -> torch.Tensor:
    """NDCG 风格的奖励（DCG gain-aware）

    基于 ReasonRank 论文的 multi-view ranking reward 设计。
    使用 DCG 的 gain 公式计算每个 doc 的贡献。

    Args:
        scores: [N] 每个 doc 的分数（P(yes)）
        labels: [N] 每个 doc 的标签（0/1）
        k: 只考虑 top-k 的奖励

    Returns:
        rewards: [N] 每个 doc 的奖励

    Note:
        奖励设计：
        - 正例：reward = (2^label - 1) / log2(rank + 1)（DCG gain）
        - 负例在 top-k 内：reward = -1 / log2(rank + 1)（惩罚占位）
        - 负例在 top-k 外：reward = 0
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    N = len(scores)
    device = scores.device

    sorted_indices = torch.argsort(scores, descending=True)
    ranks = torch.zeros(N, dtype=torch.long, device=device)
    ranks[sorted_indices] = torch.arange(1, N + 1, device=device)

    pos_mask = labels == 1
    neg_mask = labels == 0

    rewards = torch.zeros(N, dtype=torch.float32, device=device)

    discounts = 1.0 / torch.log2(ranks.float() + 1)

    rewards[pos_mask] = discounts[pos_mask]

    in_topk = ranks <= k
    rewards[neg_mask & in_topk] = -discounts[neg_mask & in_topk]

    return rewards


def compute_recall_based_rewards(
    scores: Union[torch.Tensor, List[float]],
    labels: Union[torch.Tensor, List[int]],
    k: int = 10,
) -> torch.Tensor:
    """Recall 风格的奖励（threshold-aware）

    基于 ReasonRank 论文的 Recall@k 设计。
    鼓励正例进入 top-k，惩罚负例占位。

    Args:
        scores: [N] 每个 doc 的分数（P(yes)）
        labels: [N] 每个 doc 的标签（0/1）
        k: Recall 的阈值

    Returns:
        rewards: [N] 每个 doc 的奖励

    Note:
        奖励设计：
        - 正例在 top-k 内：reward = 1.0（成功召回）
        - 正例在 top-k 外：reward = -1.0（召回失败，惩罚）
        - 负例在 top-k 内：reward = -1.0（占位，惩罚）
        - 负例在 top-k 外：reward = 0.0（正确排除）
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    N = len(scores)
    device = scores.device

    sorted_indices = torch.argsort(scores, descending=True)
    ranks = torch.zeros(N, dtype=torch.long, device=device)
    ranks[sorted_indices] = torch.arange(1, N + 1, device=device)

    pos_mask = labels == 1
    neg_mask = labels == 0
    in_topk = ranks <= k

    rewards = torch.zeros(N, dtype=torch.float32, device=device)

    rewards[pos_mask & in_topk] = 1.0
    rewards[pos_mask & ~in_topk] = -1.0

    rewards[neg_mask & in_topk] = -1.0

    return rewards


def compute_doc_level_rewards(
    scores: Union[torch.Tensor, List[float]],
    labels: Union[torch.Tensor, List[int]],
    reward_type: str = "rank_based",
    k: int = 10,
) -> torch.Tensor:
    """计算 doc-level 奖励（用于组内归一化）

    Args:
        scores: [N] 每个 doc 的分数（P(yes)）
        labels: [N] 每个 doc 的标签（0/1）
        reward_type: 奖励类型
            - "rank_based": 基于排名的奖励（类似 ERANK r_RR）
            - "score_based": 基于分数的奖励
            - "ndcg_based": 基于 DCG 的奖励（ReasonRank 风格）
            - "recall_based": 基于 Recall@k 的奖励（ReasonRank 风格）
        k: ndcg_based/recall_based 的阈值参数

    Returns:
        rewards: [N] 每个 doc 的奖励

    Note:
        rank_based 奖励设计：
        - 正例：reward = 1/rank（排名越高奖励越大）
        - 负例越界（排在某正例前）：reward = -1/min_pos_rank（惩罚）
        - 负例正确排序：reward = 0
    """
    if not isinstance(scores, torch.Tensor):
        scores = torch.tensor(scores, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.long)

    N = len(scores)
    device = scores.device

    sorted_indices = torch.argsort(scores, descending=True)
    ranks = torch.zeros(N, dtype=torch.long, device=device)
    ranks[sorted_indices] = torch.arange(1, N + 1, device=device)

    pos_mask = labels == 1
    neg_mask = labels == 0

    rewards = torch.zeros(N, dtype=torch.float32, device=device)

    if reward_type == "rank_based":
        if pos_mask.any():
            pos_ranks = ranks[pos_mask]
            min_pos_rank = pos_ranks.min().item()
            max_pos_rank = pos_ranks.max().item()

            rewards[pos_mask] = 1.0 / ranks[pos_mask].float()

            for i in range(N):
                if labels[i] == 0:
                    if ranks[i].item() < max_pos_rank:
                        rewards[i] = -1.0 / min_pos_rank
                    else:
                        rewards[i] = 0.0

    elif reward_type == "score_based":
        rewards = torch.where(
            pos_mask,
            scores,
            -scores,
        )

    elif reward_type == "ndcg_based":
        return compute_ndcg_based_rewards(scores, labels, k)

    elif reward_type == "recall_based":
        return compute_recall_based_rewards(scores, labels, k)

    else:
        raise ValueError(f"Unknown reward_type: {reward_type}. "
                        f"Choose from 'rank_based', 'score_based', 'ndcg_based', 'recall_based'")

    return rewards


def compute_doc_level_advantages(
    scores: Union[torch.Tensor, List[float]],
    labels: Union[torch.Tensor, List[int]],
    reward_type: str = "rank_based",
    scale_rewards: Union[bool, str] = True,
    k: int = 10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """计算 doc-level advantages（组内归一化）

    类似 GRPO 的组内归一化，但用于同一 query 的多个 doc。

    Args:
        scores: [N] 每个 doc 的分数
        labels: [N] 每个 doc 的标签
        reward_type: 奖励类型
        scale_rewards: 是否按 std 缩放（与 TRL GRPOConfig 对齐）
            - True 或 "group": advantage = (r - mean) / std（标准 GRPO，默认）
            - False 或 "none": advantage = r - mean（避免难度偏差）
        k: ndcg_based/recall_based 的阈值参数
        eps: 数值稳定性

    Returns:
        advantages: [N] 每个 doc 的 advantage（组内归一化后）
    """
    rewards = compute_doc_level_rewards(scores, labels, reward_type, k)

    mean_reward = rewards.mean()

    advantages = rewards - mean_reward

    should_scale = scale_rewards is True or scale_rewards == "group"
    if should_scale:
        std_reward = rewards.std()
        if std_reward > eps:
            advantages = advantages / std_reward

    return advantages
