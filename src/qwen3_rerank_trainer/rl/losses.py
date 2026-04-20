"""
RL 损失函数模块

提供 Doc-level REINFORCE 和 DPO 损失函数，用于 Reranker 的二阶段训练。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F

from .rewards import compute_doc_level_advantages, compute_doc_level_rewards


def reinforce_loss(
    yes_logits: torch.Tensor,
    no_logits: torch.Tensor,
    labels: torch.Tensor,
    reward_type: str = "rank_based",
    scale_rewards: Union[bool, str] = True,
    loss_type: str = "dapo",
    max_completion_length: Optional[int] = None,
    reward_k: int = 10,
    clip_range: float = 0.2,
    ref_yes_logits: Optional[torch.Tensor] = None,
    ref_no_logits: Optional[torch.Tensor] = None,
    kl_coef: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Doc-level REINFORCE 损失（组内归一化，类似 GRPO）

    对同一 query 的 N 个 doc 计算 doc-level advantage，
    然后用各自的 advantage 优化各自的 log_prob。

    Args:
        yes_logits: [N] 每个 doc 的 yes token logit
        no_logits: [N] 每个 doc 的 no token logit
        labels: [N] 每个 doc 的标签（0/1）
        reward_type: 奖励类型
            - "rank_based": 基于排名（ERANK 风格）
            - "score_based": 基于分数
            - "ndcg_based": 基于 DCG（ReasonRank 风格）
            - "recall_based": 基于 Recall@k（ReasonRank 风格）
        scale_rewards: 是否按 std 缩放 advantage
            - True 或 "group": advantage = (r - mean) / std（标准 GRPO，默认）
            - False 或 "none": advantage = r - mean（Dr. GRPO 推荐）
        loss_type: Loss 归一化方式
            - "grpo": 原始 GRPO
            - "dapo": DAPO（默认）
            - "dr_grpo": Dr. GRPO
        max_completion_length: dr_grpo 模式下的常数 L
        reward_k: ndcg_based/recall_based 的阈值参数
        clip_range: PPO 风格的 importance ratio clipping
        ref_yes_logits: [N] 参考模型的 yes logits（可选）
        ref_no_logits: [N] 参考模型的 no logits（可选）
        kl_coef: KL 惩罚系数

    Returns:
        loss: Policy gradient loss
        advantages: [N] 每个 doc 的 advantage
        rewards: [N] 每个 doc 的 reward
        kl: KL 散度
    """
    scores = torch.sigmoid(yes_logits - no_logits)

    advantages = compute_doc_level_advantages(
        scores, labels, reward_type=reward_type, scale_rewards=scale_rewards, k=reward_k
    )

    logit_diff = yes_logits - no_logits
    log_probs = torch.where(
        labels == 1,
        F.logsigmoid(logit_diff),
        F.logsigmoid(-logit_diff),
    )

    losses = -advantages * log_probs

    if clip_range > 0 and ref_yes_logits is not None and ref_no_logits is not None:
        ref_logit_diff = ref_yes_logits - ref_no_logits
        ref_log_probs = torch.where(
            labels == 1,
            F.logsigmoid(ref_logit_diff),
            F.logsigmoid(-ref_logit_diff),
        )
        log_ratio = log_probs - ref_log_probs
        ratio = torch.exp(log_ratio)

        ratio_diff = (ratio - 1.0).abs().mean()
        if ratio_diff > 0.01:
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            loss1 = -advantages * ratio
            loss2 = -advantages * clipped_ratio
            losses = torch.where(advantages >= 0, torch.max(loss1, loss2), torch.min(loss1, loss2))

    G = len(yes_logits)

    if loss_type == "grpo":
        pg_loss = losses.mean()
    elif loss_type == "dapo":
        pg_loss = losses.mean()
    elif loss_type == "dr_grpo":
        L = max_completion_length if max_completion_length is not None else G
        pg_loss = losses.sum() / (L * G)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose from 'grpo', 'dapo', 'dr_grpo'")

    kl = torch.tensor(0.0, device=yes_logits.device)
    if kl_coef > 0 and ref_yes_logits is not None and ref_no_logits is not None:
        logit_diff = yes_logits - no_logits
        ref_logit_diff = ref_yes_logits - ref_no_logits

        log_prob_policy = torch.where(
            labels == 1,
            F.logsigmoid(logit_diff),
            F.logsigmoid(-logit_diff),
        )
        log_prob_ref = torch.where(
            labels == 1,
            F.logsigmoid(ref_logit_diff),
            F.logsigmoid(-ref_logit_diff),
        )

        log_ratio = log_prob_ref - log_prob_policy
        ratio = torch.exp(log_ratio)
        kl = (ratio - log_ratio - 1).mean()

    loss = pg_loss + kl_coef * kl

    rewards = compute_doc_level_rewards(scores, labels, reward_type, k=reward_k)

    return loss, advantages, rewards, kl


def dpo_loss(
    pos_yes_logits: torch.Tensor,
    pos_no_logits: torch.Tensor,
    neg_yes_logits: torch.Tensor,
    neg_no_logits: torch.Tensor,
    beta: float = 0.1,
    ref_pos_yes_logits: Optional[torch.Tensor] = None,
    ref_pos_no_logits: Optional[torch.Tensor] = None,
    ref_neg_yes_logits: Optional[torch.Tensor] = None,
    ref_neg_no_logits: Optional[torch.Tensor] = None,
    reference_free: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """DPO (Direct Preference Optimization) 排序损失

    支持多个正例和多个负例，自动取平均后计算 loss。
    优化目标：正例整体的 P(yes) > 负例整体的 P(yes)

    Args:
        pos_yes_logits: [num_pos] 所有正例的 yes logit
        pos_no_logits: [num_pos] 所有正例的 no logit
        neg_yes_logits: [num_neg] 所有负例的 yes logit
        neg_no_logits: [num_neg] 所有负例的 no logit
        beta: DPO 温度参数
        ref_pos_yes_logits: [num_pos] 参考模型的正例 yes logits
        ref_pos_no_logits: [num_pos] 参考模型的正例 no logits
        ref_neg_yes_logits: [num_neg] 参考模型的负例 yes logits
        ref_neg_no_logits: [num_neg] 参考模型的负例 no logits
        reference_free: 是否不使用参考模型

    Returns:
        loss: DPO loss
        pos_score: 正例平均 P(yes)
        neg_score: 负例平均 P(yes)
    """
    pos_log_probs = F.logsigmoid(pos_yes_logits - pos_no_logits)
    pos_avg_log_prob = pos_log_probs.mean()

    neg_log_probs = F.logsigmoid(neg_yes_logits - neg_no_logits)
    neg_avg_log_prob = neg_log_probs.mean()

    log_ratio_policy = pos_avg_log_prob - neg_avg_log_prob

    if reference_free:
        log_ratio_ref = 0.0
    else:
        if ref_pos_yes_logits is None or ref_neg_yes_logits is None:
            raise ValueError(
                "reference_free=False 但未提供参考模型 logits。"
                "请提供 ref_pos_yes_logits, ref_pos_no_logits, ref_neg_yes_logits, ref_neg_no_logits"
            )
        ref_pos_log_probs = F.logsigmoid(ref_pos_yes_logits - ref_pos_no_logits)
        ref_neg_log_probs = F.logsigmoid(ref_neg_yes_logits - ref_neg_no_logits)
        log_ratio_ref = ref_pos_log_probs.mean() - ref_neg_log_probs.mean()

    loss = -F.logsigmoid(beta * (log_ratio_policy - log_ratio_ref))

    pos_score = torch.sigmoid(pos_yes_logits - pos_no_logits).mean()
    neg_score = torch.sigmoid(neg_yes_logits - neg_no_logits).mean()

    return loss, pos_score, neg_score
