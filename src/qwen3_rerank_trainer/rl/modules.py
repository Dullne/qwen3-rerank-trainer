"""nn.Module wrappers for qwen3 reranker RL losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .losses import dpo_loss, reinforce_loss
from .rewards import (
    compute_doc_level_advantages,
    compute_doc_level_rewards,
    compute_ndcg_based_rewards,
    compute_recall_based_rewards,
)


class REINFORCELoss(nn.Module):
    """Doc-level REINFORCE loss wrapper."""

    def __init__(
        self,
        reward_type: str = "rank_based",
        scale_rewards: bool | str = True,
        loss_type: str = "dapo",
        max_completion_length: Optional[int] = None,
        reward_k: int = 10,
        clip_range: float = 0.2,
        kl_coef: float = 0.0,
    ):
        super().__init__()
        self.reward_type = reward_type
        self.scale_rewards = scale_rewards
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.reward_k = reward_k
        self.clip_range = clip_range
        self.kl_coef = kl_coef

    def forward(
        self,
        yes_logits: torch.Tensor,
        no_logits: torch.Tensor,
        labels: torch.Tensor,
        ref_yes_logits: Optional[torch.Tensor] = None,
        ref_no_logits: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        result = reinforce_loss(
            yes_logits=yes_logits,
            no_logits=no_logits,
            labels=labels,
            reward_type=self.reward_type,
            scale_rewards=self.scale_rewards,
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
            reward_k=self.reward_k,
            clip_range=self.clip_range,
            ref_yes_logits=ref_yes_logits,
            ref_no_logits=ref_no_logits,
            kl_coef=self.kl_coef,
        )
        return result if return_stats else result[0]


class GRPOLoss(REINFORCELoss):
    """GRPO-style normalized REINFORCE loss."""

    def __init__(self, **kwargs):
        kwargs.setdefault("loss_type", "grpo")
        super().__init__(**kwargs)


class DAPOLoss(REINFORCELoss):
    """DAPO-style normalized REINFORCE loss."""

    def __init__(self, **kwargs):
        kwargs.setdefault("loss_type", "dapo")
        super().__init__(**kwargs)


class DRGRPOLoss(REINFORCELoss):
    """Dr. GRPO loss."""

    def __init__(self, **kwargs):
        kwargs.setdefault("loss_type", "dr_grpo")
        super().__init__(**kwargs)


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss wrapper."""

    def __init__(self, beta: float = 0.1, reference_free: bool = False):
        super().__init__()
        self.beta = beta
        self.reference_free = reference_free

    def forward(
        self,
        pos_yes_logits: torch.Tensor,
        pos_no_logits: torch.Tensor,
        neg_yes_logits: torch.Tensor,
        neg_no_logits: torch.Tensor,
        ref_pos_yes_logits: Optional[torch.Tensor] = None,
        ref_pos_no_logits: Optional[torch.Tensor] = None,
        ref_neg_yes_logits: Optional[torch.Tensor] = None,
        ref_neg_no_logits: Optional[torch.Tensor] = None,
        return_stats: bool = False,
    ):
        result = dpo_loss(
            pos_yes_logits=pos_yes_logits,
            pos_no_logits=pos_no_logits,
            neg_yes_logits=neg_yes_logits,
            neg_no_logits=neg_no_logits,
            beta=self.beta,
            ref_pos_yes_logits=ref_pos_yes_logits,
            ref_pos_no_logits=ref_pos_no_logits,
            ref_neg_yes_logits=ref_neg_yes_logits,
            ref_neg_no_logits=ref_neg_no_logits,
            reference_free=self.reference_free,
        )
        return result if return_stats else result[0]


def rank_based_reward(scores, labels):
    """Compute rank-based rewards."""

    return compute_doc_level_rewards(scores, labels, reward_type="rank_based")


def score_based_reward(scores, labels):
    """Compute score-based rewards."""

    return compute_doc_level_rewards(scores, labels, reward_type="score_based")


def ndcg_based_reward(scores, labels, k: int = 10):
    """Compute NDCG-based rewards."""

    return compute_ndcg_based_rewards(scores, labels, k=k)


def recall_based_reward(scores, labels, k: int = 10):
    """Compute Recall@k-based rewards."""

    return compute_recall_based_rewards(scores, labels, k=k)
