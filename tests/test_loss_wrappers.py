"""Tests for nn.Module loss wrappers and RL helper exports."""

import torch

from qwen3_rerank_trainer import (
    DAPOLoss,
    DPOLoss,
    LambdaLoss,
    ListMLELoss,
    ListwiseLoss,
    PointwiseCELoss,
    RankNetLoss,
    rank_based_reward,
)


def test_ranking_wrappers_run():
    scores = torch.tensor([[2.0, 1.0, 0.5]])
    labels = torch.tensor([[1.0, 0.0, 1.0]])

    listwise_loss = ListwiseLoss(temperature=1.0)(scores, labels)
    listmle_loss = ListMLELoss()(scores, labels)
    lambda_loss = LambdaLoss(metric="ndcg")(scores, labels)
    ranknet_loss = RankNetLoss()(scores, labels)

    for loss in [listwise_loss, listmle_loss, lambda_loss, ranknet_loss]:
        assert torch.is_tensor(loss)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def test_pointwise_wrapper_runs():
    loss = PointwiseCELoss()(
        yes_logits=torch.tensor([2.0, 0.1]),
        no_logits=torch.tensor([0.1, 1.0]),
        labels=torch.tensor([1, 0]),
    )

    assert torch.is_tensor(loss)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_rl_wrappers_and_reward_aliases_run():
    yes_logits = torch.tensor([2.5, 0.5, -0.5])
    no_logits = torch.tensor([0.1, 0.2, 0.3])
    labels = torch.tensor([1, 0, 0])

    rewards = rank_based_reward(torch.sigmoid(yes_logits - no_logits), labels)
    loss, advantages, reward_values, kl = DAPOLoss()(
        yes_logits,
        no_logits,
        labels,
        return_stats=True,
    )

    assert rewards.shape == labels.shape
    assert advantages.shape == labels.shape
    assert reward_values.shape == labels.shape
    assert torch.isfinite(loss)
    assert torch.isfinite(kl)


def test_dpo_wrapper_runs():
    loss_fn = DPOLoss(beta=0.1, reference_free=True)
    loss, pos_score, neg_score = loss_fn(
        pos_yes_logits=torch.tensor([2.0, 2.5]),
        pos_no_logits=torch.tensor([0.0, 0.1]),
        neg_yes_logits=torch.tensor([0.2, 0.4]),
        neg_no_logits=torch.tensor([0.3, 0.5]),
        return_stats=True,
    )

    assert torch.isfinite(loss)
    assert torch.isfinite(pos_score)
    assert torch.isfinite(neg_score)
