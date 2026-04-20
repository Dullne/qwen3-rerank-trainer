"""Tests for loss functions."""

import torch
import pytest


class TestLambdaLoss:
    """Tests for LambdaLoss framework."""

    def test_lambda_loss_basic(self):
        """Test basic lambda_loss computation."""
        from qwen3_rerank_trainer.losses import lambda_loss

        scores = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
        labels = torch.tensor([[1.0, 0.0, 1.0, 0.0]])

        loss = lambda_loss(scores, labels, weighting_scheme='ndcg_loss2pp')
        assert loss.item() >= 0
        assert not torch.isnan(loss)

    def test_lambda_loss_metric(self):
        """Test lambda_loss metric shortcut."""
        from qwen3_rerank_trainer.losses import lambda_loss

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])

        loss = lambda_loss(scores, labels, metric="ndcg")
        assert loss.item() >= 0

    def test_weighting_schemes(self):
        """Test all weighting schemes."""
        from qwen3_rerank_trainer.losses import lambda_loss, WEIGHTING_SCHEMES

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])

        for scheme_name in WEIGHTING_SCHEMES:
            loss = lambda_loss(scores, labels, weighting_scheme=scheme_name)
            assert not torch.isnan(loss), f"NaN loss for {scheme_name}"


class TestListwiseLoss:
    """Tests for listwise losses."""

    def test_list_mle(self):
        """Test ListMLE loss."""
        from qwen3_rerank_trainer.losses import list_mle

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])

        loss = list_mle(scores, labels)
        assert loss.item() >= 0

    def test_p_list_mle(self):
        """Test position-aware ListMLE."""
        from qwen3_rerank_trainer.losses import p_list_mle

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])

        loss = p_list_mle(scores, labels)
        assert loss.item() >= 0

    def test_listwise_softmax_ce(self):
        """Test ListNet (softmax CE)."""
        from qwen3_rerank_trainer.losses import listwise_softmax_ce

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])

        loss = listwise_softmax_ce(scores, labels)
        assert loss.item() >= 0


class TestContrastiveLoss:
    """Tests for contrastive losses."""

    def test_infonce_loss(self):
        """Test InfoNCE loss."""
        from qwen3_rerank_trainer.losses import infonce_loss

        scores = torch.tensor([[2.0, 1.0, 0.5, 0.1]])
        positive_indices = torch.tensor([0])  # 每个样本的正例索引

        loss = infonce_loss(scores, positive_indices)
        assert loss.item() >= 0

    def test_multipos_infonce(self):
        """Test multi-positive InfoNCE."""
        from qwen3_rerank_trainer.losses import infonce_loss

        scores = torch.tensor([[2.0, 1.5, 0.5, 0.1]])
        positive_mask = torch.tensor([[1, 1, 0, 0]])

        loss_posset = infonce_loss(scores, pos_mask=positive_mask, mode="posset")
        loss_avgpos = infonce_loss(scores, pos_mask=positive_mask, mode="avgpos")
        assert loss_posset.item() >= 0
        assert loss_avgpos.item() >= 0


class TestPairwiseLoss:
    """Tests for pairwise losses."""

    def test_ranknet_loss(self):
        """Test RankNet loss."""
        from qwen3_rerank_trainer.losses import ranknet_loss

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])

        loss = ranknet_loss(scores, labels)
        assert loss.item() >= 0


class TestRLLoss:
    """Tests for RL losses."""

    def test_dpo_loss(self):
        """Test DPO loss."""
        from qwen3_rerank_trainer.rl import dpo_loss

        pos_yes = torch.tensor([1.0, 1.2])
        pos_no = torch.tensor([0.3, 0.1])
        neg_yes = torch.tensor([0.2, 0.1])
        neg_no = torch.tensor([0.8, 0.9])

        loss, pos_score, neg_score = dpo_loss(
            pos_yes_logits=pos_yes,
            pos_no_logits=pos_no,
            neg_yes_logits=neg_yes,
            neg_no_logits=neg_no,
            beta=0.1,
            reference_free=True,
        )
        assert loss.item() >= 0
        assert pos_score.item() >= 0
        assert neg_score.item() >= 0
