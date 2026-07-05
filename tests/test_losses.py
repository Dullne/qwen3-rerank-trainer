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

    def test_lambda_loss_no_valid_pair_keeps_backward_graph(self):
        """Rows without positive-negative pairs should return a connected zero loss."""
        from qwen3_rerank_trainer.losses import lambda_loss

        scores = torch.tensor([[2.0, 1.0, 0.5]], requires_grad=True)
        labels = torch.tensor([[1.0, 1.0, 1.0]])

        loss = lambda_loss(scores, labels)
        assert loss.item() == 0.0
        loss.backward()
        assert scores.grad is not None
        assert torch.equal(scores.grad, torch.zeros_like(scores))


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

    def test_listwise_softmax_ce_ignores_padding(self):
        """ListNet padding should not create 0 * -inf NaNs."""
        from qwen3_rerank_trainer.losses import listwise_softmax_ce

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, float("-inf"), 0.0]])

        loss = listwise_softmax_ce(scores, labels)
        assert torch.isfinite(loss)

    def test_list_mle_ties_are_order_invariant(self):
        """Equal relevance labels should not impose an arbitrary order."""
        from qwen3_rerank_trainer.losses import list_mle, p_list_mle

        labels = torch.tensor([[1.0, 1.0, 0.0]])
        scores_a = torch.tensor([[2.0, 1.0, 0.0]])
        scores_b = torch.tensor([[1.0, 2.0, 0.0]])

        assert torch.allclose(list_mle(scores_a, labels), list_mle(scores_b, labels))
        assert torch.allclose(p_list_mle(scores_a, labels), p_list_mle(scores_b, labels))

    def test_list_mle_tied_positives_train_lower_scored_tie(self):
        """Tied positives should not cancel the gradient for a low-scored positive."""
        from qwen3_rerank_trainer.losses import list_mle

        scores = torch.tensor([[100.0, -100.0, 99.0]], requires_grad=True)
        labels = torch.tensor([[1.0, 1.0, 0.0]])

        loss = list_mle(scores, labels)
        loss.backward()

        assert scores.grad is not None
        assert scores.grad[0, 1] < -0.1


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

    def test_single_infonce_rejects_multiple_positives(self):
        """Single-positive mode should not silently treat positives as negatives."""
        from qwen3_rerank_trainer.losses import infonce_loss

        scores = torch.tensor([[1.0, 10.0, 0.0]])
        positive_mask = torch.tensor([[1, 1, 0]])

        with pytest.raises(ValueError, match="exactly 1 positive"):
            infonce_loss(scores, pos_mask=positive_mask, mode="single")


class TestPairwiseLoss:
    """Tests for pairwise losses."""

    def test_ranknet_loss(self):
        """Test RankNet loss."""
        from qwen3_rerank_trainer.losses import ranknet_loss

        scores = torch.tensor([[2.0, 1.0, 0.5]])
        labels = torch.tensor([[1.0, 0.0, 1.0]])

        loss = ranknet_loss(scores, labels)
        assert loss.item() >= 0

    def test_ranknet_no_valid_pair_keeps_backward_graph(self):
        """No-pair batches should still be safe to backprop through."""
        from qwen3_rerank_trainer.losses import ranknet_loss

        scores = torch.tensor([[2.0, 1.0, 0.5]], requires_grad=True)
        labels = torch.tensor([[1.0, 1.0, 1.0]])

        loss = ranknet_loss(scores, labels)
        assert loss.item() == 0.0
        loss.backward()
        assert scores.grad is not None
        assert torch.equal(scores.grad, torch.zeros_like(scores))


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

    def test_reinforce_score_based_advantages_are_detached(self):
        """Policy-gradient advantages should not backprop through rewards."""
        from qwen3_rerank_trainer.rl import reinforce_loss

        yes_logits = torch.tensor([1.0, 0.5, -0.5], requires_grad=True)
        no_logits = torch.tensor([0.0, 0.0, 0.0], requires_grad=True)
        labels = torch.tensor([1, 0, 0])

        loss, advantages, rewards, kl = reinforce_loss(
            yes_logits,
            no_logits,
            labels,
            reward_type="score_based",
            scale_rewards=False,
            kl_coef=0.0,
        )

        assert not advantages.requires_grad
        assert not rewards.requires_grad
        loss.backward()
        assert yes_logits.grad is not None
        assert no_logits.grad is not None

    def test_reinforce_negative_advantage_lowers_yes_score(self):
        """A penalized negative document should get a positive yes-logit gradient."""
        from qwen3_rerank_trainer.rl import reinforce_loss

        yes_logits = torch.tensor([0.0, 0.0], requires_grad=True)
        no_logits = torch.tensor([0.0, 0.0], requires_grad=True)
        labels = torch.tensor([1, 0])

        loss, advantages, rewards, kl = reinforce_loss(
            yes_logits,
            no_logits,
            labels,
            reward_type="score_based",
            scale_rewards=False,
            kl_coef=0.0,
        )
        loss.backward()

        assert advantages.tolist() == [0.5, -0.5]
        assert yes_logits.grad[0] < 0
        assert yes_logits.grad[1] > 0

    def test_reinforce_rejects_non_1d_labels(self):
        """REINFORCE loss should fail fast on accidental matrix labels."""
        from qwen3_rerank_trainer.rl import reinforce_loss

        with pytest.raises(ValueError, match="labels must be 1D"):
            reinforce_loss(
                yes_logits=torch.tensor([1.0, 0.5]),
                no_logits=torch.tensor([0.0, 0.0]),
                labels=torch.tensor([[1, 0]]),
            )

    def test_dpo_requires_complete_reference_logits(self):
        """DPO should validate all reference logits before tensor math."""
        from qwen3_rerank_trainer.rl import dpo_loss

        with pytest.raises(ValueError, match="参考模型 logits"):
            dpo_loss(
                pos_yes_logits=torch.tensor([1.0]),
                pos_no_logits=torch.tensor([0.0]),
                neg_yes_logits=torch.tensor([0.0]),
                neg_no_logits=torch.tensor([1.0]),
                ref_pos_yes_logits=torch.tensor([0.9]),
                ref_pos_no_logits=None,
                ref_neg_yes_logits=torch.tensor([0.1]),
                ref_neg_no_logits=None,
                reference_free=False,
            )

    def test_dpo_rejects_shape_mismatch_and_empty_inputs(self):
        """DPO should validate preference tensor shapes before reduction."""
        from qwen3_rerank_trainer.rl import dpo_loss

        with pytest.raises(ValueError, match="pos_no_logits shape"):
            dpo_loss(
                pos_yes_logits=torch.tensor([1.0, 1.2]),
                pos_no_logits=torch.tensor([0.0]),
                neg_yes_logits=torch.tensor([0.1]),
                neg_no_logits=torch.tensor([0.9]),
                reference_free=True,
            )

        with pytest.raises(ValueError, match="must not be empty"):
            dpo_loss(
                pos_yes_logits=torch.tensor([]),
                pos_no_logits=torch.tensor([]),
                neg_yes_logits=torch.tensor([0.1]),
                neg_no_logits=torch.tensor([0.9]),
                reference_free=True,
            )

    def test_reward_helpers_reject_empty_inputs(self):
        """Reward normalization should not produce NaNs for empty groups."""
        from qwen3_rerank_trainer.rl import compute_doc_level_rewards

        with pytest.raises(ValueError, match="must not be empty"):
            compute_doc_level_rewards(torch.tensor([]), torch.tensor([]))
