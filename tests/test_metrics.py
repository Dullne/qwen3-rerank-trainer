"""Tests for evaluation metrics."""

import pytest


class TestRankingMetrics:
    """Tests for ranking-based metrics."""

    def test_mrr(self):
        """Test MRR computation."""
        from qwen3_rerank_trainer.evaluation import mrr

        # First positive at position 1
        ranking = [0, 1, 2]
        positive_indices = {0}
        assert mrr(ranking, positive_indices) == 1.0

        # First positive at position 2
        ranking = [1, 0, 2]
        positive_indices = {0}
        assert mrr(ranking, positive_indices) == 0.5

        # First positive at position 3
        ranking = [1, 2, 0]
        positive_indices = {0}
        assert abs(mrr(ranking, positive_indices) - 1/3) < 1e-6

    def test_ap(self):
        """Test Average Precision."""
        from qwen3_rerank_trainer.evaluation import ap

        # Perfect ranking
        ranking = [0, 1, 2, 3]
        positive_indices = {0, 1}
        assert ap(ranking, positive_indices) == 1.0

        # All negatives first
        ranking = [2, 3, 0, 1]
        positive_indices = {0, 1}
        expected = (1/3 + 2/4) / 2
        assert abs(ap(ranking, positive_indices) - expected) < 1e-6

    def test_ndcg_at_k_binary(self):
        """Test binary NDCG@k."""
        from qwen3_rerank_trainer.evaluation import ndcg_at_k_binary

        # Perfect ranking
        ranking = [0, 1, 2, 3]
        positive_indices = {0, 1}
        assert ndcg_at_k_binary(ranking, positive_indices, k=2) == 1.0

    def test_precision_at_k(self):
        """Test Precision@k."""
        from qwen3_rerank_trainer.evaluation import precision_at_k

        ranking = [0, 2, 1, 3]
        positive_indices = {0, 1}

        # P@1 = 1/1 (doc 0 is positive)
        assert precision_at_k(ranking, positive_indices, k=1) == 1.0

        # P@2 = 1/2 (doc 0 positive, doc 2 negative)
        assert precision_at_k(ranking, positive_indices, k=2) == 0.5

        # P@3 = 2/3
        assert abs(precision_at_k(ranking, positive_indices, k=3) - 2/3) < 1e-6

    def test_recall_at_k(self):
        """Test Recall@k."""
        from qwen3_rerank_trainer.evaluation import recall_at_k

        ranking = [0, 2, 1, 3]
        positive_indices = {0, 1}

        # R@1 = 1/2 (found 1 of 2 positives)
        assert recall_at_k(ranking, positive_indices, k=1) == 0.5

        # R@3 = 2/2 (found both positives)
        assert recall_at_k(ranking, positive_indices, k=3) == 1.0

    def test_hit_at_k(self):
        """Test Hit@k (Success@k)."""
        from qwen3_rerank_trainer.evaluation import hit_at_k

        ranking = [2, 0, 1, 3]
        positive_indices = {0, 1}

        # Hit@1 = 0 (doc 2 is negative)
        assert hit_at_k(ranking, positive_indices, k=1) == 0.0

        # Hit@2 = 1 (doc 0 is positive)
        assert hit_at_k(ranking, positive_indices, k=2) == 1.0


class TestScoreBasedMetrics:
    """Tests for score-based metrics."""

    def test_mrr_from_scores(self):
        """Test MRR from scores."""
        from qwen3_rerank_trainer.evaluation import mrr_from_scores

        scores = [0.9, 0.1, 0.5]
        labels = [1, 0, 0]
        assert mrr_from_scores(scores, labels) == 1.0

        scores = [0.1, 0.9, 0.5]
        labels = [1, 0, 0]
        assert abs(mrr_from_scores(scores, labels) - 1/3) < 1e-6

    def test_ndcg_from_scores(self):
        """Test NDCG from scores."""
        from qwen3_rerank_trainer.evaluation import ndcg_from_scores

        # Perfect ranking
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        assert ndcg_from_scores(scores, labels, k=2) == 1.0


class TestComputeAllMetrics:
    """Tests for batch computation."""

    def test_compute_all_metrics(self):
        """Test computing all metrics at once."""
        from qwen3_rerank_trainer.evaluation import compute_all_metrics

        ranking = [0, 1, 2, 3]
        positive_indices = {0, 1}

        metrics = compute_all_metrics(ranking, positive_indices, ks=[1, 5, 10])

        assert 'MRR' in metrics
        assert 'AP' in metrics
        assert 'NDCG@1' in metrics
        assert 'NDCG@5' in metrics
        assert 'P@1' in metrics
        assert 'R@1' in metrics

    def test_aggregate_metrics(self):
        """Test aggregating multiple results."""
        from qwen3_rerank_trainer.evaluation import aggregate_metrics

        results = [
            {'MRR': 1.0, 'AP': 0.8},
            {'MRR': 0.5, 'AP': 0.6},
        ]

        agg = aggregate_metrics(results)
        assert agg['MRR'] == 0.75
        assert agg['AP'] == 0.7
