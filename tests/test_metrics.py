"""Tests for evaluation metrics."""

import asyncio
import json

import numpy as np
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
        from qwen3_rerank_trainer.evaluation import ndcg_at_k, ndcg_from_scores

        # Perfect ranking
        scores = [0.9, 0.8, 0.1, 0.05]
        labels = [1, 1, 0, 0]
        assert ndcg_from_scores(scores, labels, k=2) == 1.0

        # Graded labels: ranking the weaker positive first should be penalized.
        scores = [0.1, 0.9]
        labels = [2, 1]
        expected = ndcg_at_k([1, 0], {0: 2.0, 1: 1.0}, k=2)
        assert ndcg_from_scores(scores, labels, k=2) == expected
        assert ndcg_from_scores(scores, labels, k=2) < 1.0


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

    def test_mteb_aggregate_uses_graded_ndcg(self):
        """MTEB v2 qrels should keep graded relevance for NDCG."""
        from qwen3_rerank_trainer.evaluation import MTEBRerankEvaluator

        evaluator = MTEBRerankEvaluator(rerank_fn=lambda query, docs: ([1, 0], {}))
        metrics = evaluator._aggregate_results(
            [
                {
                    "ranking": [1, 0],
                    "positive_indices": {0, 1},
                    "relevance_scores": {0: 2.0, 1: 1.0},
                }
            ],
            ks=[2],
        )

        assert metrics["NDCG@2"] < 1.0

    def test_mteb_normalizes_ranked_documents_and_pairs(self):
        """Evaluator should accept local reranker outputs, not only API tuples."""
        from qwen3_rerank_trainer.evaluation import MTEBRerankEvaluator

        docs = ["alpha", "beta", "gamma"]

        evaluator = MTEBRerankEvaluator(rerank_fn=lambda query, documents: ["gamma", "alpha"])
        ranking, scores = evaluator._call_rerank("q", docs)
        assert ranking == [2, 0, 1]
        assert scores[2] > scores[0] > scores[1]

        evaluator = MTEBRerankEvaluator(
            rerank_fn=lambda query, documents: [("beta", 0.8), ("alpha", 0.2)]
        )
        ranking, scores = evaluator._call_rerank("q", docs)
        assert ranking == [1, 0, 2]
        assert scores[1] == 0.8

    def test_mteb_normalizes_partial_index_ranking(self):
        """Partial index rankings should append unreturned documents."""
        from qwen3_rerank_trainer.evaluation import MTEBRerankEvaluator

        evaluator = MTEBRerankEvaluator(rerank_fn=lambda query, documents: [1])
        ranking, scores = evaluator._call_rerank("q", ["alpha", "beta", "gamma"])

        assert ranking == [1, 0, 2]
        assert set(scores) == {0, 1, 2}

    def test_mteb_deduplicates_index_ranking(self):
        """Duplicate reranker indices should not inflate ranking metrics."""
        from qwen3_rerank_trainer.evaluation import MTEBRerankEvaluator, ap

        evaluator = MTEBRerankEvaluator(rerank_fn=lambda query, documents: ([0, 0], {0: 1.0}))
        ranking, scores = evaluator._call_rerank("q", ["alpha", "beta", "gamma"])

        assert ranking == [0, 1, 2]
        assert ap(ranking, {0}) == 1.0

    def test_mteb_v2_uses_query_field_when_text_missing(self):
        """MTEB v2 query rows may use query/question instead of text."""
        from qwen3_rerank_trainer.evaluation import MTEBRerankEvaluator

        seen_queries = []

        def fake_rerank(query, documents):
            seen_queries.append(query)
            return list(range(len(documents)))

        evaluator = MTEBRerankEvaluator(rerank_fn=fake_rerank)
        metrics = evaluator._evaluate_ranking_dataset(
            queries={"q1": {"query": "query in query field"}},
            corpus={
                "p1": {"text": "positive"},
                "n1": {"text": "negative one"},
                "n2": {"text": "negative two"},
            },
            qrels={"q1": {"p1": 1}},
            top_ranked={},
            max_samples=None,
            shuffle_seed=42,
            ks=[1],
            progress_callback=None,
            show_progress=False,
        )

        assert metrics["num_evaluated"] == 1
        assert seen_queries == ["query in query field"]

    def test_mteb_v2_fallback_without_top_ranked_includes_negatives(self):
        """When top_ranked is absent, candidates should include corpus negatives."""
        from qwen3_rerank_trainer.evaluation import MTEBRerankEvaluator

        seen_docs = []

        def fake_rerank(query, documents):
            seen_docs.extend(documents)
            return list(range(len(documents)))

        evaluator = MTEBRerankEvaluator(rerank_fn=fake_rerank)
        metrics = evaluator._evaluate_ranking_dataset(
            queries={"q1": {"text": "query"}},
            corpus={
                "p1": {"text": "positive"},
                "n1": {"text": "negative one"},
                "n2": {"text": "negative two"},
            },
            qrels={"q1": {"p1": 1}},
            top_ranked={},
            max_samples=None,
            shuffle_seed=42,
            ks=[1],
            progress_callback=None,
            show_progress=False,
        )

        assert metrics["num_evaluated"] == 1
        assert "positive" in seen_docs
        assert any(doc.startswith("negative") for doc in seen_docs)

    def test_report_json_handles_numpy_values_and_zero_score_models(self, tmp_path):
        """Reports should serialize array scalars and keep zero-valued models ranked."""
        from qwen3_rerank_trainer.evaluation.report import (
            ReportConfig,
            generate_report,
            save_results_json,
        )

        json_path = save_results_json(
            {
                "Task": {
                    "NDCG@10": np.float32(0.0),
                    "curve": np.array([0.0, 1.0]),
                    "labels": {"b", "a"},
                }
            },
            tmp_path / "raw" / "results.json",
        )
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["results"]["Task"]["NDCG@10"] == 0.0
        assert payload["results"]["Task"]["curve"] == [0.0, 1.0]
        assert payload["results"]["Task"]["labels"] == ["a", "b"]

        paths = generate_report(
            [
                {
                    "model": "zero-model",
                    "results": {"Task": {"NDCG@10": 0.0, "num_evaluated": 1}},
                }
            ],
            tmp_path / "report",
            ReportConfig(metrics=["NDCG@10"], primary_metric="NDCG@10"),
        )
        markdown = paths["markdown"].read_text(encoding="utf-8")
        assert "zero-model" in markdown
        assert "| 1 | zero-model | 0.00% | - |" in markdown

    def test_api_and_gpu_worker_validations(self, monkeypatch):
        """Invalid concurrency values should fail before launching work."""
        from qwen3_rerank_trainer.evaluation import api_client
        from qwen3_rerank_trainer.evaluation.gpu_utils import run_with_gpu_balance

        with pytest.raises(ValueError, match="max_concurrency"):
            api_client.call_rerank_batch(
                [],
                endpoint="http://127.0.0.1:9/rerank",
                max_concurrency=0,
                show_progress=False,
            )

        monkeypatch.setattr(api_client, "AIOHTTP_AVAILABLE", True)
        with pytest.raises(ValueError, match="max_concurrency"):
            api_client.call_rerank_batch(
                [("q", ["doc"])],
                endpoint="http://127.0.0.1:9/rerank",
                max_concurrency=0,
                show_progress=False,
            )

        with pytest.raises(ValueError, match="total_workers"):
            run_with_gpu_balance(
                models=["m1"],
                gpu_info={"m1": 0},
                total_workers=0,
                eval_func=lambda model: {},
                verbose=False,
            )

    def test_two_stage_prepare_rerank_model_from_config(self):
        """Two-stage evaluator should use the in-module API rerank model."""
        from qwen3_rerank_trainer.evaluation.two_stage_eval import APIRerankModel, TwoStageEvaluator

        evaluator = TwoStageEvaluator(
            rerank_config={
                "api_base": "http://localhost:9997/v1",
                "model_name": "Qwen3-Reranker",
            }
        )

        model = evaluator._prepare_rerank_model()
        assert isinstance(model, APIRerankModel)
        assert model.endpoint == "http://localhost:9997/v1/rerank"

    def test_api_safe_retry_keeps_timeout_retryable(self):
        """Timeouts raised by call_rerank_async should be retried, not wrapped away."""
        from qwen3_rerank_trainer.evaluation import api_client

        class TimeoutSession:
            def __init__(self):
                self.calls = 0

            def post(self, *args, **kwargs):
                self.calls += 1
                raise asyncio.TimeoutError()

        session = TimeoutSession()
        result = asyncio.run(
            api_client.call_rerank_async_safe(
                "q",
                ["doc"],
                endpoint="http://127.0.0.1:9/rerank",
                session=session,
                retries=2,
                backoff=0,
                jitter=0,
            )
        )

        assert result == ([], {})
        assert session.calls == 3

    def test_sync_api_wrapper_can_run_inside_existing_event_loop(self, monkeypatch):
        """Notebook/service users should get a result instead of nested-loop errors."""
        from qwen3_rerank_trainer.evaluation import api_client

        async def fake_call(query, documents, endpoint, model="Qwen3-Reranker-4B", timeout=30):
            return [0], {0: 1.0}

        monkeypatch.setattr(api_client, "call_rerank_async", fake_call)

        async def runner():
            return api_client.call_rerank("q", ["doc"], "http://127.0.0.1:9/rerank")

        assert asyncio.run(runner()) == ([0], {0: 1.0})
