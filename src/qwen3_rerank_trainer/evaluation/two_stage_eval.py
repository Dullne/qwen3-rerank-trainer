"""
两阶段评估：Embedding 检索 + Rerank 重排序

支持 MTEB Retrieval 任务的两阶段评估：
1. 第一阶段：使用 Embedding 模型进行向量检索
2. 第二阶段：使用 Rerank 模型对检索结果重排序

使用示例:
    from qwen3_rerank_trainer.evaluation import TwoStageEvaluator, set_proxy

    # 设置代理（可选）
    set_proxy("http://proxy:port")

    # 创建评估器
    evaluator = TwoStageEvaluator(
        embedding_model=my_embedding_model,
        rerank_model=my_rerank_model,
    )

    # 评估
    results = evaluator.evaluate(
        tasks=["T2Retrieval", "MMarcoRetrieval"],
        output_dir="eval_output",
    )
"""

import os
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# 数据集配置
# ============================================================================

RETRIEVAL_DATASETS = {
    # 中文数据集
    "T2Retrieval": {"split": "dev", "lang": "zh"},
    "MMarcoRetrieval": {"split": "dev", "lang": "zh"},
    "DuRetrieval": {"split": "dev", "lang": "zh"},
    "CovidRetrieval": {"split": "test", "lang": "zh"},
    "CmedqaRetrieval": {"split": "dev", "lang": "zh"},
    "EcomRetrieval": {"split": "dev", "lang": "zh"},
    "MedicalRetrieval": {"split": "dev", "lang": "zh"},
    "VideoRetrieval": {"split": "test", "lang": "zh"},
}

RERANKING_DATASETS = {
    "T2Reranking": {"split": "dev", "lang": "zh"},
    "MMarcoReranking": {"split": "dev", "lang": "zh"},
    "CMedQAv1-reranking": {"split": "test", "lang": "zh"},
    "CMedQAv2-reranking": {"split": "test", "lang": "zh"},
}

ALL_DATASETS = {**RETRIEVAL_DATASETS, **RERANKING_DATASETS}


# ============================================================================
# 模型协议
# ============================================================================

class EmbeddingModelProtocol(Protocol):
    """Embedding 模型协议"""

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        **kwargs,
    ) -> Any:
        """编码文本为向量"""
        ...


class RerankModelProtocol(Protocol):
    """Rerank 模型协议"""

    def rerank(
        self,
        query: str,
        documents: List[str],
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        重排序文档

        Returns:
            排序结果列表，每个元素包含 'corpus_id' 和 'score'
        """
        ...


# ============================================================================
# 评估配置
# ============================================================================

@dataclass
class TwoStageEvalConfig:
    """两阶段评估配置"""
    top_k: int = 100  # 第一阶段检索的 top-k
    rerank_top_k: int = 100  # 重排序的 top-k
    max_samples: Optional[int] = None  # 最大样本数
    hub: str = "modelscope"  # 数据集来源 (modelscope/huggingface)
    save_predictions: bool = True  # 是否保存预测结果
    overwrite_results: bool = True  # 是否覆盖已有结果


# ============================================================================
# 两阶段评估器
# ============================================================================

class TwoStageEvaluator:
    """
    两阶段评估器：Embedding 检索 + Rerank 重排序

    Note:
        此类需要安装 evalscope 和 mteb 依赖：
        pip install evalscope mteb
    """

    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        rerank_model: Optional[Any] = None,
        embedding_config: Optional[Dict[str, Any]] = None,
        rerank_config: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化评估器

        Args:
            embedding_model: Embedding 模型实例
            rerank_model: Rerank 模型实例
            embedding_config: Embedding API 配置（用于 evalscope）
            rerank_config: Rerank API 配置（用于 evalscope）

        Note:
            可以提供模型实例或 API 配置，二者选一
        """
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.embedding_config = embedding_config
        self.rerank_config = rerank_config

    def evaluate(
        self,
        tasks: List[str],
        output_dir: str,
        config: Optional[TwoStageEvalConfig] = None,
    ) -> Dict[str, Any]:
        """
        运行两阶段评估

        Args:
            tasks: 评估任务列表
            output_dir: 输出目录
            config: 评估配置

        Returns:
            评估结果字典
        """
        if config is None:
            config = TwoStageEvalConfig()

        try:
            import mteb
            from evalscope.backend.rag_eval.utils.embedding import EmbeddingModel
            from evalscope.backend.rag_eval.cmteb.base import TaskBase
        except ImportError:
            logger.error(
                "evalscope and mteb are required for two-stage evaluation. "
                "Install with: pip install evalscope mteb"
            )
            return {"error": "Missing dependencies: evalscope, mteb"}

        # 准备模型
        embedding_model = self._prepare_embedding_model(EmbeddingModel)
        rerank_model = self._prepare_rerank_model()

        if embedding_model is None or rerank_model is None:
            return {"error": "Failed to prepare models"}

        # 创建输出目录
        stage1_path = os.path.join(output_dir, "stage1")
        stage2_path = os.path.join(output_dir, "stage2")
        os.makedirs(stage1_path, exist_ok=True)
        os.makedirs(stage2_path, exist_ok=True)

        all_results = {}

        for task_name in tasks:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {task_name}")
            logger.info(f"{'='*60}")

            try:
                result = self._evaluate_task(
                    task_name=task_name,
                    embedding_model=embedding_model,
                    rerank_model=rerank_model,
                    stage1_path=stage1_path,
                    stage2_path=stage2_path,
                    config=config,
                    TaskBase=TaskBase,
                    mteb=mteb,
                )
                all_results[task_name] = result
            except Exception as e:
                logger.error(f"Error evaluating {task_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[task_name] = {"error": str(e)}

        return all_results

    def _prepare_embedding_model(self, EmbeddingModel):
        """准备 Embedding 模型"""
        if self.embedding_model is not None:
            return self.embedding_model

        if self.embedding_config is not None:
            return EmbeddingModel.load(**self.embedding_config)

        logger.error("No embedding model or config provided")
        return None

    def _prepare_rerank_model(self):
        """准备 Rerank 模型"""
        if self.rerank_model is not None:
            return self.rerank_model

        if self.rerank_config is not None:
            # 使用 API 配置创建模型
            from .api_rerank_model import APIRerankModel
            return APIRerankModel(**self.rerank_config)

        logger.error("No rerank model or config provided")
        return None

    def _evaluate_task(
        self,
        task_name: str,
        embedding_model,
        rerank_model,
        stage1_path: str,
        stage2_path: str,
        config: TwoStageEvalConfig,
        TaskBase,
        mteb,
    ) -> Dict[str, Any]:
        """评估单个任务"""
        # 获取任务配置
        task_config = ALL_DATASETS.get(task_name, {})
        eval_split = task_config.get("split", "dev")

        # 使用 evalscope 的 TaskBase 获取任务
        task = TaskBase.get_task(task_name)

        # 加载数据
        is_retrieval = "Retrieval" in task_name
        if is_retrieval and config.max_samples:
            # 对于 Retrieval 任务，先加载完整数据，再限制 queries
            task.load_data(eval_splits=[eval_split], hub=config.hub)

            if eval_split in task.queries:
                original_queries = task.queries[eval_split]
                original_qrels = task.relevant_docs[eval_split]

                limited_query_ids = list(original_queries.keys())[:config.max_samples]
                task.queries[eval_split] = {qid: original_queries[qid] for qid in limited_query_ids}
                task.relevant_docs[eval_split] = {
                    qid: original_qrels[qid]
                    for qid in limited_query_ids
                    if qid in original_qrels
                }

                logger.info(
                    f"Queries: {len(task.queries[eval_split])}, "
                    f"Corpus: {len(task.corpus.get(eval_split, task.corpus))}"
                )

        evaluation = mteb.MTEB(tasks=[task])

        # 第一阶段：Embedding 检索
        logger.info("Stage 1: Embedding retrieval...")
        run_kwargs = {
            "save_predictions": config.save_predictions,
            "output_folder": stage1_path,
            "overwrite_results": config.overwrite_results,
            "eval_splits": [eval_split],
            "top_k": config.top_k,
            "hub": config.hub,
        }
        if config.max_samples and not is_retrieval:
            run_kwargs["limits"] = config.max_samples

        stage1_results = evaluation.run(embedding_model, **run_kwargs)

        # 第二阶段：Rerank 重排序
        logger.info(f"Stage 2: Reranking...")
        previous_results_path = os.path.join(
            stage1_path,
            f"{task_name}_default_predictions.json"
        )

        if os.path.exists(previous_results_path):
            run_kwargs2 = {
                "top_k": config.rerank_top_k,
                "save_predictions": config.save_predictions,
                "output_folder": stage2_path,
                "previous_results": previous_results_path,
                "overwrite_results": config.overwrite_results,
                "eval_splits": [eval_split],
                "hub": config.hub,
            }
            if config.max_samples and not is_retrieval:
                run_kwargs2["limits"] = config.max_samples

            stage2_results = evaluation.run(rerank_model, **run_kwargs2)
            return stage2_results
        else:
            logger.warning(f"Stage 1 predictions not found: {previous_results_path}")
            return stage1_results


# ============================================================================
# API Rerank 模型（用于 evalscope 集成）
# ============================================================================

class APIRerankModel:
    """
    API 调用的 Rerank 模型

    兼容 evalscope 的 rerank 模型接口
    """

    def __init__(
        self,
        api_base: str,
        model_name: str,
        batch_size: int = 32,
        max_concurrency: int = 8,
        timeout: int = 60,
    ):
        """
        初始化 API Rerank 模型

        Args:
            api_base: API 基础地址（如 http://host:port/v1）
            model_name: 模型名称
            batch_size: 每批处理的文档数
            max_concurrency: 最大并发数
            timeout: 请求超时时间
        """
        self.api_base = api_base.rstrip('/')
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.timeout = timeout

        # 构建 rerank endpoint
        if not self.api_base.endswith('/rerank'):
            if not self.api_base.endswith('/v1'):
                self.api_base += '/v1'
            self.endpoint = self.api_base + '/rerank'
        else:
            self.endpoint = self.api_base

    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回 top-k 结果

        Returns:
            排序结果列表，每个元素包含 'corpus_id' 和 'score'
        """
        import requests

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": documents,
        }
        if top_k:
            payload["top_n"] = top_k

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            # 转换为 evalscope 期望的格式
            results = []
            for item in data.get("results", []):
                results.append({
                    "corpus_id": item.get("index", 0),
                    "score": item.get("relevance_score", 0.0),
                })
            return results

        except Exception as e:
            logger.error(f"Rerank API call failed: {e}")
            # 返回原始顺序
            return [{"corpus_id": i, "score": 0.0} for i in range(len(documents))]


# ============================================================================
# 便捷函数
# ============================================================================

def run_two_stage_eval(
    embedding_config: Dict[str, Any],
    rerank_config: Dict[str, Any],
    tasks: List[str],
    output_dir: str,
    top_k: int = 100,
    max_samples: Optional[int] = None,
    hub: str = "modelscope",
    proxy: Optional[str] = None,
) -> Dict[str, Any]:
    """
    运行两阶段评估

    Args:
        embedding_config: Embedding 模型配置
            - model_name: 模型名称
            - api_base: API 地址
            - api_key: API 密钥（可选）
        rerank_config: Rerank 模型配置
            - model_name: 模型名称
            - api_base: API 地址
            - batch_size: 批大小（可选）
        tasks: 评估任务列表
        output_dir: 输出目录
        top_k: 检索/重排序的 top-k
        max_samples: 最大样本数
        hub: 数据集来源
        proxy: 代理地址

    Returns:
        评估结果字典
    """
    if proxy:
        from .mteb_runner import set_proxy
        set_proxy(proxy)

    evaluator = TwoStageEvaluator(
        embedding_config=embedding_config,
        rerank_config=rerank_config,
    )

    config = TwoStageEvalConfig(
        top_k=top_k,
        rerank_top_k=top_k,
        max_samples=max_samples,
        hub=hub,
    )

    return evaluator.evaluate(
        tasks=tasks,
        output_dir=output_dir,
        config=config,
    )
