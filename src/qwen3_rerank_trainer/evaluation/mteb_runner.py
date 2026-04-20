"""
MTEB Reranking 评估运行器

提供 MTEB reranking 数据集的评估功能：
- 支持代理配置
- 支持本地和 MTEB 数据集
- 支持批量评估多个数据集
- 支持异步批量处理（batch_size 控制每次请求文档数）
- 支持多模型并行评估（model_workers 控制并发数）

使用示例:
    from qwen3_rerank_trainer.evaluation import MTEBRerankEvaluator, set_proxy

    # 设置代理（可选）
    set_proxy("http://proxy:port")

    # 创建评估器（支持批量处理）
    evaluator = MTEBRerankEvaluator(
        reranker=my_reranker,  # 实现 rerank(query, docs) 接口
        batch_size=50,         # 每次请求最多 50 个文档（避免 OOM）
        workers=8,             # 8 个并发请求
    )

    # 评估单个数据集
    results = evaluator.evaluate("T2Reranking", split="dev", max_samples=1000)

    # 评估多个数据集
    results = evaluator.evaluate_multiple(["T2Reranking", "MMarcoReranking"])

    # 多模型并行评估
    from qwen3_rerank_trainer.evaluation import evaluate_multiple_models
    results = evaluate_multiple_models(
        rerankers={"model_a": reranker_a, "model_b": reranker_b},
        task_names=["T2Reranking"],
        model_workers=2,  # 2 个模型同时评估
    )
"""

import os
import random
import logging
import asyncio
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .metrics import (
    mrr, ap, ndcg_at_k_binary,
    precision_at_k, recall_at_k,
    compute_all_metrics, aggregate_metrics,
)

# 可选依赖
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False

logger = logging.getLogger(__name__)


# ============================================================================
# 代理配置
# ============================================================================

def set_proxy(proxy_url: str) -> None:
    """
    设置 HTTP/HTTPS 代理

    Args:
        proxy_url: 代理地址，如 "http://172.16.10.3:10810"

    Note:
        必须在 import mteb 之前调用此函数
    """
    os.environ['http_proxy'] = proxy_url
    os.environ['https_proxy'] = proxy_url
    os.environ['HTTP_PROXY'] = proxy_url
    os.environ['HTTPS_PROXY'] = proxy_url
    logger.info(f"Proxy set to: {proxy_url}")


def clear_proxy() -> None:
    """清除代理设置"""
    for key in ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']:
        os.environ.pop(key, None)
    logger.info("Proxy cleared")


# ============================================================================
# 数据集配置
# ============================================================================

# MTEB Reranking 数据集配置
RERANKING_DATASETS = {
    # 中文数据集
    "T2Reranking": {"split": "dev", "lang": "zh"},
    "MMarcoReranking": {"split": "dev", "lang": "zh"},
    "CMedQAv1-reranking": {"split": "test", "lang": "zh"},
    "CMedQAv2-reranking": {"split": "test", "lang": "zh"},
    # 英文数据集
    "AskUbuntuDupQuestions": {"split": "test", "lang": "en"},
    "MindSmallReranking": {"split": "test", "lang": "en"},
    "SciDocsRR": {"split": "test", "lang": "en"},
    "StackOverflowDupQuestions": {"split": "test", "lang": "en"},
    "WebLINXCandidatesReranking": {"split": "test", "lang": "en"},
    "BuiltBenchReranking": {"split": "test", "lang": "en"},
    # 多语言数据集
    "MIRACLReranking": {"split": "test", "lang": "multilingual"},
    "WikipediaRerankingMultilingual": {"split": "test", "lang": "multilingual"},
    "ESCIReranking": {"split": "test", "lang": "multilingual"},  # eng, jpn, spa
    # 其他语言
    "RuBQReranking": {"split": "test", "lang": "ru"},
    "VoyageMMarcoReranking": {"split": "test", "lang": "ja"},
    "AlloprofReranking": {"split": "test", "lang": "fr"},
    "SyntecReranking": {"split": "test", "lang": "fr"},
    "NamaaMrTydiReranking": {"split": "test", "lang": "ar"},
    # 代码数据集
    "CodeRAGLibraryDocumentationSolutions": {"split": "test", "lang": "code"},
    "CodeRAGOnlineTutorials": {"split": "test", "lang": "code"},
    "CodeRAGProgrammingSolutions": {"split": "test", "lang": "code"},
    "CodeRAGStackoverflowPosts": {"split": "test", "lang": "code"},
}

# 数据集组
DATASET_GROUPS = {
    "chinese": ["T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking"],
    "cn": ["T2Reranking", "MMarcoReranking", "CMedQAv1-reranking", "CMedQAv2-reranking"],
    "english": ["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR",
                "StackOverflowDupQuestions", "WebLINXCandidatesReranking", "BuiltBenchReranking"],
    "en": ["AskUbuntuDupQuestions", "MindSmallReranking", "SciDocsRR",
           "StackOverflowDupQuestions", "WebLINXCandidatesReranking", "BuiltBenchReranking"],
    "multilingual": ["MIRACLReranking", "WikipediaRerankingMultilingual", "ESCIReranking"],
    "multi": ["MIRACLReranking", "WikipediaRerankingMultilingual", "ESCIReranking"],
    "other": ["RuBQReranking", "VoyageMMarcoReranking", "AlloprofReranking",
              "SyntecReranking", "NamaaMrTydiReranking"],
    "code": ["CodeRAGLibraryDocumentationSolutions", "CodeRAGOnlineTutorials",
             "CodeRAGProgrammingSolutions", "CodeRAGStackoverflowPosts"],
    "all": list(RERANKING_DATASETS.keys()),
}


def expand_dataset_names(datasets: List[str]) -> List[str]:
    """展开数据集组名"""
    expanded = []
    for ds in datasets:
        if ds in DATASET_GROUPS:
            expanded.extend(DATASET_GROUPS[ds])
        else:
            expanded.append(ds)
    # 去重保持顺序
    seen = set()
    result = []
    for ds in expanded:
        if ds not in seen:
            seen.add(ds)
            result.append(ds)
    return result


# ============================================================================
# Reranker 协议
# ============================================================================

@dataclass
class RerankResult:
    """Rerank 结果"""
    ranking: List[int]  # 排序后的文档索引
    scores: Dict[int, float]  # 文档索引 -> 分数


class BaseRerankerProtocol:
    """Reranker 协议（用于类型提示）"""

    def rerank(
        self,
        query: str,
        documents: List[str],
        **kwargs,
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        重排序文档

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            (ranking, scores): 排序后的索引列表和分数字典
        """
        raise NotImplementedError


# ============================================================================
# 本地数据集
# ============================================================================

@dataclass
class LocalSample:
    """本地数据集样本"""
    query: str
    positive: List[str]
    negative: List[str]


class LocalDataset:
    """
    本地数据集包装器

    支持格式:
    - {query, positives, negatives} - 复数形式
    - {query, positive, negative} - 单数形式（MTEB 标准）
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.samples = []
        for record in data:
            query = record.get('query', '')

            # 支持复数形式
            if 'positives' in record:
                positive = record['positives']
                negative = record.get('negatives', [])
            else:
                positive = record.get('positive', [])
                negative = record.get('negative', [])
                if isinstance(positive, str):
                    positive = [positive]
                if isinstance(negative, str):
                    negative = [negative]

            if query and positive:
                self.samples.append(LocalSample(
                    query=query,
                    positive=positive,
                    negative=negative,
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> LocalSample:
        return self.samples[idx]

    @classmethod
    def from_jsonl(cls, file_path: str) -> "LocalDataset":
        """从 jsonl 文件加载"""
        import json
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        return cls(data)


# ============================================================================
# 评估器
# ============================================================================

class MTEBRerankEvaluator:
    """
    MTEB Reranking 评估器

    Args:
        reranker: Reranker 实例，需实现 rerank(query, docs) 方法
        batch_rerank_fn: 可选的批量 rerank 函数，签名为:
            async def batch_rerank(items: List[Tuple[str, List[str]]]) -> List[Tuple[List[int], Dict[int, float]]]
        batch_size: 每次请求的最大文档数（避免 OOM），默认 50
        workers: 并发请求数，默认 8
    """

    def __init__(
        self,
        reranker: Optional[Any] = None,
        rerank_fn: Optional[Callable] = None,
        batch_rerank_fn: Optional[Callable] = None,
        batch_size: int = 50,
        workers: int = 8,
    ):
        """
        初始化评估器

        Args:
            reranker: Reranker 实例（需实现 rerank 方法）
            rerank_fn: 单条 rerank 函数（替代 reranker）
            batch_rerank_fn: 批量 rerank 函数（异步，用于加速）
            batch_size: 每次请求的最大文档数（避免 OOM）
            workers: 并发请求数
        """
        self.reranker = reranker
        self.rerank_fn = rerank_fn
        self.batch_rerank_fn = batch_rerank_fn
        self.batch_size = batch_size
        self.workers = workers

        if reranker is None and rerank_fn is None and batch_rerank_fn is None:
            raise ValueError("Must provide reranker, rerank_fn, or batch_rerank_fn")

    def _call_rerank(
        self,
        query: str,
        documents: List[str],
    ) -> Tuple[List[int], Dict[int, float]]:
        """调用 rerank"""
        if self.rerank_fn is not None:
            return self.rerank_fn(query, documents)
        elif self.reranker is not None:
            return self.reranker.rerank(query, documents)
        else:
            raise ValueError("No rerank function available")

    def evaluate(
        self,
        task_name: str,
        split: Optional[str] = None,
        max_samples: Optional[int] = None,
        shuffle_seed: int = 42,
        ks: List[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        评估 MTEB 数据集

        Args:
            task_name: MTEB 任务名称
            split: 数据集划分（默认自动选择）
            max_samples: 最大样本数
            shuffle_seed: 打乱文档的随机种子
            ks: 评估的 k 值列表
            progress_callback: 进度回调函数 (current, total)
            show_progress: 是否显示 tqdm 进度条

        Returns:
            评估指标字典
        """
        if ks is None:
            ks = [1, 5, 10]

        # 确定 split
        if split is None:
            split = RERANKING_DATASETS.get(task_name, {}).get("split", "dev")

        # 加载 MTEB 数据集
        try:
            import mteb
            task = mteb.get_task(task_name)
            task.load_data()
        except Exception as e:
            logger.error(f"Failed to load MTEB task {task_name}: {e}")
            return {"error": str(e)}

        # 提取数据集
        dataset = self._extract_dataset(task, split)
        if dataset is None:
            return {"error": f"Failed to extract dataset for {task_name}/{split}"}

        # 检查数据集格式并选择评估方法
        # 格式1: HF Dataset with 'positive' column (旧版 Reranking 格式)
        if hasattr(dataset, 'column_names') and 'positive' in dataset.column_names:
            return self._evaluate_reranking_dataset(
                dataset, max_samples, shuffle_seed, ks, progress_callback,
                progress_desc=task_name, show_progress=show_progress
            )

        # 格式2: dict with queries/corpus/relevant_docs/top_ranked (MTEB v2 格式)
        if isinstance(dataset, dict):
            queries = dataset.get('queries', {})
            corpus = dataset.get('corpus', {})
            qrels = dataset.get('qrels', dataset.get('relevant_docs', {}))
            top_ranked = dataset.get('top_ranked', {})

            if queries and corpus and qrels:
                logger.info(f"Detected MTEB v2 format: {len(queries)} queries, {len(corpus)} corpus docs")
                return self._evaluate_ranking_dataset(
                    queries, corpus, qrels, top_ranked,
                    max_samples, shuffle_seed, ks, progress_callback,
                    progress_desc=task_name, show_progress=show_progress
                )

        logger.error(f"Unsupported dataset format for {task_name}: {type(dataset)}")
        return {"error": "Unsupported dataset format"}

    def _extract_dataset(self, task, split: str):
        """从 MTEB task 中提取数据集"""
        if not hasattr(task, 'dataset'):
            return None

        raw_dataset = task.dataset

        if hasattr(raw_dataset, 'keys'):
            if 'default' in raw_dataset and split in raw_dataset['default']:
                return raw_dataset['default'][split]
            elif split in raw_dataset:
                return raw_dataset[split]
            else:
                first_key = list(raw_dataset.keys())[0]
                if split in raw_dataset[first_key]:
                    return raw_dataset[first_key][split]
                else:
                    return raw_dataset[first_key]
        else:
            return raw_dataset

    def _evaluate_reranking_dataset(
        self,
        dataset,
        max_samples: Optional[int],
        shuffle_seed: int,
        ks: List[int],
        progress_callback: Optional[Callable],
        progress_desc: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """评估 Reranking 格式数据集

        支持两种模式：
        1. 批量异步模式（有 batch_rerank_fn）：分块处理，高并发
        2. 逐条同步模式（无 batch_rerank_fn）：逐条处理

        Args:
            dataset: 数据集
            max_samples: 最大样本数
            shuffle_seed: 随机种子
            ks: 评估的 k 值列表
            progress_callback: 进度回调函数
            progress_desc: 进度条描述
            show_progress: 是否显示进度条
        """
        num_samples = len(dataset)
        if max_samples:
            num_samples = min(max_samples, num_samples)

        logger.info(f"Evaluating {num_samples} samples (batch_size={self.batch_size}, workers={self.workers})")

        # 如果有批量处理函数，使用异步批量模式
        if self.batch_rerank_fn is not None:
            return self._evaluate_reranking_dataset_batched(
                dataset, num_samples, shuffle_seed, ks, progress_callback
            )

        # 否则使用逐条处理模式（支持多线程）
        all_results = []

        # 创建进度条（如果 tqdm 可用）
        use_tqdm = show_progress and TQDM_AVAILABLE and num_samples > 1
        pbar = None
        if use_tqdm:
            import sys
            desc = progress_desc or "Evaluating"
            pbar = tqdm(total=num_samples, desc=desc, unit="sample", file=sys.stdout, leave=False)

        if self.workers > 1:
            # 多线程并发处理
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = {}
                for i in range(num_samples):
                    sample = dataset[i]
                    future = executor.submit(self._evaluate_single_sample, sample, i, shuffle_seed)
                    futures[future] = i

                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_results.append(result)
                    if pbar:
                        pbar.update(1)
                    if progress_callback:
                        progress_callback(len(all_results), num_samples)
        else:
            # 单线程顺序处理
            for i in range(num_samples):
                sample = dataset[i]
                result = self._evaluate_single_sample(sample, i, shuffle_seed)
                if result is not None:
                    all_results.append(result)

                if pbar:
                    pbar.update(1)
                if progress_callback:
                    progress_callback(i + 1, num_samples)

        if pbar:
            pbar.close()

        # 聚合结果
        return self._aggregate_results(all_results, ks)

    def _evaluate_reranking_dataset_batched(
        self,
        dataset,
        num_samples: int,
        shuffle_seed: int,
        ks: List[int],
        progress_callback: Optional[Callable],
    ) -> Dict[str, float]:
        """批量异步评估 Reranking 数据集

        将大文档列表拆分成多个小批次，避免 OOM。
        """
        # 准备批量请求（大文档列表会拆分成多个请求）
        batch_items = []
        batch_mapping = []  # (sample_idx, offset) 用于合并结果
        sample_data = []

        for i in range(num_samples):
            sample = dataset[i]
            query = sample['query']
            positive = sample['positive']
            negative = sample['negative']

            # 合并正负样本
            if isinstance(positive, list):
                docs = positive + (negative if isinstance(negative, list) else [negative])
                num_positive = len(positive)
            else:
                docs = [positive] + (negative if isinstance(negative, list) else [negative])
                num_positive = 1

            # 打乱文档顺序，消除 tie-breaking 偏见
            indices = list(range(len(docs)))
            random.seed(shuffle_seed + i)
            random.shuffle(indices)
            shuffled_docs = [docs[idx] for idx in indices]

            # 记录正例在打乱后的新位置
            positive_indices = set(indices.index(j) for j in range(num_positive))

            # 分批处理大文档列表，避免 OOM
            if len(shuffled_docs) <= self.batch_size:
                batch_items.append((query, shuffled_docs))
                batch_mapping.append((i, 0))
            else:
                for start in range(0, len(shuffled_docs), self.batch_size):
                    end = min(start + self.batch_size, len(shuffled_docs))
                    batch_items.append((query, shuffled_docs[start:end]))
                    batch_mapping.append((i, start))

            sample_data.append((shuffled_docs, positive_indices))

        logger.info(f"批量调用 Rerank（{len(batch_items)} 个请求，{num_samples} 个样本，并发={self.workers}，批大小={self.batch_size}）")

        # 调用批量 rerank 函数
        try:
            # 支持同步和异步两种批量函数
            if asyncio.iscoroutinefunction(self.batch_rerank_fn):
                raw_batch_results = asyncio.run(self.batch_rerank_fn(batch_items))
            else:
                raw_batch_results = self.batch_rerank_fn(batch_items)
        except Exception as e:
            logger.error(f"批量 Rerank 调用失败: {e}")
            return {
                "num_queries": num_samples,
                "num_evaluated": 0,
                "MRR": 0.0, "AP": 0.0,
                **{f"NDCG@{k}": 0.0 for k in ks},
                **{f"P@{k}": 0.0 for k in ks},
                **{f"R@{k}": 0.0 for k in ks},
            }

        # 合并分批结果
        merged_scores = [{} for _ in range(num_samples)]
        for batch_idx, (_, scores_map) in enumerate(raw_batch_results):
            sample_idx, offset = batch_mapping[batch_idx]
            for local_idx, score in scores_map.items():
                merged_scores[sample_idx][offset + local_idx] = score

        # 生成最终排序结果
        all_results = []
        for i, scores_map in enumerate(merged_scores):
            if scores_map:
                # 排序规则：分数降序，分数相同时索引降序（与 API tie-breaking 一致）
                ranking = sorted(scores_map.keys(), key=lambda x: (scores_map[x], x), reverse=True)
                _, positive_indices = sample_data[i]
                all_results.append({
                    "ranking": ranking,
                    "scores": scores_map,
                    "positive_indices": positive_indices,
                    "num_docs": len(sample_data[i][0]),
                })

        if progress_callback:
            progress_callback(num_samples, num_samples)

        return self._aggregate_results(all_results, ks)

    def _evaluate_single_sample(
        self,
        sample: Dict[str, Any],
        sample_idx: int,
        shuffle_seed: int,
    ) -> Optional[Dict[str, Any]]:
        """评估单个样本"""
        query = sample['query']
        positive = sample['positive']
        negative = sample['negative']

        # 合并文档
        if isinstance(positive, list):
            docs = positive + (negative if isinstance(negative, list) else [negative])
            num_positive = len(positive)
        else:
            docs = [positive] + (negative if isinstance(negative, list) else [negative])
            num_positive = 1

        if len(docs) < 2:
            return None

        # 打乱文档顺序（消除位置偏见）
        indices = list(range(len(docs)))
        random.seed(shuffle_seed + sample_idx)
        random.shuffle(indices)
        shuffled_docs = [docs[idx] for idx in indices]

        # 记录正例在打乱后的位置
        positive_indices = set(indices.index(j) for j in range(num_positive))

        try:
            ranking, scores = self._call_rerank(query, shuffled_docs)
            return {
                "ranking": ranking,
                "scores": scores,
                "positive_indices": positive_indices,
                "num_docs": len(docs),
            }
        except Exception as e:
            logger.warning(f"Rerank failed for sample {sample_idx}: {e}")
            return None

    def _evaluate_ranking_dataset(
        self,
        queries: Dict[str, Any],
        corpus: Dict[str, Any],
        qrels: Dict[str, Dict[str, int]],
        top_ranked: Dict[str, List[str]],
        max_samples: Optional[int],
        shuffle_seed: int,
        ks: List[int],
        progress_callback: Optional[Callable],
        progress_desc: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """评估 MTEB v2 格式数据集 (queries/corpus/qrels/top_ranked)

        Args:
            queries: 查询字典或 HF Dataset
            corpus: 文档字典或 HF Dataset
            qrels: 相关性判断 {qid: {doc_id: score}}
            top_ranked: 候选文档 {qid: [doc_ids]}
            max_samples: 最大样本数
            shuffle_seed: 随机种子
            ks: 评估的 k 值列表
            progress_callback: 进度回调函数
            progress_desc: 进度条描述
            show_progress: 是否显示进度条
        """
        # 将 queries 和 corpus 转换为可索引格式
        if hasattr(queries, '__iter__') and not isinstance(queries, dict):
            # HF Dataset 格式，转为 {id: {id, text}} 字典
            queries_dict = {}
            for item in queries:
                if isinstance(item, dict) and 'id' in item:
                    queries_dict[item['id']] = item
            queries = queries_dict

        if hasattr(corpus, '__iter__') and not isinstance(corpus, dict):
            # HF Dataset 格式
            corpus_dict = {}
            for item in corpus:
                if isinstance(item, dict) and 'id' in item:
                    corpus_dict[item['id']] = item
            corpus = corpus_dict

        # 如果没有 top_ranked，从 qrels 构建
        if not top_ranked:
            top_ranked = {qid: list(rels.keys()) for qid, rels in qrels.items()}

        # 限制样本数
        query_ids = list(queries.keys()) if isinstance(queries, dict) else list(range(len(queries)))
        if max_samples and max_samples < len(query_ids):
            random.seed(shuffle_seed)
            query_ids = random.sample(query_ids, max_samples)

        num_queries = len(query_ids)
        logger.info(f"Evaluating {num_queries} queries (MTEB v2 format)")

        all_results = []

        # 创建进度条
        use_tqdm = show_progress and TQDM_AVAILABLE and num_queries > 1
        pbar = None
        if use_tqdm:
            import sys
            desc = progress_desc or "Evaluating"
            pbar = tqdm(total=num_queries, desc=desc, unit="query", file=sys.stdout, leave=False)

        # 逐条评估
        for idx, qid in enumerate(query_ids):
            # 获取查询文本
            if isinstance(queries, dict):
                query_item = queries.get(qid, {})
                query_text = query_item.get('text', '') if isinstance(query_item, dict) else str(query_item)
            else:
                query_text = str(queries[qid])

            # 检查 qrels 和 top_ranked
            if qid not in qrels or qid not in top_ranked:
                if pbar:
                    pbar.update(1)
                continue

            # 获取候选文档 ID（限制数量避免 OOM）
            candidate_doc_ids = top_ranked[qid][:100]

            # 构建文档列表和标签
            documents = []
            labels = []
            for doc_id in candidate_doc_ids:
                doc_text = None
                if isinstance(corpus, dict) and doc_id in corpus:
                    doc = corpus[doc_id]
                    if isinstance(doc, dict):
                        doc_text = doc.get('text', '') or doc.get('title', '')
                    else:
                        doc_text = str(doc)

                if doc_text:
                    documents.append(doc_text)
                    labels.append(qrels[qid].get(doc_id, 0))

            if len(documents) < 2 or sum(labels) == 0:
                if pbar:
                    pbar.update(1)
                continue

            # 调用 rerank
            try:
                ranking, scores = self._call_rerank(query_text, documents)
                positive_indices = {i for i, lab in enumerate(labels) if lab > 0}
                all_results.append({
                    "ranking": ranking,
                    "scores": scores,
                    "positive_indices": positive_indices,
                    "num_docs": len(documents),
                })
            except Exception as e:
                logger.warning(f"Rerank failed for query {qid}: {e}")

            if pbar:
                pbar.update(1)
            if progress_callback:
                progress_callback(idx + 1, num_queries)

        if pbar:
            pbar.close()

        return self._aggregate_results(all_results, ks)

    def _aggregate_results(
        self,
        all_results: List[Dict[str, Any]],
        ks: List[int],
    ) -> Dict[str, float]:
        """聚合所有样本的结果"""
        if not all_results:
            return {
                "num_queries": 0,
                "num_evaluated": 0,
                "MRR": 0.0,
                "AP": 0.0,
                **{f"NDCG@{k}": 0.0 for k in ks},
                **{f"P@{k}": 0.0 for k in ks},
                **{f"R@{k}": 0.0 for k in ks},
            }

        mrr_scores = []
        ap_scores = []
        ndcg_scores = {k: [] for k in ks}
        precision_scores = {k: [] for k in ks}
        recall_scores = {k: [] for k in ks}

        for result in all_results:
            ranking = result["ranking"]
            positive_indices = result["positive_indices"]

            mrr_scores.append(mrr(ranking, positive_indices))
            ap_scores.append(ap(ranking, positive_indices))

            for k in ks:
                ndcg_scores[k].append(ndcg_at_k_binary(ranking, positive_indices, k))
                precision_scores[k].append(precision_at_k(ranking, positive_indices, k))
                recall_scores[k].append(recall_at_k(ranking, positive_indices, k))

        results = {
            "num_queries": len(all_results),
            "num_evaluated": len(all_results),
            "MRR": np.mean(mrr_scores),
            "AP": np.mean(ap_scores),
        }

        for k in ks:
            results[f"NDCG@{k}"] = np.mean(ndcg_scores[k])
            results[f"P@{k}"] = np.mean(precision_scores[k])
            results[f"R@{k}"] = np.mean(recall_scores[k])

        return results

    def evaluate_local(
        self,
        dataset: Union[LocalDataset, List[Dict[str, Any]], str],
        max_samples: Optional[int] = None,
        shuffle_seed: int = 42,
        ks: List[int] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, float]:
        """
        评估本地数据集

        Args:
            dataset: LocalDataset, 数据列表, 或 jsonl 文件路径
            max_samples: 最大样本数
            shuffle_seed: 打乱文档的随机种子
            ks: 评估的 k 值列表
            progress_callback: 进度回调函数

        Returns:
            评估指标字典
        """
        if ks is None:
            ks = [1, 5, 10]

        # 处理不同的输入格式
        if isinstance(dataset, str):
            dataset = LocalDataset.from_jsonl(dataset)
        elif isinstance(dataset, list):
            dataset = LocalDataset(dataset)

        num_samples = len(dataset)
        if max_samples:
            num_samples = min(max_samples, num_samples)

        logger.info(f"Evaluating {num_samples} local samples")

        all_results = []
        for i in range(num_samples):
            sample = dataset[i]
            result = self._evaluate_local_sample(sample, i, shuffle_seed)
            if result is not None:
                all_results.append(result)

            if progress_callback:
                progress_callback(i + 1, num_samples)

        return self._aggregate_results(all_results, ks)

    def _evaluate_local_sample(
        self,
        sample: LocalSample,
        sample_idx: int,
        shuffle_seed: int,
    ) -> Optional[Dict[str, Any]]:
        """评估本地数据集的单个样本"""
        query = sample.query
        positive = sample.positive
        negative = sample.negative

        docs = positive + negative
        num_positive = len(positive)

        if len(docs) < 2:
            return None

        # 打乱文档顺序
        indices = list(range(len(docs)))
        random.seed(shuffle_seed + sample_idx)
        random.shuffle(indices)
        shuffled_docs = [docs[idx] for idx in indices]

        positive_indices = set(indices.index(j) for j in range(num_positive))

        try:
            ranking, scores = self._call_rerank(query, shuffled_docs)
            return {
                "ranking": ranking,
                "scores": scores,
                "positive_indices": positive_indices,
                "num_docs": len(docs),
            }
        except Exception as e:
            logger.warning(f"Rerank failed for sample {sample_idx}: {e}")
            return None

    def evaluate_multiple(
        self,
        task_names: List[str],
        max_samples: Optional[int] = None,
        ks: List[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        评估多个 MTEB 数据集

        Args:
            task_names: 任务名称列表（支持组名如 "chinese", "english", "all"）
            max_samples: 每个数据集的最大样本数
            ks: 评估的 k 值列表

        Returns:
            {task_name: metrics_dict} 的字典
        """
        expanded_tasks = expand_dataset_names(task_names)
        all_results = {}

        for task_name in expanded_tasks:
            logger.info(f"\nEvaluating: {task_name}")
            results = self.evaluate(
                task_name=task_name,
                max_samples=max_samples,
                ks=ks,
            )
            all_results[task_name] = results

            # 打印简要结果
            if "error" not in results:
                ndcg10 = results.get("NDCG@10", 0)
                mrr_val = results.get("MRR", 0)
                logger.info(f"  NDCG@10: {ndcg10:.4f}, MRR: {mrr_val:.4f}")

        return all_results


# ============================================================================
# 便捷函数
# ============================================================================

def evaluate_reranking_dataset(
    rerank_fn: Callable[[str, List[str]], Tuple[List[int], Dict[int, float]]],
    task_name: str,
    split: Optional[str] = None,
    max_samples: Optional[int] = None,
    ks: List[int] = None,
    proxy: Optional[str] = None,
    batch_size: int = 50,
    workers: int = 8,
) -> Dict[str, float]:
    """
    评估单个 MTEB reranking 数据集

    Args:
        rerank_fn: rerank 函数，接受 (query, documents) 返回 (ranking, scores)
        task_name: MTEB 任务名称
        split: 数据集划分
        max_samples: 最大样本数
        ks: 评估的 k 值列表
        proxy: 代理地址
        batch_size: 每次请求的最大文档数（避免 OOM）
        workers: 并发请求数

    Returns:
        评估指标字典
    """
    if proxy:
        set_proxy(proxy)

    evaluator = MTEBRerankEvaluator(
        rerank_fn=rerank_fn,
        batch_size=batch_size,
        workers=workers,
    )
    return evaluator.evaluate(
        task_name=task_name,
        split=split,
        max_samples=max_samples,
        ks=ks,
    )


def evaluate_local_dataset(
    rerank_fn: Callable[[str, List[str]], Tuple[List[int], Dict[int, float]]],
    dataset: Union[LocalDataset, List[Dict[str, Any]], str],
    max_samples: Optional[int] = None,
    ks: List[int] = None,
    batch_size: int = 50,
    workers: int = 8,
) -> Dict[str, float]:
    """
    评估本地数据集

    Args:
        rerank_fn: rerank 函数
        dataset: 本地数据集（LocalDataset、数据列表或 jsonl 路径）
        max_samples: 最大样本数
        ks: 评估的 k 值列表
        batch_size: 每次请求的最大文档数
        workers: 并发请求数

    Returns:
        评估指标字典
    """
    evaluator = MTEBRerankEvaluator(
        rerank_fn=rerank_fn,
        batch_size=batch_size,
        workers=workers,
    )
    return evaluator.evaluate_local(
        dataset=dataset,
        max_samples=max_samples,
        ks=ks,
    )


# ============================================================================
# 多模型并行评估
# ============================================================================

def _evaluate_single_model(
    model_name: str,
    evaluator: MTEBRerankEvaluator,
    task_names: List[str],
    max_samples: Optional[int],
    ks: List[int],
    semaphore: Optional[threading.Semaphore],
) -> Dict[str, Any]:
    """评估单个模型（用于多模型并发）"""
    if semaphore:
        semaphore.acquire()

    try:
        logger.info(f"开始评估模型: {model_name}")
        results = evaluator.evaluate_multiple(
            task_names=task_names,
            max_samples=max_samples,
            ks=ks,
        )
        logger.info(f"模型 {model_name} 评估完成")
        return {
            "model": model_name,
            "results": results,
        }
    finally:
        if semaphore:
            semaphore.release()


def evaluate_multiple_models(
    rerankers: Dict[str, Any],
    task_names: List[str],
    max_samples: Optional[int] = None,
    ks: List[int] = None,
    model_workers: int = 2,
    batch_size: int = 50,
    workers: int = 8,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    多模型并行评估

    Args:
        rerankers: 模型名称到 Reranker 实例的映射
            - 可以是 Reranker 实例（需实现 rerank 方法）
            - 或者是 rerank 函数
        task_names: 任务名称列表（支持组名）
        max_samples: 每个数据集的最大样本数
        ks: 评估的 k 值列表
        model_workers: 多模型并行评估数（控制同时评估的模型数量）
        batch_size: 每次请求的最大文档数（避免 OOM）
        workers: 单模型的 API 请求并发数

    Returns:
        {model_name: {task_name: metrics_dict}} 的嵌套字典

    Example:
        >>> results = evaluate_multiple_models(
        ...     rerankers={
        ...         "model_a": reranker_a,
        ...         "model_b": reranker_b,
        ...     },
        ...     task_names=["chinese"],  # 评估所有中文数据集
        ...     model_workers=2,  # 2 个模型同时评估
        ...     batch_size=50,    # 每次请求最多 50 个文档
        ... )
        >>> print(results["model_a"]["T2Reranking"]["NDCG@10"])
    """
    if ks is None:
        ks = [1, 5, 10]

    # 展开任务名称
    expanded_tasks = expand_dataset_names(task_names)

    # 创建评估器
    evaluators = {}
    for name, reranker in rerankers.items():
        if callable(reranker) and not hasattr(reranker, 'rerank'):
            # 是函数
            evaluators[name] = MTEBRerankEvaluator(
                rerank_fn=reranker,
                batch_size=batch_size,
                workers=workers,
            )
        else:
            # 是 Reranker 实例
            evaluators[name] = MTEBRerankEvaluator(
                reranker=reranker,
                batch_size=batch_size,
                workers=workers,
            )

    # 全局并发控制
    semaphore = threading.Semaphore(model_workers) if model_workers < len(rerankers) else None

    logger.info(f"开始多模型评估: {len(rerankers)} 个模型, {len(expanded_tasks)} 个任务")
    logger.info(f"  模型并发数: {model_workers}")
    logger.info(f"  请求并发数: {workers}")
    logger.info(f"  批大小: {batch_size}")

    all_results = {}

    # 多模型并发评估
    with ThreadPoolExecutor(max_workers=len(rerankers)) as executor:
        futures = {
            executor.submit(
                _evaluate_single_model,
                name,
                evaluator,
                expanded_tasks,
                max_samples,
                ks,
                semaphore,
            ): name
            for name, evaluator in evaluators.items()
        }

        for future in as_completed(futures):
            model_name = futures[future]
            try:
                result = future.result()
                all_results[model_name] = result["results"]
            except Exception as e:
                logger.error(f"模型 {model_name} 评估失败: {e}")
                all_results[model_name] = {"error": str(e)}

    return all_results


def print_comparison_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str] = None,
) -> None:
    """
    打印多模型对比表格

    Args:
        results: evaluate_multiple_models 返回的结果
        metrics: 要显示的指标列表，默认 ["NDCG@10", "MRR", "AP"]
    """
    if metrics is None:
        metrics = ["NDCG@10", "MRR", "AP"]

    # 收集所有任务
    all_tasks = set()
    for model_results in results.values():
        if isinstance(model_results, dict) and "error" not in model_results:
            all_tasks.update(model_results.keys())
    all_tasks = sorted(all_tasks)

    if not all_tasks:
        print("没有评估结果")
        return

    # 打印表格
    print("\n" + "=" * 80)
    print("多模型评估对比")
    print("=" * 80)

    # 表头
    header = f"{'任务':<20} {'模型':<15}"
    for m in metrics:
        header += f" {m:<12}"
    print(header)
    print("-" * 80)

    for task in all_tasks:
        for model_name, model_results in sorted(results.items()):
            if isinstance(model_results, dict) and task in model_results:
                task_results = model_results[task]
                if "error" in task_results:
                    print(f"{task:<20} {model_name:<15} ERROR")
                    continue

                row = f"{task:<20} {model_name:<15}"
                for m in metrics:
                    val = task_results.get(m, 0)
                    row += f" {val:<12.4f}"
                print(row)

    print("=" * 80)
