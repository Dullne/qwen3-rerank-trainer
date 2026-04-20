"""
异步 Rerank API 客户端

提供高性能的异步批量 Rerank API 调用：
- 异步并发请求（aiohttp）
- 批量处理大文档列表（自动分块避免 OOM）
- 进度条显示（tqdm）
- 自动重试和错误处理

使用示例:
    from qwen3_rerank_trainer.evaluation import (
        APIReranker,
        call_rerank_batch_async,
    )

    # 方式1：使用 APIReranker 类
    reranker = APIReranker(
        endpoint="http://localhost:9997/v1/rerank",
        model="Qwen3-Reranker-4B",
    )
    ranking, scores = reranker.rerank(query, documents)

    # 方式2：批量异步调用
    items = [(query1, docs1), (query2, docs2), ...]
    results = await call_rerank_batch_async(
        items,
        endpoint="http://localhost:9997/v1/rerank",
        max_concurrency=10,
    )
"""

import asyncio
import logging
import random
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

# 可选依赖
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    aiohttp = None
    AIOHTTP_AVAILABLE = False

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False


# ============================================================================
# 结果解析
# ============================================================================

def _extract_rerank_results(
    data: Any,
    num_docs: int,
) -> Tuple[List[int], Dict[int, float]]:
    """
    从 Rerank API 响应中提取排序和分数

    支持的响应格式:
        - {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        - {"results": [{"index": 0, "score": 0.9}, ...]}
        - {"data": {"results": [...]}}
        - [{"index": 0, "score": 0.9}, ...]  (SGLang format)

    Returns:
        (ranking, scores): 排序后的索引列表和分数字典
    """
    # 提取 results 列表
    results = None
    if isinstance(data, list):
        # SGLang 直接返回数组格式
        results = data
    elif isinstance(data, dict):
        if "results" in data:
            results = data["results"]
        elif "data" in data and isinstance(data["data"], dict):
            results = data["data"].get("results", [])

    if not results:
        return list(range(num_docs)), {i: 0.0 for i in range(num_docs)}

    # 解析每个结果
    scores = {}
    for item in results:
        if isinstance(item, dict):
            idx = item.get("index", -1)
            # 支持 relevance_score 和 score 两种字段名
            score = item.get("relevance_score", item.get("score", 0.0))
            if 0 <= idx < num_docs:
                scores[idx] = float(score)

    # 生成排序（分数降序，相同分数时索引降序）
    ranking = sorted(scores.keys(), key=lambda x: (scores[x], x), reverse=True)

    # 补充缺失的索引
    for i in range(num_docs):
        if i not in scores:
            scores[i] = 0.0
            ranking.append(i)

    return ranking, scores


# ============================================================================
# 异步 API 调用
# ============================================================================

async def call_rerank_async(
    query: str,
    documents: List[str],
    endpoint: str,
    model: str = "Qwen3-Reranker-4B",
    timeout: int = 30,
    session: Optional["aiohttp.ClientSession"] = None,
) -> Tuple[List[int], Dict[int, float]]:
    """
    异步调用 Rerank API

    Args:
        query: 查询文本
        documents: 文档列表
        endpoint: API 端点 (如 "http://localhost:9997/v1/rerank")
        model: 模型名称
        timeout: 超时时间（秒）
        session: 可选，复用 aiohttp.ClientSession

    Returns:
        (ranking, scores): 排序后的索引列表和分数字典
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not installed, please run: pip install aiohttp")

    if not documents:
        return [], {}

    payload = {
        "model": model,
        "query": query,
        "documents": documents,
    }

    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True

    try:
        async with session.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return _extract_rerank_results(data, len(documents))
    except Exception as e:
        raise RuntimeError(f"Rerank API call failed: {e}") from e
    finally:
        if close_session:
            await session.close()


async def call_rerank_async_safe(
    query: str,
    documents: List[str],
    endpoint: str,
    model: str = "Qwen3-Reranker-4B",
    timeout: int = 30,
    session: Optional["aiohttp.ClientSession"] = None,
    retries: int = 2,
    backoff: float = 0.5,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
    retry_statuses: Tuple[int, ...] = (429, 500, 502, 503, 504),
) -> Tuple[List[int], Dict[int, float]]:
    """
    异步调用 Rerank API（安全版本，失败时返回空结果）

    支持可选重试与退避策略。
    """
    def _is_retryable_error(exc: Exception) -> bool:
        if not AIOHTTP_AVAILABLE:
            return False
        if isinstance(exc, asyncio.TimeoutError):
            return True
        if isinstance(exc, aiohttp.ClientResponseError):
            return exc.status in retry_statuses
        if isinstance(exc, (aiohttp.ClientConnectionError, aiohttp.ServerTimeoutError)):
            return True
        return False

    last_error: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            return await call_rerank_async(query, documents, endpoint, model, timeout, session)
        except Exception as e:
            last_error = e
            if attempt < retries and _is_retryable_error(e):
                sleep_s = backoff * (backoff_factor ** attempt)
                if jitter > 0:
                    sleep_s *= 1.0 + random.random() * jitter
                await asyncio.sleep(sleep_s)
                continue
            logger.warning(f"Rerank API call failed: {e}")
            break
    return [], {}


async def call_rerank_batch_async(
    items: List[Tuple[str, List[str]]],
    endpoint: str,
    model: str = "Qwen3-Reranker-4B",
    timeout: int = 30,
    max_concurrency: int = 10,
    show_progress: bool = True,
    progress_desc: str = None,
    retries: int = 2,
    backoff: float = 0.5,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
) -> List[Tuple[List[int], Dict[int, float]]]:
    """
    批量异步调用 Rerank API

    Args:
        items: [(query, documents), ...] 列表
        endpoint: API 端点
        model: 模型名称
        timeout: 单次请求超时时间
        max_concurrency: 最大并发数
        show_progress: 是否显示进度条
        progress_desc: 进度条描述
        retries: 重试次数（仅对可重试错误生效）
        backoff: 重试初始等待秒数
        backoff_factor: 退避倍率
        jitter: 随机抖动比例

    Returns:
        [(ranking, scores), ...] 结果列表，与输入顺序对应
    """
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp not installed, please run: pip install aiohttp")

    if not items:
        return []

    results: List[Optional[Tuple[List[int], Dict[int, float]]]] = [None] * len(items)
    total = len(items)
    worker_count = min(max_concurrency, total)

    queue: asyncio.Queue[Tuple[int, str, List[str]]] = asyncio.Queue()
    for i, (query, docs) in enumerate(items):
        queue.put_nowait((i, query, docs))

    pbar = None
    progress_lock = asyncio.Lock()
    if show_progress and TQDM_AVAILABLE and total > 1:
        import sys
        desc = progress_desc or "Rerank"
        pbar = tqdm(total=total, desc=desc, unit="req", file=sys.stdout, leave=False)

    async def _worker(session: "aiohttp.ClientSession"):
        while True:
            try:
                idx, query, docs = queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            result = await call_rerank_async_safe(
                query,
                docs,
                endpoint,
                model,
                timeout,
                session,
                retries=retries,
                backoff=backoff,
                backoff_factor=backoff_factor,
                jitter=jitter,
            )
            results[idx] = result
            if pbar is not None:
                async with progress_lock:
                    pbar.update(1)
            queue.task_done()

    async with aiohttp.ClientSession() as session:
        tasks = [asyncio.create_task(_worker(session)) for _ in range(worker_count)]
        await asyncio.gather(*tasks)

    if pbar is not None:
        pbar.close()

    return [item if item is not None else ([], {}) for item in results]


# ============================================================================
# 同步包装器
# ============================================================================

def call_rerank(
    query: str,
    documents: List[str],
    endpoint: str,
    model: str = "Qwen3-Reranker-4B",
    timeout: int = 30,
) -> Tuple[List[int], Dict[int, float]]:
    """
    同步调用 Rerank API

    Args:
        query: 查询文本
        documents: 文档列表
        endpoint: API 端点
        model: 模型名称
        timeout: 超时时间

    Returns:
        (ranking, scores)
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            call_rerank_async(query, documents, endpoint, model, timeout)
        )
    finally:
        loop.close()


def call_rerank_batch(
    items: List[Tuple[str, List[str]]],
    endpoint: str,
    model: str = "Qwen3-Reranker-4B",
    timeout: int = 30,
    max_concurrency: int = 10,
    show_progress: bool = True,
    progress_desc: str = None,
    retries: int = 2,
    backoff: float = 0.5,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
) -> List[Tuple[List[int], Dict[int, float]]]:
    """
    同步批量调用 Rerank API

    内部使用异步实现，对外提供同步接口。
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            call_rerank_batch_async(
                items, endpoint, model, timeout,
                max_concurrency, show_progress, progress_desc,
                retries, backoff, backoff_factor, jitter
            )
        )
    finally:
        loop.close()


# ============================================================================
# APIReranker 类
# ============================================================================

class APIReranker:
    """
    基于 API 的 Reranker

    提供与本地 Reranker 兼容的接口，方便切换。

    Args:
        endpoint: API 端点 (如 "http://localhost:9997/v1/rerank")
        model: 模型名称
        timeout: 单次请求超时时间
        batch_size: 批量处理时每次请求的最大文档数（避免 OOM）
        max_concurrency: 最大并发请求数

    Example:
        >>> reranker = APIReranker("http://localhost:9997/v1/rerank")
        >>> ranking, scores = reranker.rerank(query, documents)

        # 批量评估
        >>> evaluator = MTEBRerankEvaluator(reranker=reranker)
        >>> results = evaluator.evaluate("T2Reranking")
    """

    # vLLM Qwen3-Reranker prompt template
    VLLM_PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
    VLLM_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
    VLLM_DEFAULT_INSTRUCTION = 'Given a web search query, retrieve relevant passages that answer the query'

    def __init__(
        self,
        endpoint: str,
        model: str = "Qwen3-Reranker-4B",
        timeout: int = 30,
        batch_size: int = 100,
        max_concurrency: int = 10,
        inference_framework: str = "",
        instruction: str = "",
    ):
        """
        Args:
            endpoint: API 端点
            model: 模型名称
            timeout: 单次请求超时时间
            batch_size: 批量处理时每次请求的最大文档数
            max_concurrency: 最大并发请求数
            inference_framework: 推理框架 (vllm/sglang/xinference)，vLLM 需要预格式化
            instruction: 自定义 instruction（仅 vLLM 使用）
        """
        # 自动补全 /v1/rerank 路径
        endpoint = endpoint.rstrip('/')
        if not endpoint.endswith('/rerank'):
            if not endpoint.endswith('/v1'):
                endpoint += '/v1'
            endpoint += '/rerank'

        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.inference_framework = inference_framework.lower()
        self.instruction = instruction or self.VLLM_DEFAULT_INSTRUCTION

    def _format_vllm_request(
        self, query: str, documents: List[str]
    ) -> Tuple[str, List[str]]:
        """Format query and documents for vLLM Qwen3-Reranker."""
        formatted_query = f'{self.VLLM_PREFIX}<Instruct>: {self.instruction}\n<Query>: {query}\n'
        formatted_docs = [f'<Document>: {doc}{self.VLLM_SUFFIX}' for doc in documents]
        return formatted_query, formatted_docs

    def rerank(
        self,
        query: str,
        documents: List[str],
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        对文档进行重排序

        Args:
            query: 查询文本
            documents: 文档列表

        Returns:
            (ranking, scores): 排序后的索引列表和分数字典
        """
        # vLLM 需要预格式化
        if self.inference_framework == 'vllm':
            query, documents = self._format_vllm_request(query, documents)

        if len(documents) <= self.batch_size:
            return call_rerank(query, documents, self.endpoint, self.model, self.timeout)

        # 分批处理大文档列表
        all_scores = {}
        for start in range(0, len(documents), self.batch_size):
            end = min(start + self.batch_size, len(documents))
            batch_docs = documents[start:end]
            _, batch_scores = call_rerank(query, batch_docs, self.endpoint, self.model, self.timeout)

            # 合并分数（调整索引）
            for local_idx, score in batch_scores.items():
                all_scores[start + local_idx] = score

        # 生成最终排序
        ranking = sorted(all_scores.keys(), key=lambda x: (all_scores[x], x), reverse=True)
        return ranking, all_scores

    def rerank_batch(
        self,
        items: List[Tuple[str, List[str]]],
        show_progress: bool = True,
        progress_desc: str = None,
    ) -> List[Tuple[List[int], Dict[int, float]]]:
        """
        批量重排序

        Args:
            items: [(query, documents), ...] 列表
            show_progress: 是否显示进度条
            progress_desc: 进度条描述

        Returns:
            [(ranking, scores), ...] 结果列表
        """
        # vLLM 需要预格式化
        if self.inference_framework == 'vllm':
            items = [self._format_vllm_request(q, docs) for q, docs in items]

        # 展开大文档列表
        expanded_items = []
        item_mapping = []  # (原始索引, 偏移量)

        for i, (query, docs) in enumerate(items):
            if len(docs) <= self.batch_size:
                expanded_items.append((query, docs))
                item_mapping.append((i, 0))
            else:
                for start in range(0, len(docs), self.batch_size):
                    end = min(start + self.batch_size, len(docs))
                    expanded_items.append((query, docs[start:end]))
                    item_mapping.append((i, start))

        # 批量调用
        raw_results = call_rerank_batch(
            expanded_items,
            self.endpoint,
            self.model,
            self.timeout,
            self.max_concurrency,
            show_progress,
            progress_desc,
        )

        # 合并分批结果
        merged_scores = [{} for _ in range(len(items))]
        for batch_idx, (_, scores_map) in enumerate(raw_results):
            orig_idx, offset = item_mapping[batch_idx]
            for local_idx, score in scores_map.items():
                merged_scores[orig_idx][offset + local_idx] = score

        # 生成最终结果
        results = []
        for scores_map in merged_scores:
            if scores_map:
                ranking = sorted(scores_map.keys(), key=lambda x: (scores_map[x], x), reverse=True)
            else:
                ranking = []
            results.append((ranking, scores_map))

        return results

    def test_connection(self) -> bool:
        """
        测试 API 连接

        Returns:
            True 如果连接成功
        """
        test_query = "test query"
        test_docs = ["document 1", "document 2"]

        try:
            ranking, scores = self.rerank(test_query, test_docs)
            if ranking and len(ranking) == 2:
                logger.info(f"API connection test passed: {self.endpoint}")
                return True
            else:
                logger.warning(f"API returned unexpected result: {ranking}")
                return False
        except Exception as e:
            logger.error(f"API connection test failed: {e}")
            return False


# ============================================================================
# 便捷函数
# ============================================================================

def create_api_reranker(
    endpoint: str,
    model: str = "Qwen3-Reranker-4B",
    **kwargs,
) -> APIReranker:
    """
    创建 API Reranker

    Args:
        endpoint: API 端点（支持简写，如 "localhost:9997"）
        model: 模型名称
        **kwargs: 其他参数传递给 APIReranker

    Returns:
        APIReranker 实例
    """
    # 自动补全 http://
    if not endpoint.startswith("http"):
        endpoint = f"http://{endpoint}"

    return APIReranker(endpoint, model, **kwargs)
