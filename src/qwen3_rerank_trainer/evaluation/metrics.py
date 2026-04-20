"""
统一的排序评估指标计算模块

标准定义：
- NDCG@k: 只看前 k 个位置的排序质量（需要 @k）
- MRR: 第一个相关文档的倒数排名（不需要 @k，全局计算）
- MAP/AP: 所有相关文档位置的平均精度（不需要 @k，分母是总相关数）
- P@k: 前 k 个位置的精确率（需要 @k）
- R@k: 前 k 个位置的召回率（需要 @k）
"""

from typing import List, Dict, Set, Optional
import math
import numpy as np


def mrr(ranking: List[int], positive_indices: Set[int]) -> float:
    """
    计算 MRR (Mean Reciprocal Rank)

    标准定义：第一个相关文档的倒数排名，全局计算，不截断。

    Args:
        ranking: 排序后的文档索引列表（按分数从高到低）
        positive_indices: 相关文档的索引集合

    Returns:
        MRR 值 (0-1)，如果没有相关文档返回 0
    """
    if not positive_indices:
        return 0.0

    for rank, doc_idx in enumerate(ranking, 1):
        if doc_idx in positive_indices:
            return 1.0 / rank
    return 0.0


def ap(ranking: List[int], positive_indices: Set[int]) -> float:
    """
    计算 AP (Average Precision)

    标准定义：遍历所有位置，在每个相关文档位置计算 P@k，然后平均。

    Args:
        ranking: 排序后的文档索引列表（按分数从高到低）
        positive_indices: 相关文档的索引集合

    Returns:
        AP 值 (0-1)
    """
    if not positive_indices:
        return 0.0

    num_positive = len(positive_indices)
    num_hits = 0
    precision_sum = 0.0

    for rank, doc_idx in enumerate(ranking, 1):
        if doc_idx in positive_indices:
            num_hits += 1
            precision_sum += num_hits / rank

    return precision_sum / num_positive


def ndcg_at_k(
    ranking: List[int],
    relevance_scores: Dict[int, float],
    k: int,
) -> float:
    """
    计算 NDCG@k (Normalized Discounted Cumulative Gain at k)

    Args:
        ranking: 排序后的文档索引列表
        relevance_scores: 文档索引到相关性分数的映射（支持 graded relevance）
        k: 截断位置

    Returns:
        NDCG@k 值 (0-1)
    """
    if not relevance_scores or k <= 0:
        return 0.0

    dcg = 0.0
    for i, doc_idx in enumerate(ranking[:k], 1):
        rel = relevance_scores.get(doc_idx, 0.0)
        dcg += (2 ** rel - 1) / np.log2(i + 1)

    sorted_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(sorted_rels, 1):
        idcg += (2 ** rel - 1) / np.log2(i + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def ndcg_at_k_binary(
    ranking: List[int],
    positive_indices: Set[int],
    k: int,
) -> float:
    """二元相关性的 NDCG@k"""
    relevance_scores = {idx: 1.0 for idx in positive_indices}
    return ndcg_at_k(ranking, relevance_scores, k)


def precision_at_k(
    ranking: List[int],
    positive_indices: Set[int],
    k: int,
) -> float:
    """计算 P@k (Precision at k)"""
    if k <= 0:
        return 0.0

    top_k = ranking[:k]
    num_hits = sum(1 for idx in top_k if idx in positive_indices)
    return num_hits / k


def recall_at_k(
    ranking: List[int],
    positive_indices: Set[int],
    k: int,
) -> float:
    """计算 R@k (Recall at k)"""
    if not positive_indices or k <= 0:
        return 0.0

    top_k = ranking[:k]
    num_hits = sum(1 for idx in top_k if idx in positive_indices)
    return num_hits / len(positive_indices)


def hit_at_k(
    ranking: List[int],
    positive_indices: Set[int],
    k: int,
) -> float:
    """计算 Hit@k (也称 Success@k)"""
    if k <= 0:
        return 0.0

    top_k = ranking[:k]
    return 1.0 if any(idx in positive_indices for idx in top_k) else 0.0


success_at_k = hit_at_k


def f1_at_k(
    ranking: List[int],
    positive_indices: Set[int],
    k: int,
) -> float:
    """计算 F1@k"""
    p = precision_at_k(ranking, positive_indices, k)
    r = recall_at_k(ranking, positive_indices, k)

    if p + r == 0:
        return 0.0

    return 2 * p * r / (p + r)


# ============================================================================
# 批量计算函数
# ============================================================================

def compute_all_metrics(
    ranking: List[int],
    positive_indices: Set[int],
    ks: List[int] = None,
    relevance_scores: Dict[int, float] = None,
) -> Dict[str, float]:
    """一次性计算所有评估指标"""
    if ks is None:
        ks = [1, 5, 10]

    results = {
        'MRR': mrr(ranking, positive_indices),
        'AP': ap(ranking, positive_indices),
    }

    for k in ks:
        if relevance_scores:
            results[f'NDCG@{k}'] = ndcg_at_k(ranking, relevance_scores, k)
        else:
            results[f'NDCG@{k}'] = ndcg_at_k_binary(ranking, positive_indices, k)

        results[f'P@{k}'] = precision_at_k(ranking, positive_indices, k)
        results[f'R@{k}'] = recall_at_k(ranking, positive_indices, k)
        results[f'F1@{k}'] = f1_at_k(ranking, positive_indices, k)
        results[f'Hit@{k}'] = hit_at_k(ranking, positive_indices, k)

    return results


def aggregate_metrics(
    all_results: List[Dict[str, float]],
) -> Dict[str, float]:
    """聚合多个样本的指标（计算平均值）"""
    if not all_results:
        return {}

    aggregated = {}
    keys = all_results[0].keys()

    for key in keys:
        values = [r[key] for r in all_results if key in r]
        if values:
            aggregated[key] = np.mean(values)
            aggregated[f'{key}_std'] = np.std(values)

    return aggregated


# ============================================================================
# 兼容性函数
# ============================================================================

def calculate_mrr(ranking: List[int], labels: List[int]) -> float:
    """兼容旧接口的 MRR 计算"""
    positive_indices = {i for i, label in enumerate(labels) if label == 1}
    return mrr(ranking, positive_indices)


def calculate_ndcg(ranking: List[int], labels: List[int], k: int = 10) -> float:
    """兼容旧接口的 NDCG@k 计算"""
    positive_indices = {i for i, label in enumerate(labels) if label == 1}
    return ndcg_at_k_binary(ranking, positive_indices, k)


def calculate_ap(ranking: List[int], labels: List[int]) -> float:
    """兼容旧接口的 AP 计算"""
    positive_indices = {i for i, label in enumerate(labels) if label == 1}
    return ap(ranking, positive_indices)


# ============================================================================
# 基于分数的接口
# ============================================================================

def mrr_from_scores(scores: List[float], labels: List[int]) -> float:
    """基于分数计算 MRR"""
    if not scores or sum(labels) == 0:
        return 0.0

    sorted_indices = list(np.argsort(scores)[::-1])
    positive_indices = {i for i, label in enumerate(labels) if label > 0}
    return mrr(sorted_indices, positive_indices)


def ap_from_scores(scores: List[float], labels: List[int]) -> float:
    """基于分数计算 AP"""
    if not scores or sum(labels) == 0:
        return 0.0

    sorted_indices = list(np.argsort(scores)[::-1])
    positive_indices = {i for i, label in enumerate(labels) if label > 0}
    return ap(sorted_indices, positive_indices)


def ndcg_from_scores(scores: List[float], labels: List[int], k: int = 10) -> float:
    """基于分数计算 NDCG@k"""
    if len(scores) < 2 or sum(labels) == 0:
        return 0.0

    sorted_indices = list(np.argsort(scores)[::-1])
    positive_indices = {i for i, label in enumerate(labels) if label > 0}
    return ndcg_at_k_binary(sorted_indices, positive_indices, min(k, len(scores)))


def precision_from_scores(scores: List[float], labels: List[int], k: int) -> float:
    """基于分数计算 Precision@k"""
    if not scores:
        return 0.0

    sorted_indices = list(np.argsort(scores)[::-1])
    positive_indices = {i for i, label in enumerate(labels) if label > 0}
    return precision_at_k(sorted_indices, positive_indices, k)


def recall_from_scores(scores: List[float], labels: List[int], k: int) -> float:
    """基于分数计算 Recall@k"""
    if not scores or sum(labels) == 0:
        return 0.0

    sorted_indices = list(np.argsort(scores)[::-1])
    positive_indices = {i for i, label in enumerate(labels) if label > 0}
    return recall_at_k(sorted_indices, positive_indices, k)


def hit_from_scores(scores: List[float], labels: List[int], k: int) -> float:
    """基于分数计算 Hit@k"""
    if not scores or sum(labels) == 0:
        return 0.0

    sorted_indices = list(np.argsort(scores)[::-1])
    positive_indices = {i for i, label in enumerate(labels) if label > 0}
    return hit_at_k(sorted_indices, positive_indices, k)


# ============================================================================
# 基于已排序标签的接口
# ============================================================================

def mrr_from_sorted_labels(labels: List[int]) -> float:
    """基于已排序标签计算 MRR"""
    if not labels:
        return 0.0

    for i, y in enumerate(labels):
        if int(y) == 1:
            return 1.0 / (i + 1)
    return 0.0


def ap_from_sorted_labels(labels: List[int]) -> float:
    """基于已排序标签计算 AP"""
    if not labels or sum(labels) == 0:
        return 0.0

    num_positive = sum(1 for y in labels if int(y) == 1)
    num_hits = 0
    precision_sum = 0.0

    for i, y in enumerate(labels, 1):
        if int(y) == 1:
            num_hits += 1
            precision_sum += num_hits / i

    return precision_sum / num_positive


def ndcg_from_sorted_labels(labels: List[int], k: int = 10) -> float:
    """基于已排序标签计算 NDCG@k"""
    if not labels:
        return 0.0

    dcg = 0.0
    for i, y in enumerate(labels[:k]):
        if int(y) == 1:
            dcg += 1.0 / math.log2(i + 2)

    num_positives = sum(1 for y in labels if int(y) == 1)
    idcg = 0.0
    for i in range(min(k, num_positives)):
        idcg += 1.0 / math.log2(i + 2)

    return dcg / idcg if idcg > 0 else 0.0


def precision_from_sorted_labels(labels: List[int], k: int) -> float:
    """基于已排序标签计算 Precision@k"""
    if not labels or k <= 0:
        return 0.0

    top_k = labels[:k]
    return sum(1 for y in top_k if int(y) == 1) / k


def recall_from_sorted_labels(labels: List[int], k: int) -> float:
    """基于已排序标签计算 Recall@k"""
    total_positives = sum(1 for y in labels if int(y) == 1)
    if not labels or k <= 0 or total_positives == 0:
        return 0.0

    top_k = labels[:k]
    return sum(1 for y in top_k if int(y) == 1) / total_positives


def hit_from_sorted_labels(labels: List[int], k: int) -> float:
    """基于已排序标签计算 Hit@k"""
    if not labels or k <= 0:
        return 0.0

    top_k = labels[:k]
    return 1.0 if any(int(y) == 1 for y in top_k) else 0.0
