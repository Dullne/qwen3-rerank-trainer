"""
文档采样模块

提供用于 RL 训练的文档采样函数。
"""

from __future__ import annotations

import random
from typing import List, Dict, Optional, Tuple, Union


def sample_documents(
    docs: List[Dict[str, Union[str, int, float]]],
    n_total: int = 6,
    n_pos: int = 2,
    neg_distribution: Optional[Dict[str, int]] = None,
    shuffle: bool = True,
    seed: Optional[int] = None,
    dynamic_ratio: bool = False,
    min_pos: int = 1,
    min_neg: int = 1,
    allow_repeat: bool = False,
) -> Tuple[List[str], List[int]]:
    """从候选文档中选择 N 个代表性文档用于 RL 训练

    类似 ERANK 的 N×G 设置，选择适量文档进行组内归一化。

    Args:
        docs: 候选文档列表，每个文档是 dict，包含：
            - "text": str, 文档文本
            - "label": int, 标签（1=正例, 0=负例）
            - "difficulty": str, 可选，负例难度
            - "score": float, 可选，检索分数
        n_total: 总共选择的文档数量
        n_pos: 最多选择的正例数量
        neg_distribution: 负例难度分布
        shuffle: 是否打乱文档顺序
        seed: 随机种子
        dynamic_ratio: 是否按样本自身比例动态分配
        min_pos: 动态分配时最少正例数
        min_neg: 动态分配时最少负例数
        allow_repeat: 是否允许重复采样

    Returns:
        selected_docs: List[str], 选中的文档文本
        selected_labels: List[int], 对应的标签
    """
    if seed is not None:
        random.seed(seed)

    positives = [d for d in docs if d["label"] == 1]
    negatives = [d for d in docs if d["label"] == 0]

    if dynamic_ratio:
        total_available = len(positives) + len(negatives)
        if total_available == 0:
            return [], []
        pos_ratio = len(positives) / total_available
        n_pos_target = max(min_pos, round(n_total * pos_ratio))
        n_neg_target = n_total - n_pos_target
        if n_neg_target < min_neg:
            n_neg_target = min_neg
            n_pos_target = n_total - n_neg_target
    else:
        n_pos_target = n_pos
        n_neg_target = n_total - n_pos_target

    selected = []

    n_pos_actual = min(n_pos_target, len(positives))
    if n_pos_actual > 0:
        selected_pos = random.sample(positives, n_pos_actual) if len(positives) > n_pos_actual else positives.copy()
        selected.extend(selected_pos)

    n_neg_needed = min(n_neg_target, n_total - len(selected))

    if n_neg_needed > 0 and negatives:
        if neg_distribution:
            selected_neg = _sample_by_difficulty(negatives, neg_distribution, n_neg_needed)
        else:
            n_neg_actual = min(n_neg_needed, len(negatives))
            selected_neg = random.sample(negatives, n_neg_actual) if len(negatives) > n_neg_actual else negatives.copy()
        selected.extend(selected_neg)

    if allow_repeat:
        while len(selected) < n_total and docs:
            selected.append(random.choice(docs))

    if shuffle:
        random.shuffle(selected)

    selected_docs = [d["text"] for d in selected]
    selected_labels = [d["label"] for d in selected]

    return selected_docs, selected_labels


def _sample_by_difficulty(
    negatives: List[Dict],
    distribution: Dict[str, int],
    n_needed: int,
) -> List[Dict]:
    """按难度分布采样负例"""
    by_difficulty = {}
    for neg in negatives:
        diff = neg.get("difficulty", "unknown")
        if diff not in by_difficulty:
            by_difficulty[diff] = []
        by_difficulty[diff].append(neg)

    selected = []

    for difficulty, count in distribution.items():
        if difficulty in by_difficulty and count > 0:
            pool = by_difficulty[difficulty]
            n_sample = min(count, len(pool), n_needed - len(selected))
            if n_sample > 0:
                selected.extend(random.sample(pool, n_sample))

        if len(selected) >= n_needed:
            break

    if len(selected) < n_needed:
        remaining = [n for n in negatives if n not in selected]
        n_extra = min(n_needed - len(selected), len(remaining))
        if n_extra > 0:
            selected.extend(random.sample(remaining, n_extra))

    return selected[:n_needed]


def sample_documents_by_score(
    docs: List[Dict[str, Union[str, int, float]]],
    n_total: int = 6,
    n_pos: int = 2,
    hard_ratio: float = 0.6,
    shuffle: bool = True,
    seed: Optional[int] = None,
    dynamic_ratio: bool = False,
    min_pos: int = 1,
    min_neg: int = 1,
) -> Tuple[List[str], List[int]]:
    """根据检索分数选择文档（分数越高的负例越难）

    Args:
        docs: 候选文档列表，每个文档需要有 "score" 字段
        n_total: 总共选择的文档数量
        n_pos: 最多选择的正例数量
        hard_ratio: hard negative 的比例
        shuffle: 是否打乱顺序
        seed: 随机种子
        dynamic_ratio: 是否按样本自身比例动态分配
        min_pos: 动态分配时最少正例数
        min_neg: 动态分配时最少负例数

    Returns:
        selected_docs: List[str], 选中的文档文本
        selected_labels: List[int], 对应的标签
    """
    if seed is not None:
        random.seed(seed)

    positives = [d for d in docs if d["label"] == 1]
    negatives = [d for d in docs if d["label"] == 0]

    if dynamic_ratio:
        total_available = len(positives) + len(negatives)
        if total_available == 0:
            return [], []
        pos_ratio = len(positives) / total_available
        n_pos_target = max(min_pos, round(n_total * pos_ratio))
        n_neg_target = n_total - n_pos_target
        if n_neg_target < min_neg:
            n_neg_target = min_neg
            n_pos_target = n_total - n_neg_target
    else:
        n_pos_target = n_pos
        n_neg_target = n_total - n_pos_target

    selected = []

    n_pos_actual = min(n_pos_target, len(positives))
    if n_pos_actual > 0:
        selected_pos = random.sample(positives, n_pos_actual) if len(positives) > n_pos_actual else positives.copy()
        selected.extend(selected_pos)

    n_neg_needed = min(n_neg_target, n_total - len(selected), len(negatives))
    if n_neg_needed > 0 and negatives:
        negatives_sorted = sorted(negatives, key=lambda x: x.get("score", 0), reverse=True)

        n_hard = int(n_neg_needed * hard_ratio)
        n_easy = n_neg_needed - n_hard

        mid = len(negatives_sorted) // 2
        hard_pool = negatives_sorted[:mid] if mid > 0 else negatives_sorted
        easy_pool = negatives_sorted[mid:] if mid > 0 else []

        selected_hard = random.sample(hard_pool, min(n_hard, len(hard_pool))) if hard_pool else []
        selected_easy = random.sample(easy_pool, min(n_easy, len(easy_pool))) if easy_pool else []

        selected.extend(selected_hard)
        selected.extend(selected_easy)

    if shuffle:
        random.shuffle(selected)

    selected_docs = [d["text"] for d in selected]
    selected_labels = [d["label"] for d in selected]

    return selected_docs, selected_labels
