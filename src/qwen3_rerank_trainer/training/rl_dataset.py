"""
RL Training Dataset and Collator

提供 RL 训练所需的数据集和批次构造器：
- RLRerankDataset: 支持 max_docs 限制的 RL 数据集
- RLCollator: RL 批次构造器（返回 group_sizes）
"""
import torch
import random
import logging
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .dataset import load_data, iter_data
from ..data.formatting import PREFIX, SUFFIX, format_input

logger = logging.getLogger(__name__)


class RLRerankDataset(Dataset):
    """RL 训练数据集

    每个样本返回一组文档用于组内归一化。
    支持多种数据格式：
    - 简单格式: {query, positives, negatives}
    - 分级格式: {query, pos, neg_very_hard, neg_hard, neg_medium}

    正负例采样策略：
    - n_docs = 0: 使用所有文档（不限制数量，仅受 min_pos/min_neg 约束）
    - n_pos > 0: 固定正例数（n_pos 正，n_docs - n_pos 负）
    - n_pos = 0: 动态分配（按原始比例，min_pos/min_neg 约束）

    Args:
        data_file: JSONL 数据文件路径
        tokenizer: 可选 tokenizer（仅在 filter_overlength=True 时用于长度过滤）
        max_length: 最大 token 长度，超过的文档会被过滤
        n_docs: 每组文档数（0 表示使用所有文档）
        n_pos: 固定正例数（0 表示按原始比例动态分配）
        min_pos: 最少正例数（仅动态分配时生效）
        min_neg: 最少负例数（仅动态分配时生效）
        max_docs: 单样本最大文档数（0 表示不限制，用于避免极端样本 OOM）
        seed: 随机种子
        max_samples: 最大样本数（0 表示不限制）
        format_fn: 自定义格式化函数
        filter_overlength: 是否过滤超过 max_length 的样本（默认关闭）

    Returns:
        每个样本返回:
        - query: str
        - documents: List[str] - 打乱后的文档列表
        - labels: List[int] - 对应的标签（1=正例, 0=负例）
    """

    def __init__(
        self,
        data_file: str,
        tokenizer=None,
        max_length: int = 4096,
        n_docs: int = 6,
        n_pos: int = 0,
        min_pos: int = 1,
        min_neg: int = 1,
        max_docs: int = 0,
        seed: int = 42,
        max_samples: int = 0,
        format_fn: Optional[callable] = None,
        filter_overlength: bool = False,
    ):
        self.n_docs = n_docs
        self.n_pos = n_pos
        self.min_pos = min_pos
        self.min_neg = min_neg
        self.max_docs = max_docs
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.format_fn = format_fn or self._default_format
        self.filter_overlength = filter_overlength
        self.seed = seed
        self.rng = random.Random(seed)
        self._worker_rng = None
        self._worker_id = None

        # 加载数据
        raw_data = load_data(data_file)
        logger.info(f"加载 {len(raw_data)} 条记录")

        # 检测数据格式
        sample = raw_data[0] if raw_data else {}
        is_simple_format = 'positives' in sample and 'negatives' in sample
        is_graded_format = 'neg_very_hard' in sample or 'neg_hard' in sample

        if is_simple_format:
            logger.info("数据格式: 简单格式 (positives/negatives)")
        elif is_graded_format:
            logger.info("数据格式: 分级格式 (neg_very_hard/hard/medium)")
        else:
            logger.info("数据格式: 旧格式 (pos/statement_*_negatives)")

        # 处理数据
        self.data = []
        for item in raw_data:
            query = item.get("query", "")
            if not query:
                continue

            # 提取正例
            positives = self._extract_positives(item, is_simple_format)
            # 提取负例
            negatives = self._extract_negatives(item, is_simple_format)

            # 过滤超长文档
            if self.tokenizer is not None and self.filter_overlength:
                positives = [p for p in positives if self._check_length(query, p)]
                negatives = [n for n in negatives if self._check_length(query, n)]

            # 限制单样本最大文档数（避免极端样本导致 OOM）
            if self.max_docs > 0:
                total_docs = len(positives) + len(negatives)
                if total_docs > self.max_docs:
                    # 按原始比例截断
                    pos_ratio = len(positives) / total_docs
                    max_pos = max(self.min_pos, int(self.max_docs * pos_ratio))
                    max_neg = self.max_docs - max_pos
                    positives = positives[:max_pos]
                    negatives = negatives[:max_neg]

            # 确保有足够的正例和负例
            if not self._has_enough_docs(positives, negatives):
                continue

            self.data.append({
                "query": query,
                "positives": positives,
                "negatives": negatives
            })

        logger.info(f"有效样本: {len(self.data)}")
        if self.tokenizer is not None and self.filter_overlength:
            logger.info(f"已过滤超过 {self.max_length} tokens 的文档")

        # 限制样本数量
        if max_samples > 0 and len(self.data) > max_samples:
            self.rng.shuffle(self.data)
            self.data = self.data[:max_samples]
            logger.info(f"限制样本数: {max_samples}")

    def _extract_positives(self, item: Dict, is_simple_format: bool) -> List[str]:
        """提取正例"""
        if is_simple_format:
            return item.get("positives", [])

        positives = []
        if item.get("answer"):
            positives.append(item["answer"])
        positives.extend(item.get("pos", []) or [])
        positives.extend(item.get("positives_strong", []) or [])
        positives.extend(item.get("positives_medium", []) or [])
        positives.extend(item.get("positives_weak", []) or [])
        return positives

    def _extract_negatives(self, item: Dict, is_simple_format: bool) -> List[str]:
        """提取负例"""
        if is_simple_format:
            return item.get("negatives", [])

        negatives = []
        for key in ["neg_very_hard", "neg_hard", "neg_medium",
                   "statement_very_hard_negatives", "statement_hard_negatives", "statement_medium_negatives"]:
            negs = item.get(key, []) or []
            for neg in negs:
                if isinstance(neg, dict):
                    negatives.append(neg.get("statement", "") or neg.get("text", ""))
                else:
                    negatives.append(str(neg))
        return [n.strip() for n in negatives if n and n.strip()]

    def _has_enough_docs(self, positives: List[str], negatives: List[str]) -> bool:
        """检查是否有足够的文档"""
        if self.n_docs == 0:
            # 使用所有文档模式：只检查最小值约束
            return len(positives) >= self.min_pos and len(negatives) >= self.min_neg
        elif self.n_pos > 0:
            # 固定正例数模式
            required_neg = self.n_docs - self.n_pos
            return len(positives) >= self.n_pos and len(negatives) >= required_neg
        else:
            # 动态分配模式
            if len(positives) < self.min_pos or len(negatives) < self.min_neg:
                return False
            return len(positives) + len(negatives) >= self.n_docs

    def _default_format(self, query: str, document: str) -> str:
        """默认格式化（Qwen3-Reranker 格式）"""
        return f"{PREFIX}{format_input(query, document)}{SUFFIX}"

    def _check_length(self, query: str, document: str) -> bool:
        """检查文本长度是否超过 max_length"""
        text = self.format_fn(query, document)
        tokens = self.tokenizer(text, add_special_tokens=False)
        return len(tokens["input_ids"]) <= self.max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item["query"]

        # 采样文档
        positives = item["positives"].copy()
        negatives = item["negatives"].copy()

        rng = self._get_rng()
        rng.shuffle(positives)
        rng.shuffle(negatives)

        if self.n_docs == 0:
            # 使用所有文档模式
            n_pos = len(positives)
            n_neg = len(negatives)
        elif self.n_pos > 0:
            # 固定正例数模式
            n_pos = self.n_pos
            n_neg = self.n_docs - self.n_pos
        else:
            # 动态分配模式
            total_available = len(positives) + len(negatives)
            pos_ratio = len(positives) / total_available

            n_pos = max(self.min_pos, round(self.n_docs * pos_ratio))
            n_neg = self.n_docs - n_pos

            if n_neg < self.min_neg:
                n_neg = self.min_neg
                n_pos = self.n_docs - n_neg

            n_pos = min(n_pos, len(positives))
            n_neg = min(n_neg, len(negatives))

            if n_pos + n_neg < self.n_docs:
                n_neg = min(self.n_docs - n_pos, len(negatives))
                n_pos = min(self.n_docs - n_neg, len(positives))

        # 选择正例和负例
        selected_pos = positives[:n_pos]
        selected_neg = negatives[:n_neg]

        # 合并并记录标签
        documents = selected_pos + selected_neg
        labels = [1] * len(selected_pos) + [0] * len(selected_neg)

        # 打乱顺序
        combined = list(zip(documents, labels))
        rng.shuffle(combined)
        documents, labels = zip(*combined) if combined else ([], [])

        return {
            "query": query,
            "documents": list(documents),
            "labels": list(labels)
        }

    def _get_rng(self) -> random.Random:
        """为多 worker 提供独立 RNG，避免采样重复。"""
        worker_info = get_worker_info()
        if worker_info is None:
            return self.rng
        if self._worker_rng is None or self._worker_id != worker_info.id:
            self._worker_id = worker_info.id
            self._worker_rng = random.Random(worker_info.seed)
        return self._worker_rng


class RLCollator:
    """RL 数据整理器

    将一个 batch 的样本整理成模型输入格式。

    Args:
        tokenizer: tokenizer 实例
        max_length: 最大序列长度
        format_fn: 自定义格式化函数
        pad_to_multiple_of: 可选，对齐到倍数（提升张量核效率）

    Returns:
        整理后的 batch:
        - input_ids: [batch_size * n_docs, seq_len]
        - attention_mask: [batch_size * n_docs, seq_len]
        - labels: [batch_size * n_docs]
        - group_sizes: List[int] - 每个 query 的文档数
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 4096,
        format_fn: Optional[callable] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_fn = format_fn or self._default_format
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict]) -> Dict[str, Any]:
        """整理一个 batch"""
        all_texts = []
        all_labels = []
        group_sizes = []

        for item in batch:
            query = item["query"]
            documents = item["documents"]
            labels = item["labels"]

            group_sizes.append(len(documents))

            for doc, label in zip(documents, labels):
                text = self.format_fn(query, doc)
                all_texts.append(text)
                all_labels.append(label)

        encoded = self.tokenizer(
            all_texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "group_sizes": group_sizes
        }

    def _default_format(self, query: str, document: str) -> str:
        """默认格式化（Qwen3-Reranker 格式）"""
        return f"{PREFIX}{format_input(query, document)}{SUFFIX}"


class StreamingRLRerankDataset(IterableDataset):
    """流式版本的 RLRerankDataset（用于大数据集，避免内存占用）。"""

    def __init__(
        self,
        data_file: str,
        tokenizer=None,
        max_length: int = 4096,
        n_docs: int = 6,
        n_pos: int = 0,
        min_pos: int = 1,
        min_neg: int = 1,
        max_docs: int = 0,
        seed: int = 42,
        max_samples: int = 0,
        format_fn: Optional[callable] = None,
        filter_overlength: bool = False,
    ):
        self.data_file = data_file
        self.n_docs = n_docs
        self.n_pos = n_pos
        self.min_pos = min_pos
        self.min_neg = min_neg
        self.max_docs = max_docs
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.format_fn = format_fn or self._default_format
        self.filter_overlength = filter_overlength
        self.seed = seed
        self.rng = random.Random(seed)
        self._worker_rng = None
        self._worker_id = None
        self.max_samples = max_samples

    def __iter__(self):
        rng = self._get_rng()
        count = 0
        for item in iter_data(self.data_file):
            query = item.get("query", "")
            if not query:
                continue

            sample = item if item else {}
            is_simple_format = 'positives' in sample and 'negatives' in sample

            positives = self._extract_positives(item, is_simple_format)
            negatives = self._extract_negatives(item, is_simple_format)

            if self.tokenizer is not None and self.filter_overlength:
                positives = [p for p in positives if self._check_length(query, p)]
                negatives = [n for n in negatives if self._check_length(query, n)]

            if self.max_docs > 0:
                total_docs = len(positives) + len(negatives)
                if total_docs > self.max_docs:
                    pos_ratio = len(positives) / total_docs if total_docs else 0
                    max_pos = max(self.min_pos, int(self.max_docs * pos_ratio))
                    max_neg = self.max_docs - max_pos
                    positives = positives[:max_pos]
                    negatives = negatives[:max_neg]

            if not self._has_enough_docs(positives, negatives):
                continue

            rng.shuffle(positives)
            rng.shuffle(negatives)

            if self.n_docs == 0:
                n_pos = len(positives)
                n_neg = len(negatives)
            elif self.n_pos > 0:
                n_pos = self.n_pos
                n_neg = self.n_docs - self.n_pos
            else:
                total_available = len(positives) + len(negatives)
                pos_ratio = len(positives) / total_available

                n_pos = max(self.min_pos, round(self.n_docs * pos_ratio))
                n_neg = self.n_docs - n_pos

                if n_neg < self.min_neg:
                    n_neg = self.min_neg
                    n_pos = self.n_docs - n_neg

                n_pos = min(n_pos, len(positives))
                n_neg = min(n_neg, len(negatives))

                if n_pos + n_neg < self.n_docs:
                    n_neg = min(self.n_docs - n_pos, len(negatives))
                    n_pos = min(self.n_docs - n_neg, len(positives))

            selected_pos = positives[:n_pos]
            selected_neg = negatives[:n_neg]

            documents = selected_pos + selected_neg
            labels = [1] * len(selected_pos) + [0] * len(selected_neg)
            combined = list(zip(documents, labels))
            rng.shuffle(combined)
            documents, labels = zip(*combined) if combined else ([], [])

            yield {
                "query": query,
                "documents": list(documents),
                "labels": list(labels),
            }

            count += 1
            if self.max_samples > 0 and count >= self.max_samples:
                break

    def _extract_positives(self, item: Dict, is_simple_format: bool) -> List[str]:
        if is_simple_format:
            return item.get("positives", [])
        positives = []
        if item.get("answer"):
            positives.append(item["answer"])
        positives.extend(item.get("pos", []) or [])
        positives.extend(item.get("positives_strong", []) or [])
        positives.extend(item.get("positives_medium", []) or [])
        positives.extend(item.get("positives_weak", []) or [])
        return positives

    def _extract_negatives(self, item: Dict, is_simple_format: bool) -> List[str]:
        if is_simple_format:
            return item.get("negatives", [])
        negatives = []
        for key in ["neg_very_hard", "neg_hard", "neg_medium",
                   "statement_very_hard_negatives", "statement_hard_negatives", "statement_medium_negatives"]:
            negs = item.get(key, []) or []
            for neg in negs:
                if isinstance(neg, dict):
                    negatives.append(neg.get("statement", "") or neg.get("text", ""))
                else:
                    negatives.append(str(neg))
        return [n.strip() for n in negatives if n and n.strip()]

    def _has_enough_docs(self, positives: List[str], negatives: List[str]) -> bool:
        if self.n_docs == 0:
            return len(positives) >= self.min_pos and len(negatives) >= self.min_neg
        elif self.n_pos > 0:
            required_neg = self.n_docs - self.n_pos
            return len(positives) >= self.n_pos and len(negatives) >= required_neg
        else:
            if len(positives) < self.min_pos or len(negatives) < self.min_neg:
                return False
            return len(positives) + len(negatives) >= self.n_docs

    def _default_format(self, query: str, document: str) -> str:
        return f"{PREFIX}{format_input(query, document)}{SUFFIX}"

    def _check_length(self, query: str, document: str) -> bool:
        text = self.format_fn(query, document)
        tokens = self.tokenizer(text, add_special_tokens=False)
        return len(tokens["input_ids"]) <= self.max_length

    def _get_rng(self) -> random.Random:
        worker_info = get_worker_info()
        if worker_info is None:
            return self.rng
        if self._worker_rng is None or self._worker_id != worker_info.id:
            self._worker_id = worker_info.id
            self._worker_rng = random.Random(worker_info.seed)
        return self._worker_rng
