"""
Data Collator for Rerank Training

将数据集样本转换为模型输入。
"""
import torch
from typing import List, Dict, Optional, Callable

from ..data.formatting import PREFIX, SUFFIX, format_input


class RerankCollator:
    """数据整理器：将样本转换为模型输入

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: 最大序列长度
        format_fn: 自定义格式化函数，接收 (query, document) 返回格式化文本
        pad_to_multiple_of: 可选，对齐到倍数（提升张量核效率）

    Example:
        >>> collator = RerankCollator(tokenizer, max_length=4096)
        >>> batch = collator([dataset[0], dataset[1]])
        >>> print(batch.keys())  # dict_keys(['input_ids', 'attention_mask', 'labels'])
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 4096,
        format_fn: Optional[Callable[[str, str], str]] = None,
        pad_to_multiple_of: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.format_fn = format_fn or self._default_format
        self.pad_to_multiple_of = pad_to_multiple_of

    def _default_format(self, query: str, document: str) -> str:
        """默认格式化（Qwen3-Reranker 格式）"""
        return f"{PREFIX}{format_input(query, document)}{SUFFIX}"

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """整理一个 batch 的数据

        Args:
            batch: 数据集返回的样本列表，每个样本包含 query, positives, negatives

        Returns:
            dict: 包含 input_ids, attention_mask, labels 的字典
                - labels: 1 for positive, 0 for negative
        """
        all_texts = []
        all_labels = []
        group_sizes = []  # 记录每个样本的文档数

        for item in batch:
            query = item["query"]
            positives = item["positives"]
            negatives = item["negatives"]

            sample_size = 0

            # 正例
            for pos in positives:
                pos_text = self.format_fn(query, pos)
                all_texts.append(pos_text)
                all_labels.append(1)
                sample_size += 1

            # 负例
            for neg in negatives:
                neg_text = self.format_fn(query, neg)
                all_texts.append(neg_text)
                all_labels.append(0)
                sample_size += 1

            group_sizes.append(sample_size)

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
            "group_sizes": group_sizes,  # 用于 RL 训练中的分组
        }
