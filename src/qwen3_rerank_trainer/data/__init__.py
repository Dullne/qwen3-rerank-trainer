"""
数据处理模块

提供 Qwen3-Reranker 格式的数据处理函数。
"""

from .formatting import (
    PREFIX,
    SUFFIX,
    DEFAULT_INSTRUCTION,
    format_input,
)

from .sampling import (
    sample_documents,
    sample_documents_by_score,
)

from .tokenization import (
    tokenize_for_training,
    extract_yes_no_logits,
    compute_scores,
    forward_and_get_logits,
)

__all__ = [
    # 格式化常量
    "PREFIX",
    "SUFFIX",
    "DEFAULT_INSTRUCTION",
    "format_input",
    # 采样
    "sample_documents",
    "sample_documents_by_score",
    # Tokenization
    "tokenize_for_training",
    "extract_yes_no_logits",
    "compute_scores",
    "forward_and_get_logits",
]
