"""
Qwen3-Reranker 格式化模块

提供 prompt 格式化常量和函数。
"""

from typing import Optional


# Qwen3-Reranker 官方 prompt 格式
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"


def format_input(
    query: str,
    doc: str,
    instruction: Optional[str] = None,
) -> str:
    """格式化单个 (query, doc) 对的输入文本

    Args:
        query: 查询
        doc: 文档
        instruction: 任务指令

    Returns:
        格式化后的文本（不含 prefix/suffix）
    """
    if instruction is None:
        instruction = DEFAULT_INSTRUCTION

    return "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(
        instruction=instruction,
        query=query,
        doc=doc
    )
