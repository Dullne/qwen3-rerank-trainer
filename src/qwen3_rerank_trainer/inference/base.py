"""
Reranker 抽象基类
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union


class BaseReranker(ABC):
    """Reranker 抽象基类"""

    @abstractmethod
    def rerank(
        self,
        query: Union[str, List[str]],
        documents: List[str],
        instruction: Optional[str] = None,
        top_k: Optional[int] = None,
        return_scores: bool = False,
        batch_size: int = 32,
    ):
        """
        重排序文档

        Args:
            query: 查询文本（支持批量）
            documents: 候选文档列表
            instruction: 任务指令
            top_k: 返回前k个文档
            return_scores: 是否返回分数
            batch_size: 批处理大小

        Returns:
            排序后的文档（和分数）
        """
        raise NotImplementedError
