"""
Training module for Qwen3-Reranker

提供 SFT 和 RL 训练的核心组件：

SFT 训练:
- RerankDataset: 重排序数据集
- RerankCollator: 数据整理器
- ContrastiveSFTTrainer: 对比学习 SFT 训练器

RL 训练:
- RLRerankDataset: RL 数据集（支持 max_docs 限制）
- RLCollator: RL 批次构造器
- RLTrainer: Doc-level REINFORCE 训练器
- load_sft_model: 加载 SFT 模型并合并 LoRA

Usage:
    # SFT 训练
    from qwen3_rerank_trainer.training import (
        RerankDataset,
        RerankCollator,
        ContrastiveSFTTrainer,
    )

    # RL 训练
    from qwen3_rerank_trainer.training import (
        RLRerankDataset,
        RLCollator,
        RLTrainer,
        load_sft_model,
    )
"""

# SFT 训练组件
from .dataset import RerankDataset, StreamingRerankDataset, load_data, iter_data
from .collator import RerankCollator
from .sft_trainer import ContrastiveSFTTrainer

# RL 训练组件
from .rl_dataset import RLRerankDataset, StreamingRLRerankDataset, RLCollator
from .rl_trainer import RLTrainer, load_sft_model

__all__ = [
    # SFT
    "RerankDataset",
    "StreamingRerankDataset",
    "RerankCollator",
    "ContrastiveSFTTrainer",
    "load_data",
    "iter_data",
    # RL
    "RLRerankDataset",
    "StreamingRLRerankDataset",
    "RLCollator",
    "RLTrainer",
    "load_sft_model",
]
