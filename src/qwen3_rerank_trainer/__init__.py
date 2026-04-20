"""
qwen3-rerank-trainer: Training and evaluation toolkit for Qwen3-Reranker

提供用于 Qwen3-Reranker 训练和评估的核心组件：
- 损失函数（LambdaLoss、ListMLE、InfoNCE 等）
- SFT 训练（ContrastiveSFTTrainer）
- RL 训练（REINFORCE、DPO）
- 评估指标（MRR、NDCG、AP 等）
- 推理引擎（Qwen3-Reranker）
- 数据处理（格式化、采样、Tokenization）

Usage:
    from qwen3_rerank_trainer import lambda_loss, mrr, Qwen3Reranker
    from qwen3_rerank_trainer.losses import NDCGLoss2PPScheme
    from qwen3_rerank_trainer.evaluation import compute_all_metrics
    from qwen3_rerank_trainer.training import ContrastiveSFTTrainer
"""

__version__ = "0.1.4"

# ============================================================================
# 损失函数
# ============================================================================
from .losses import (
    # LambdaLoss 框架
    lambda_loss,
    # 权重方案
    NDCGLoss2PPScheme,
    MAPScheme,
    MRRScheme,
    # Listwise
    list_mle,
    p_list_mle,
    listwise_softmax_ce,
    # Pairwise
    ranknet_loss,
    pairwise_posrank_loss,
    # Pointwise
    yes_no_to_score,
    pointwise_ce_from_yes_no_logits,
    # Contrastive
    infonce_loss,
)

# ============================================================================
# RL 训练
# ============================================================================
from .rl import (
    reinforce_loss,
    dpo_loss,
    compute_doc_level_rewards,
    compute_doc_level_advantages,
)

# ============================================================================
# 评估指标
# ============================================================================
from .evaluation import (
    mrr,
    ap,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    hit_at_k,
    compute_all_metrics,
    aggregate_metrics,
    # scores 接口
    mrr_from_scores,
    ndcg_from_scores,
    ap_from_scores,
    # 代理配置
    set_proxy,
    clear_proxy,
    # MTEB 评估
    MTEBRerankEvaluator,
    evaluate_reranking_dataset,
    # 报告生成
    print_results_summary,
    generate_report,
)

# ============================================================================
# 数据处理
# ============================================================================
from .data import (
    PREFIX,
    SUFFIX,
    DEFAULT_INSTRUCTION,
    format_input,
    sample_documents,
    tokenize_for_training,
    extract_yes_no_logits,
    compute_scores,
)

# ============================================================================
# 推理 (可选依赖)
# ============================================================================
try:
    from .inference import Qwen3Reranker
except ImportError:
    Qwen3Reranker = None

# ============================================================================
# SFT 训练 (可选依赖)
# ============================================================================
try:
    from .training import (
        RerankDataset,
        StreamingRerankDataset,
        RerankCollator,
        ContrastiveSFTTrainer,
    )
except ImportError:
    RerankDataset = None
    StreamingRerankDataset = None
    RerankCollator = None
    ContrastiveSFTTrainer = None

# ============================================================================
# RL 训练 (可选依赖)
# ============================================================================
try:
    from .training import (
        RLRerankDataset,
        StreamingRLRerankDataset,
        RLCollator,
        RLTrainer,
        load_sft_model,
    )
except ImportError:
    RLRerankDataset = None
    StreamingRLRerankDataset = None
    RLCollator = None
    RLTrainer = None
    load_sft_model = None


__all__ = [
    # 版本
    "__version__",
    # 损失函数
    "lambda_loss",
    "NDCGLoss2PPScheme",
    "MAPScheme",
    "MRRScheme",
    "list_mle",
    "p_list_mle",
    "listwise_softmax_ce",
    "ranknet_loss",
    "pairwise_posrank_loss",
    "yes_no_to_score",
    "pointwise_ce_from_yes_no_logits",
    "infonce_loss",
    # RL
    "reinforce_loss",
    "dpo_loss",
    "compute_doc_level_rewards",
    "compute_doc_level_advantages",
    # 评估
    "mrr",
    "ap",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "compute_all_metrics",
    "aggregate_metrics",
    "mrr_from_scores",
    "ndcg_from_scores",
    "ap_from_scores",
    # 代理配置
    "set_proxy",
    "clear_proxy",
    # MTEB 评估
    "MTEBRerankEvaluator",
    "evaluate_reranking_dataset",
    # 报告生成
    "print_results_summary",
    "generate_report",
    # 数据
    "PREFIX",
    "SUFFIX",
    "DEFAULT_INSTRUCTION",
    "format_input",
    "sample_documents",
    "tokenize_for_training",
    "extract_yes_no_logits",
    "compute_scores",
    # 推理
    "Qwen3Reranker",
    # SFT 训练
    "RerankDataset",
    "StreamingRerankDataset",
    "RerankCollator",
    "ContrastiveSFTTrainer",
    # RL 训练
    "RLRerankDataset",
    "StreamingRLRerankDataset",
    "RLCollator",
    "RLTrainer",
    "load_sft_model",
]
