"""
评估模块

提供排序评估功能：
- 评估指标 (MRR, AP, NDCG@k, P@k, R@k)
- MTEB Reranking 评估
- 两阶段评估 (Embedding + Rerank)
- 评估报告生成

使用示例:
    # 评估指标
    from qwen3_rerank_trainer.evaluation import mrr, ndcg_at_k, compute_all_metrics

    # MTEB 评估
    from qwen3_rerank_trainer.evaluation import (
        MTEBRerankEvaluator,
        set_proxy,
        evaluate_reranking_dataset,
    )

    # 两阶段评估
    from qwen3_rerank_trainer.evaluation import (
        TwoStageEvaluator,
        run_two_stage_eval,
    )

    # 报告生成
    from qwen3_rerank_trainer.evaluation import (
        generate_report,
        print_results_summary,
    )
"""

# ============================================================================
# 评估指标
# ============================================================================

from .metrics import (
    # 基于 ranking 的指标
    mrr,
    ap,
    ndcg_at_k,
    ndcg_at_k_binary,
    precision_at_k,
    recall_at_k,
    hit_at_k,
    success_at_k,  # 别名
    f1_at_k,
    # 基于 scores 的指标
    mrr_from_scores,
    ap_from_scores,
    ndcg_from_scores,
    precision_from_scores,
    recall_from_scores,
    hit_from_scores,
    # 基于 sorted_labels 的指标
    mrr_from_sorted_labels,
    ap_from_sorted_labels,
    ndcg_from_sorted_labels,
    precision_from_sorted_labels,
    recall_from_sorted_labels,
    hit_from_sorted_labels,
    # 批量计算
    compute_all_metrics,
    aggregate_metrics,
    # 兼容性函数
    calculate_mrr,
    calculate_ndcg,
    calculate_ap,
)

# ============================================================================
# MTEB 评估
# ============================================================================

from .mteb_runner import (
    # 代理配置
    set_proxy,
    clear_proxy,
    # 数据集配置
    RERANKING_DATASETS,
    DATASET_GROUPS,
    expand_dataset_names,
    # 本地数据集
    LocalDataset,
    # 评估器
    MTEBRerankEvaluator,
    # 便捷函数
    evaluate_reranking_dataset,
    evaluate_local_dataset,
    # 多模型并行评估
    evaluate_multiple_models,
    print_comparison_table as print_model_comparison,
)

# ============================================================================
# API 客户端
# ============================================================================

from .api_client import (
    # 异步调用
    call_rerank_async,
    call_rerank_async_safe,
    call_rerank_batch_async,
    # 同步调用
    call_rerank,
    call_rerank_batch,
    # Reranker 类
    APIReranker,
    create_api_reranker,
)

# ============================================================================
# GPU 负载均衡
# ============================================================================

from .gpu_utils import (
    # 工具函数
    get_interleaved_order,
    get_gpu_semaphores,
    print_gpu_balance_info,
    # 执行函数
    run_with_gpu_balance,
    run_with_gpu_balance_async,
    # 高层接口
    evaluate_with_gpu_balance,
)

# ============================================================================
# 两阶段评估
# ============================================================================

from .two_stage_eval import (
    # 数据集配置
    RETRIEVAL_DATASETS,
    # 评估配置
    TwoStageEvalConfig,
    # 评估器
    TwoStageEvaluator,
    APIRerankModel,
    # 便捷函数
    run_two_stage_eval,
)

# ============================================================================
# 报告生成
# ============================================================================

from .report import (
    # 配置
    ReportConfig,
    # 打印函数
    print_results_summary,
    print_comparison_table,
    # 报告生成
    generate_report,
    # 结果保存/加载
    save_results_json,
    load_results_json,
)

__all__ = [
    # 基于 ranking 的指标
    "mrr",
    "ap",
    "ndcg_at_k",
    "ndcg_at_k_binary",
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "success_at_k",
    "f1_at_k",
    # 基于 scores 的指标
    "mrr_from_scores",
    "ap_from_scores",
    "ndcg_from_scores",
    "precision_from_scores",
    "recall_from_scores",
    "hit_from_scores",
    # 基于 sorted_labels 的指标
    "mrr_from_sorted_labels",
    "ap_from_sorted_labels",
    "ndcg_from_sorted_labels",
    "precision_from_sorted_labels",
    "recall_from_sorted_labels",
    "hit_from_sorted_labels",
    # 批量计算
    "compute_all_metrics",
    "aggregate_metrics",
    # 兼容性函数
    "calculate_mrr",
    "calculate_ndcg",
    "calculate_ap",
    # 代理配置
    "set_proxy",
    "clear_proxy",
    # 数据集配置
    "RERANKING_DATASETS",
    "RETRIEVAL_DATASETS",
    "DATASET_GROUPS",
    "expand_dataset_names",
    # 本地数据集
    "LocalDataset",
    # MTEB 评估器
    "MTEBRerankEvaluator",
    "evaluate_reranking_dataset",
    "evaluate_local_dataset",
    # 多模型并行评估
    "evaluate_multiple_models",
    "print_model_comparison",
    # API 客户端
    "call_rerank_async",
    "call_rerank_async_safe",
    "call_rerank_batch_async",
    "call_rerank",
    "call_rerank_batch",
    "APIReranker",
    "create_api_reranker",
    # GPU 负载均衡
    "get_interleaved_order",
    "get_gpu_semaphores",
    "print_gpu_balance_info",
    "run_with_gpu_balance",
    "run_with_gpu_balance_async",
    "evaluate_with_gpu_balance",
    # 两阶段评估
    "TwoStageEvalConfig",
    "TwoStageEvaluator",
    "APIRerankModel",
    "run_two_stage_eval",
    # 报告生成
    "ReportConfig",
    "print_results_summary",
    "print_comparison_table",
    "generate_report",
    "save_results_json",
    "load_results_json",
]
