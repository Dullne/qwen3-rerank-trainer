#!/usr/bin/env python3
"""
MTEB Reranking 评估命令行工具

提供命令行接口进行 Reranking 模型评估：
- 支持单模型/多模型评估
- 支持本地数据集和 MTEB 数据集
- 支持 GPU 负载均衡

使用示例:
    # 评估单个端点
    qwen3-rerank-eval --endpoint http://localhost:9997 --datasets chinese

    # 评估多个端点
    qwen3-rerank-eval --endpoints http://localhost:9997 http://localhost:9998 --datasets chinese

    # 评估本地数据集
    qwen3-rerank-eval --endpoint http://localhost:9997 --input data.jsonl

    # 列出支持的数据集
    qwen3-rerank-eval --list-datasets
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from .mteb_runner import (
    RERANKING_DATASETS,
    DATASET_GROUPS,
    expand_dataset_names,
    set_proxy,
    MTEBRerankEvaluator,
    LocalDataset,
    evaluate_multiple_models,
)
from .api_client import APIReranker, create_api_reranker
from .gpu_utils import (
    get_interleaved_order,
    run_with_gpu_balance,
    print_gpu_balance_info,
)
from .report import (
    ReportConfig,
    generate_report,
    print_results_summary,
    print_comparison_table,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='MTEB Reranking 模型评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估单个端点的中文数据集
  %(prog)s --endpoint http://localhost:9997 --datasets chinese

  # 评估多个端点（自动 GPU 负载均衡）
  %(prog)s --endpoints http://localhost:9997 http://localhost:9998 \\
           --datasets chinese --model-workers 2

  # 评估本地数据集
  %(prog)s --endpoint http://localhost:9997 --input data.jsonl

  # 列出支持的数据集
  %(prog)s --list-datasets

数据集组:
  chinese/cn    : 4 个中文数据集
  english/en    : 6 个英文数据集
  multilingual  : 3 个多语言数据集
  other         : 5 个其他语言数据集
  code          : 4 个代码数据集
  all           : 全部 22 个数据集
        """
    )

    # 端点配置
    endpoint_group = parser.add_mutually_exclusive_group()
    endpoint_group.add_argument(
        '--endpoint', '-e',
        type=str,
        help='Rerank API 端点地址（单模型）'
    )
    endpoint_group.add_argument(
        '--endpoints',
        type=str,
        nargs='+',
        help='多个 Rerank API 端点地址（多模型）'
    )

    # 数据集配置
    parser.add_argument(
        '--datasets', '-d',
        type=str,
        nargs='+',
        default=['chinese'],
        help='评估的数据集或数据集组（默认: chinese）'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='本地数据文件路径（jsonl 格式）'
    )

    # 评估参数
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='每个数据集的最大样本数（默认: 全部）'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=50,
        help='每次请求的最大文档数（避免 OOM，默认: 50）'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='单模型 API 请求并发数（默认: 8）'
    )
    parser.add_argument(
        '--model-workers',
        type=int,
        default=2,
        help='多模型并行评估数（默认: 2）'
    )

    # GPU 配置
    parser.add_argument(
        '--gpu-map',
        type=str,
        nargs='+',
        help='GPU 映射，格式: endpoint:gpu_id（如 localhost:9997:0 localhost:9998:1）'
    )

    # 输出配置
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='eval_output',
        help='输出目录（默认: eval_output）'
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen3-Reranker-4B',
        help='模型名称（默认: Qwen3-Reranker-4B）'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        help='基准模型名称（用于对比）'
    )

    # 代理配置
    parser.add_argument(
        '--proxy',
        type=str,
        help='HTTP 代理地址'
    )

    # 信息显示
    parser.add_argument(
        '--list-datasets',
        action='store_true',
        help='列出所有支持的数据集'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )

    return parser.parse_args()


def list_datasets():
    """列出所有支持的数据集"""
    print("\n支持的 MTEB Reranking 数据集:")
    print("=" * 60)

    # 按语言分组
    lang_groups = {}
    for name, config in RERANKING_DATASETS.items():
        lang = config.get('lang', 'unknown')
        if lang not in lang_groups:
            lang_groups[lang] = []
        lang_groups[lang].append((name, config.get('split', 'dev')))

    lang_names = {
        'zh': '中文',
        'en': '英文',
        'multilingual': '多语言',
        'ru': '俄语',
        'ja': '日语',
        'fr': '法语',
        'ar': '阿拉伯语',
        'code': '代码',
    }

    for lang, datasets in sorted(lang_groups.items()):
        lang_display = lang_names.get(lang, lang)
        print(f"\n{lang_display} ({len(datasets)} 个):")
        for name, split in datasets:
            print(f"  - {name} ({split})")

    print("\n" + "=" * 60)
    print("数据集组:")
    for group, datasets in DATASET_GROUPS.items():
        if group not in ['cn', 'en', 'multi']:  # 跳过别名
            print(f"  {group}: {len(datasets)} 个")

    print(f"\n总计: {len(RERANKING_DATASETS)} 个数据集")


def get_endpoint_name(endpoint: str) -> str:
    """从端点 URL 提取名称（端口号）"""
    parsed = urlparse(endpoint)
    return str(parsed.port) if parsed.port else endpoint


def parse_gpu_map(gpu_map_args: Optional[List[str]]) -> Dict[str, int]:
    """解析 GPU 映射参数"""
    if not gpu_map_args:
        return {}

    gpu_info = {}
    for item in gpu_map_args:
        parts = item.rsplit(':', 1)
        if len(parts) == 2:
            endpoint_part, gpu_id = parts
            try:
                gpu_info[endpoint_part] = int(gpu_id)
            except ValueError:
                logger.warning(f"无效的 GPU ID: {gpu_id}")
    return gpu_info


def evaluate_single_endpoint(
    endpoint: str,
    datasets: List[str],
    max_samples: Optional[int],
    batch_size: int,
    workers: int,
    model_name: str,
    output_dir: Path,
    input_file: Optional[str] = None,
) -> Dict:
    """评估单个端点"""
    # 创建 API Reranker
    reranker = create_api_reranker(
        endpoint,
        model=model_name,
        batch_size=batch_size,
        max_concurrency=workers,
    )

    # 测试连接
    if not reranker.test_connection():
        logger.error(f"API 连接失败: {endpoint}")
        return {"error": "API connection failed"}

    # 创建评估器
    evaluator = MTEBRerankEvaluator(
        reranker=reranker,
        batch_size=batch_size,
        workers=workers,
    )

    endpoint_name = get_endpoint_name(endpoint)

    # 本地数据集评估
    if input_file:
        logger.info(f"评估本地数据集: {input_file}")
        results = evaluator.evaluate_local(
            dataset=input_file,
            max_samples=max_samples,
        )
        return {"local": results}

    # MTEB 数据集评估
    expanded_datasets = expand_dataset_names(datasets)
    logger.info(f"评估 {len(expanded_datasets)} 个数据集: {expanded_datasets}")

    results = evaluator.evaluate_multiple(
        task_names=expanded_datasets,
        max_samples=max_samples,
    )

    return results


def evaluate_multiple_endpoints(
    endpoints: List[str],
    datasets: List[str],
    max_samples: Optional[int],
    batch_size: int,
    workers: int,
    model_workers: int,
    model_name: str,
    output_dir: Path,
    gpu_info: Dict[str, int],
    baseline: Optional[str] = None,
) -> Dict[str, Dict]:
    """评估多个端点"""
    # 创建 Rerankers
    rerankers = {}
    for endpoint in endpoints:
        name = get_endpoint_name(endpoint)
        rerankers[name] = create_api_reranker(
            endpoint,
            model=model_name,
            batch_size=batch_size,
            max_concurrency=workers,
        )

    # 测试连接
    for name, reranker in list(rerankers.items()):
        if not reranker.test_connection():
            logger.warning(f"API 连接失败，跳过: {name}")
            del rerankers[name]

    if not rerankers:
        logger.error("所有端点连接失败")
        return {}

    # 解析 GPU 映射（将端口号映射）
    model_gpu_info = {}
    for name in rerankers.keys():
        for key, gpu_id in gpu_info.items():
            if name in key or key in name:
                model_gpu_info[name] = gpu_id
                break
        if name not in model_gpu_info:
            model_gpu_info[name] = 0  # 默认 GPU 0

    # 如果有 GPU 信息，使用 GPU 负载均衡
    if model_gpu_info and len(set(model_gpu_info.values())) > 1:
        from .gpu_utils import evaluate_with_gpu_balance
        results = evaluate_with_gpu_balance(
            rerankers=rerankers,
            gpu_info=model_gpu_info,
            task_names=datasets,
            model_workers=model_workers,
            batch_size=batch_size,
            workers=workers,
            max_samples=max_samples,
        )
    else:
        # 普通多模型评估
        results = evaluate_multiple_models(
            rerankers=rerankers,
            task_names=datasets,
            max_samples=max_samples,
            model_workers=model_workers,
            batch_size=batch_size,
            workers=workers,
        )

    return results


def main():
    """主函数"""
    args = parse_args()

    # 列出数据集
    if args.list_datasets:
        list_datasets()
        return 0

    # 检查端点配置
    if not args.endpoint and not args.endpoints:
        print("错误: 请指定 --endpoint 或 --endpoints")
        print("使用 --help 查看帮助")
        return 1

    # 设置代理
    if args.proxy:
        set_proxy(args.proxy)

    # 准备输出目录
    timestamp = (datetime.now() + timedelta(hours=8)).strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output) / f"eval_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"输出目录: {output_dir}")

    # 单端点评估
    if args.endpoint:
        results = evaluate_single_endpoint(
            endpoint=args.endpoint,
            datasets=args.datasets,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            workers=args.workers,
            model_name=args.model_name,
            output_dir=output_dir,
            input_file=args.input,
        )

        # 打印结果
        print_results_summary(results)

        # 生成报告
        config = ReportConfig(baseline_model=args.baseline)
        generate_report(results, output_dir, config)

    # 多端点评估
    else:
        gpu_info = parse_gpu_map(args.gpu_map)

        results = evaluate_multiple_endpoints(
            endpoints=args.endpoints,
            datasets=args.datasets,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            workers=args.workers,
            model_workers=args.model_workers,
            model_name=args.model_name,
            output_dir=output_dir,
            gpu_info=gpu_info,
            baseline=args.baseline,
        )

        if not results:
            logger.error("评估失败")
            return 1

        # 转换为报告格式
        report_results = [
            {"model": name, "results": model_results}
            for name, model_results in results.items()
        ]

        # 打印对比表
        print_comparison_table(report_results)

        # 生成报告
        config = ReportConfig(baseline_model=args.baseline)
        generate_report(report_results, output_dir, config)

    logger.info(f"\n评估完成！结果保存在: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
