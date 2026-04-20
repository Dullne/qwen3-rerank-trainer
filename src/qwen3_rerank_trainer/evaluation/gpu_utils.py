"""
GPU 负载均衡工具模块

提供多模型评估时的 GPU 负载均衡功能，避免同一 GPU 上多个模型同时推理导致 OOM。

核心思想：
1. 交错调度：不同 GPU 的任务交替执行，避免一个 GPU 被排满
2. GPU 信号量：限制每个 GPU 上的并发数，避免 OOM

使用示例:
    from qwen3_rerank_trainer.evaluation import (
        evaluate_with_gpu_balance,
        get_interleaved_order,
    )

    # 定义模型和 GPU 映射
    gpu_info = {
        "model_a": 0,  # GPU 0
        "model_b": 0,  # GPU 0
        "model_c": 1,  # GPU 1
        "model_d": 1,  # GPU 1
    }

    # 带 GPU 负载均衡的评估
    results = evaluate_with_gpu_balance(
        rerankers=rerankers,
        gpu_info=gpu_info,
        task_names=["T2Reranking"],
        model_workers=2,  # 总并发数（会按 GPU 分配）
    )
"""

import asyncio
import threading
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


# ============================================================================
# 交错排序
# ============================================================================

def get_interleaved_order(
    models: List[str],
    gpu_info: Dict[str, int],
) -> List[str]:
    """
    按 GPU 交错排序模型，确保不同 GPU 的任务交替提交

    Args:
        models: 模型名称列表
        gpu_info: 模型到 GPU ID 的映射 {model_name: gpu_id}

    Returns:
        交错排序后的模型列表

    Example:
        >>> models = ["a", "b", "c", "d"]
        >>> gpu_info = {"a": 0, "b": 0, "c": 1, "d": 1}
        >>> get_interleaved_order(models, gpu_info)
        ['a', 'c', 'b', 'd']  # GPU0, GPU1, GPU0, GPU1
    """
    # 按 GPU 分组
    gpu_models = {}
    for model in models:
        gpu = gpu_info.get(model, 0)
        if gpu not in gpu_models:
            gpu_models[gpu] = []
        gpu_models[gpu].append(model)

    # 交错排序
    interleaved = []
    gpu_queues = {gpu: list(ms) for gpu, ms in gpu_models.items()}
    while any(gpu_queues.values()):
        for gpu in sorted(gpu_queues.keys()):
            if gpu_queues[gpu]:
                interleaved.append(gpu_queues[gpu].pop(0))

    return interleaved


# ============================================================================
# GPU 信号量
# ============================================================================

def get_gpu_semaphores(
    models: List[str],
    gpu_info: Dict[str, int],
    total_workers: int,
) -> Dict[int, threading.Semaphore]:
    """
    计算每个 GPU 的信号量

    按 GPU 数量均分 workers，确保每个 GPU 的负载均衡。

    Args:
        models: 模型名称列表
        gpu_info: 模型到 GPU ID 的映射
        total_workers: 总并发数

    Returns:
        {gpu_id: Semaphore} 字典
    """
    # 按 GPU 分组
    gpu_models = {}
    for model in models:
        gpu = gpu_info.get(model, 0)
        if gpu not in gpu_models:
            gpu_models[gpu] = []
        gpu_models[gpu].append(model)

    num_gpus = len(gpu_models)
    if num_gpus == 0 or total_workers == 0:
        return {}

    # 均分 workers
    base_per_gpu = total_workers // num_gpus
    extra = total_workers % num_gpus

    gpu_semaphores = {}
    for i, (gpu_id, gpu_ms) in enumerate(sorted(gpu_models.items())):
        gpu_worker_count = base_per_gpu + (1 if i < extra else 0)
        # 不能超过该 GPU 上的模型数
        gpu_worker_count = min(gpu_worker_count, len(gpu_ms))
        gpu_worker_count = max(1, gpu_worker_count)
        gpu_semaphores[gpu_id] = threading.Semaphore(gpu_worker_count)

    return gpu_semaphores


def get_gpu_semaphores_async(
    models: List[str],
    gpu_info: Dict[str, int],
    total_workers: int,
) -> Dict[int, asyncio.Semaphore]:
    """
    异步版本：计算每个 GPU 的 asyncio.Semaphore
    """
    gpu_models = {}
    for model in models:
        gpu = gpu_info.get(model, 0)
        if gpu not in gpu_models:
            gpu_models[gpu] = []
        gpu_models[gpu].append(model)

    num_gpus = len(gpu_models)
    if num_gpus == 0 or total_workers == 0:
        return {}

    base_per_gpu = total_workers // num_gpus
    extra = total_workers % num_gpus

    gpu_semaphores = {}
    for i, (gpu_id, gpu_ms) in enumerate(sorted(gpu_models.items())):
        gpu_worker_count = base_per_gpu + (1 if i < extra else 0)
        gpu_worker_count = min(gpu_worker_count, len(gpu_ms))
        gpu_worker_count = max(1, gpu_worker_count)
        gpu_semaphores[gpu_id] = asyncio.Semaphore(gpu_worker_count)

    return gpu_semaphores


# ============================================================================
# 打印信息
# ============================================================================

def print_gpu_balance_info(
    models: List[str],
    gpu_info: Dict[str, int],
    total_workers: int,
) -> None:
    """打印 GPU 负载均衡信息"""
    gpu_models = {}
    for model in models:
        gpu = gpu_info.get(model, 0)
        if gpu not in gpu_models:
            gpu_models[gpu] = []
        gpu_models[gpu].append(model)

    num_gpus = len(gpu_models)
    if num_gpus == 0:
        return

    print(f"\nGPU 负载均衡 (总 {total_workers} workers, {num_gpus} GPUs):")

    base_per_gpu = total_workers // num_gpus
    extra = total_workers % num_gpus

    for i, (gpu_id, gpu_ms) in enumerate(sorted(gpu_models.items())):
        gpu_worker_count = base_per_gpu + (1 if i < extra else 0)
        gpu_worker_count = min(gpu_worker_count, len(gpu_ms))
        gpu_worker_count = max(1, gpu_worker_count)
        print(f"  GPU {gpu_id}: {len(gpu_ms)} 个模型, 最多 {gpu_worker_count} 个并发")
        for m in gpu_ms:
            print(f"    - {m}")


# ============================================================================
# 带 GPU 负载均衡的执行
# ============================================================================

def run_with_gpu_balance(
    models: List[str],
    gpu_info: Dict[str, int],
    total_workers: int,
    eval_func: Callable[[str], Any],
    desc: str = "评估",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    带 GPU 负载均衡的多模型并发执行

    Args:
        models: 模型名称列表
        gpu_info: 模型到 GPU ID 的映射
        total_workers: 总并发数
        eval_func: 评估函数，接受模型名称参数，返回结果
        desc: 描述文字
        verbose: 是否打印进度信息

    Returns:
        {model_name: result} 字典
    """
    # 获取 GPU Semaphores
    gpu_semaphores = get_gpu_semaphores(models, gpu_info, total_workers)

    # 交错排序
    interleaved = get_interleaved_order(models, gpu_info)

    if verbose:
        print_gpu_balance_info(models, gpu_info, total_workers)
        print(f"任务顺序: {interleaved}")

    def wrapped_eval(model: str) -> tuple:
        """带 Semaphore 的评估包装"""
        gpu_id = gpu_info.get(model, 0)
        semaphore = gpu_semaphores.get(gpu_id)

        if semaphore:
            semaphore.acquire()
            if verbose:
                print(f"[GPU {gpu_id}] 开始{desc}: {model}")

        try:
            result = eval_func(model)
            if verbose:
                print(f"[GPU {gpu_id}] ✓ {model} 完成")
            return model, result
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] ✗ {model} 失败: {e}")
            return model, {"error": str(e)}
        finally:
            if semaphore:
                semaphore.release()

    # 并发执行
    results = {}
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {executor.submit(wrapped_eval, m): m for m in interleaved}

        for future in as_completed(futures):
            model, result = future.result()
            results[model] = result

    return results


async def run_with_gpu_balance_async(
    models: List[str],
    gpu_info: Dict[str, int],
    total_workers: int,
    eval_func: Callable[[str], Any],
    desc: str = "评估",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    异步版本：带 GPU 负载均衡的多模型并发执行

    Args:
        models: 模型名称列表
        gpu_info: 模型到 GPU ID 的映射
        total_workers: 总并发数
        eval_func: 异步评估函数
        desc: 描述文字
        verbose: 是否打印进度信息

    Returns:
        {model_name: result} 字典
    """
    gpu_semaphores = get_gpu_semaphores_async(models, gpu_info, total_workers)
    interleaved = get_interleaved_order(models, gpu_info)

    if verbose:
        print_gpu_balance_info(models, gpu_info, total_workers)
        print(f"任务顺序: {interleaved}")

    async def wrapped_eval(model: str) -> tuple:
        gpu_id = gpu_info.get(model, 0)
        semaphore = gpu_semaphores.get(gpu_id)

        if semaphore:
            await semaphore.acquire()
            if verbose:
                print(f"[GPU {gpu_id}] 开始{desc}: {model}")

        try:
            result = await eval_func(model)
            if verbose:
                print(f"[GPU {gpu_id}] ✓ {model} 完成")
            return model, result
        except Exception as e:
            logger.error(f"[GPU {gpu_id}] ✗ {model} 失败: {e}")
            return model, {"error": str(e)}
        finally:
            if semaphore:
                semaphore.release()

    # 并发执行
    tasks = [wrapped_eval(m) for m in interleaved]
    task_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = {}
    for r in task_results:
        if isinstance(r, Exception):
            logger.error(f"任务异常: {r}")
        else:
            model, result = r
            results[model] = result

    return results


# ============================================================================
# 高层接口：带 GPU 负载均衡的多模型评估
# ============================================================================

def evaluate_with_gpu_balance(
    rerankers: Dict[str, Any],
    gpu_info: Dict[str, int],
    task_names: List[str],
    model_workers: int = 2,
    batch_size: int = 50,
    workers: int = 8,
    max_samples: Optional[int] = None,
    ks: List[int] = None,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    带 GPU 负载均衡的多模型评估

    Args:
        rerankers: 模型名称到 Reranker 的映射
        gpu_info: 模型名称到 GPU ID 的映射
        task_names: 任务名称列表
        model_workers: 模型并发数（按 GPU 分配）
        batch_size: 每次请求的最大文档数
        workers: 单模型的 API 并发数
        max_samples: 每个数据集的最大样本数
        ks: 评估的 k 值列表
        verbose: 是否打印进度信息

    Returns:
        {model_name: {task_name: metrics_dict}}

    Example:
        >>> results = evaluate_with_gpu_balance(
        ...     rerankers={"9997": reranker_a, "9998": reranker_b},
        ...     gpu_info={"9997": 0, "9998": 1},
        ...     task_names=["chinese"],
        ...     model_workers=2,
        ... )
    """
    from .mteb_runner import MTEBRerankEvaluator, expand_dataset_names

    if ks is None:
        ks = [1, 5, 10]

    expanded_tasks = expand_dataset_names(task_names)
    models = list(rerankers.keys())

    # 创建评估器
    evaluators = {}
    for name, reranker in rerankers.items():
        if callable(reranker) and not hasattr(reranker, 'rerank'):
            evaluators[name] = MTEBRerankEvaluator(
                rerank_fn=reranker,
                batch_size=batch_size,
                workers=workers,
            )
        else:
            evaluators[name] = MTEBRerankEvaluator(
                reranker=reranker,
                batch_size=batch_size,
                workers=workers,
            )

    def eval_single_model(model_name: str) -> Dict[str, Dict[str, float]]:
        evaluator = evaluators[model_name]
        return evaluator.evaluate_multiple(
            task_names=expanded_tasks,
            max_samples=max_samples,
            ks=ks,
        )

    # 带 GPU 负载均衡执行
    results = run_with_gpu_balance(
        models=models,
        gpu_info=gpu_info,
        total_workers=model_workers,
        eval_func=eval_single_model,
        desc="评估",
        verbose=verbose,
    )

    return results
