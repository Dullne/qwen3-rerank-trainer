"""
评估报告生成模块

提供评估结果的汇总、对比和报告生成功能：
- JSON 格式保存
- Markdown 报告生成
- 多模型对比表格

使用示例:
    from qwen3_rerank_trainer.evaluation import generate_report, print_results_summary

    # 打印结果摘要
    print_results_summary(results)

    # 生成完整报告
    generate_report(results, output_dir, config)
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================================================
# 报告配置
# ============================================================================

@dataclass
class ReportConfig:
    """报告生成配置"""
    title: str = "Rerank 模型评估报告"
    baseline_model: Optional[str] = None  # 基准模型名称
    metrics: List[str] = field(default_factory=lambda: [
        "NDCG@1", "NDCG@5", "NDCG@10",
        "MRR", "AP",
        "P@1", "P@5", "P@10",
        "R@1", "R@5", "R@10",
    ])
    primary_metric: str = "NDCG@10"  # 主要对比指标
    timezone_offset: int = 8  # 时区偏移（小时）


# ============================================================================
# 结果打印
# ============================================================================

def print_results_summary(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = None,
) -> None:
    """
    打印评估结果摘要

    Args:
        results: {model_name: {metric: value}} 或 {task_name: {metric: value}}
        metrics: 要显示的指标列表
    """
    if metrics is None:
        metrics = ["NDCG@10", "MRR", "AP", "R@10"]

    print("\n" + "=" * 80)
    print("评估结果摘要")
    print("=" * 80)

    # 表头
    header = f"{'名称':<20}"
    for metric in metrics:
        header += f" {metric:>10}"
    print(header)
    print("-" * len(header))

    # 结果行
    for name, task_results in results.items():
        if isinstance(task_results, dict) and "error" not in task_results:
            row = f"{name:<20}"
            for metric in metrics:
                value = task_results.get(metric, task_results.get(metric.lower(), None))
                if value is not None:
                    row += f" {value*100:>9.2f}%"
                else:
                    row += f" {'N/A':>10}"
            print(row)
        elif isinstance(task_results, dict) and "error" in task_results:
            print(f"{name:<20} Error: {task_results['error']}")

    print("=" * 80)


def print_comparison_table(
    all_results: List[Dict[str, Any]],
    metric: str = "NDCG@10",
) -> None:
    """
    打印多模型对比表格

    Args:
        all_results: 模型评估结果列表
            每个元素为 {"model": name, "results": {task: {metric: value}}}
        metric: 对比指标
    """
    print("\n" + "=" * 100)
    print(f"模型对比 ({metric})")
    print("=" * 100)

    # 收集所有任务
    all_tasks = set()
    for item in all_results:
        if "results" in item:
            all_tasks.update(item["results"].keys())
    all_tasks = sorted(all_tasks)

    # 表头
    header = f"{'模型':<15}"
    for task in all_tasks:
        short_name = task.replace("Reranking", "").replace("Retrieval", "")[:12]
        header += f" {short_name:>12}"
    header += f" {'平均':>12}"
    print(header)
    print("-" * len(header))

    # 结果行
    for item in all_results:
        model = item.get("model", item.get("profile", "unknown"))
        row = f"{model:<15}"
        scores = []

        for task in all_tasks:
            task_results = item.get("results", {}).get(task, {})
            if isinstance(task_results, dict) and "error" not in task_results:
                value = task_results.get(metric, 0)
                row += f" {value*100:>11.2f}%"
                scores.append(value)
            else:
                row += f" {'N/A':>12}"

        # 平均分
        if scores:
            avg = sum(scores) / len(scores)
            row += f" {avg*100:>11.2f}%"
        else:
            row += f" {'N/A':>12}"

        print(row)

    print("=" * 100)


# ============================================================================
# 报告生成
# ============================================================================

def generate_report(
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    output_dir: Union[str, Path],
    config: Optional[ReportConfig] = None,
    model_info: Optional[Dict[str, str]] = None,
) -> Dict[str, Path]:
    """
    生成评估报告

    Args:
        results: 评估结果
            - 单模型：{task_name: {metric: value}}
            - 多模型：[{"model": name, "results": {...}}, ...]
        output_dir: 输出目录
        config: 报告配置
        model_info: 模型信息 {model_name: description}

    Returns:
        生成的文件路径 {"json": path, "markdown": path}
    """
    if config is None:
        config = ReportConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 生成时间戳
    timestamp = (datetime.now() + timedelta(hours=config.timezone_offset)).strftime('%Y%m%d_%H%M%S')

    # 确定是单模型还是多模型
    is_multi_model = isinstance(results, list)

    if is_multi_model:
        return _generate_multi_model_report(
            results, output_dir, config, model_info, timestamp
        )
    else:
        return _generate_single_model_report(
            results, output_dir, config, model_info, timestamp
        )


def _generate_single_model_report(
    results: Dict[str, Dict[str, float]],
    output_dir: Path,
    config: ReportConfig,
    model_info: Optional[Dict[str, str]],
    timestamp: str,
) -> Dict[str, Path]:
    """生成单模型报告"""
    # JSON
    json_path = output_dir / f"results_{timestamp}.json"
    json_data = {
        "timestamp": timestamp,
        "results": results,
    }
    if model_info:
        json_data["model_info"] = model_info

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # Markdown
    md_path = output_dir / f"report_{timestamp}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {config.title}\n\n")
        f.write(f"**评估时间**: {timestamp}\n\n")

        # 结果表格
        f.write("## 评估结果\n\n")
        f.write("| 数据集 | " + " | ".join(config.metrics) + " |\n")
        f.write("|:-------|" + "|".join([":------:" for _ in config.metrics]) + "|\n")

        for task, task_results in sorted(results.items()):
            if isinstance(task_results, dict) and "error" not in task_results:
                row = [task]
                for metric in config.metrics:
                    value = task_results.get(metric)
                    if value is not None:
                        row.append(f"{value*100:.2f}")
                    else:
                        row.append("-")
                f.write("| " + " | ".join(row) + " |\n")

        f.write("\n")

    logger.info(f"Report saved to: {output_dir}")
    return {"json": json_path, "markdown": md_path}


def _generate_multi_model_report(
    all_results: List[Dict[str, Any]],
    output_dir: Path,
    config: ReportConfig,
    model_info: Optional[Dict[str, str]],
    timestamp: str,
) -> Dict[str, Path]:
    """生成多模型对比报告"""
    # 区分成功和失败的结果
    successful_results = []
    failed_models = []

    for item in all_results:
        model_name = item.get("model", item.get("profile", "unknown"))
        results = item.get("results", {})

        # 检查是否有有效结果
        has_valid = False
        if isinstance(results, dict):
            for task_result in results.values():
                if isinstance(task_result, dict) and "error" not in task_result:
                    if task_result.get("num_evaluated", 1) > 0:
                        has_valid = True
                        break

        if has_valid:
            successful_results.append(item)
        else:
            failed_models.append(model_name)

    # 收集所有任务（只从成功结果中）
    all_tasks = set()
    model_names = []
    for item in successful_results:
        model_name = item.get("model", item.get("profile", "unknown"))
        model_names.append(model_name)
        if "results" in item:
            for task, task_result in item["results"].items():
                if isinstance(task_result, dict) and "error" not in task_result:
                    all_tasks.add(task)
    all_tasks = sorted(all_tasks)

    # 构建对比数据
    comparison = {
        "metadata": {
            "models": model_names,
            "failed_models": failed_models,
            "tasks": list(all_tasks),
            "timestamp": timestamp,
            "baseline": config.baseline_model,
        },
        "results": {},
    }

    for item in successful_results:
        model = item.get("model", item.get("profile", "unknown"))
        comparison["results"][model] = {}

        for task in all_tasks:
            task_results = item.get("results", {}).get(task, {})
            if isinstance(task_results, dict) and "error" not in task_results:
                comparison["results"][model][task] = {
                    metric: task_results.get(metric, 0)
                    for metric in config.metrics
                    if task_results.get(metric) is not None
                }

    # JSON
    json_path = output_dir / "comparison.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)

    # Markdown
    md_path = output_dir / "report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# {config.title}\n\n")

        # 元信息
        f.write("| 项目 | 信息 |\n")
        f.write("|------|------|\n")
        f.write(f"| 评估时间 | {timestamp} |\n")
        f.write(f"| 成功模型 | {', '.join(model_names)} ({len(model_names)}个) |\n")
        if failed_models:
            f.write(f"| 失败模型 | {', '.join(failed_models)} ({len(failed_models)}个) |\n")
        f.write(f"| 数据集数量 | {len(all_tasks)} 个 |\n")
        if config.baseline_model:
            f.write(f"| 基准模型 | {config.baseline_model} |\n")
        f.write("\n")

        # 数据集列表
        f.write("## 评估数据集\n\n")
        f.write("| 序号 | 数据集名称 |\n")
        f.write("|:----:|------------|\n")
        for i, task in enumerate(all_tasks, 1):
            f.write(f"| {i} | {task} |\n")
        f.write("\n")

        # 评估结果汇总
        f.write("## 评估结果\n\n")

        # 基准模型数据
        baseline_data = None
        if config.baseline_model and config.baseline_model in comparison["results"]:
            baseline_data = comparison["results"][config.baseline_model]
            f.write(f"**基准模型**: {config.baseline_model}，其他模型显示相对提升\n\n")

        # 表头
        f.write("| 数据集 | 模型 | " + " | ".join(config.metrics) + " |\n")
        f.write("|:-------|:-----|" + "|".join([":------:" for _ in config.metrics]) + "|\n")

        for task in all_tasks:
            short_name = task.replace("Reranking", "").replace("Retrieval", "").replace("-", "")
            baseline_scores = baseline_data.get(task, {}) if baseline_data else {}

            for model in model_names:
                task_scores = comparison["results"].get(model, {}).get(task, {})
                if task_scores:
                    is_baseline = (model == config.baseline_model)
                    row = [short_name, model]

                    for metric in config.metrics:
                        value = task_scores.get(metric, 0)
                        baseline_val = baseline_scores.get(metric, 0)
                        row.append(_format_metric_value(value, baseline_val, is_baseline))

                    f.write("| " + " | ".join(row) + " |\n")
        f.write("\n")

        # 总结
        f.write("## 总结\n\n")

        # 计算每个模型的平均分和各数据集最佳
        model_avg_scores = {}
        dataset_best = {}

        for model in model_names:
            scores = []
            for task in all_tasks:
                task_scores = comparison["results"].get(model, {}).get(task, {})
                primary_score = task_scores.get(config.primary_metric, 0)

                if primary_score > 0:
                    scores.append(primary_score)

                    # 记录每个数据集的最佳
                    if task not in dataset_best or primary_score > dataset_best[task][1]:
                        dataset_best[task] = (model, primary_score)

            if scores:
                model_avg_scores[model] = sum(scores) / len(scores)

        # 排名
        sorted_models = sorted(model_avg_scores.items(), key=lambda x: x[1], reverse=True)

        f.write(f"### 综合排名（按平均 {config.primary_metric}）\n\n")
        f.write("| 排名 | 模型 | 平均分 | 相对基准 |\n")
        f.write("|:----:|:-----|-------:|----------:|\n")

        baseline_avg = model_avg_scores.get(config.baseline_model, 0) if config.baseline_model else 0
        for rank, (model, avg_score) in enumerate(sorted_models, 1):
            if baseline_avg > 0:
                diff = (avg_score - baseline_avg) * 100
                diff_str = f"+{diff:.2f}%" if diff >= 0 else f"{diff:.2f}%"
            else:
                diff_str = "-"
            f.write(f"| {rank} | {model} | {avg_score*100:.2f}% | {diff_str} |\n")
        f.write("\n")

        # 各数据集最佳模型
        if len(all_tasks) > 1:
            f.write(f"### 各数据集最佳模型\n\n")
            f.write(f"| 数据集 | 最佳模型 | {config.primary_metric} |\n")
            f.write("|:-------|:---------|--------:|\n")
            for task in all_tasks:
                short_name = task.replace("Reranking", "").replace("Retrieval", "").replace("-", "")
                best_model, best_score = dataset_best.get(task, ("-", 0))
                f.write(f"| {short_name} | {best_model} | {best_score*100:.2f}% |\n")
            f.write("\n")

        # 关键发现
        f.write("### 关键发现\n\n")

        if sorted_models:
            best_model, best_avg = sorted_models[0]
            f.write(f"1. **最佳模型**: {best_model}，平均 {config.primary_metric} 为 {best_avg*100:.2f}%\n")

            # 相对基准提升
            if baseline_avg > 0 and best_model != config.baseline_model:
                improvement = (best_avg - baseline_avg) / baseline_avg * 100
                f.write(f"2. **相对基准提升**: {improvement:+.2f}%\n")

            # 与基准对比的模型数量
            if baseline_avg > 0:
                better_count = sum(1 for _, s in sorted_models if s > baseline_avg)
                worse_count = sum(1 for _, s in sorted_models if s < baseline_avg)
                f.write(f"3. **优于基准**: {better_count} 个模型，**弱于基准**: {worse_count} 个模型\n")

            # 最佳/最差数据集
            if len(all_tasks) > 1 and best_model in comparison["results"]:
                task_scores_list = [
                    (t, comparison["results"][best_model].get(t, {}).get(config.primary_metric, 0))
                    for t in all_tasks
                ]
                task_scores_list.sort(key=lambda x: x[1], reverse=True)
                best_task = task_scores_list[0][0].replace("Reranking", "").replace("-", "")
                worst_task = task_scores_list[-1][0].replace("Reranking", "").replace("-", "")
                f.write(f"4. **{best_model}** 在 {best_task} 表现最好，在 {worst_task} 表现相对较弱\n")

        # 失败模型说明
        if failed_models:
            f.write(f"\n**注意**: 以下模型评估失败，未包含在对比中: {', '.join(failed_models)}\n")

        f.write("\n")

    logger.info(f"Comparison report saved to: {output_dir}")
    return {"json": json_path, "markdown": md_path}


def _format_metric_value(
    value: float,
    baseline_value: float,
    is_baseline: bool,
) -> str:
    """格式化指标值"""
    pct = value * 100
    if is_baseline or baseline_value == 0:
        return f"{pct:.1f}"
    else:
        baseline_pct = baseline_value * 100
        diff = pct - baseline_pct
        if diff > 0:
            return f"{pct:.1f}(+{diff:.1f})"
        elif diff < 0:
            return f"{pct:.1f}({diff:.1f})"
        else:
            return f"{pct:.1f}"


# ============================================================================
# 结果保存
# ============================================================================

def save_results_json(
    results: Dict[str, Any],
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    保存结果为 JSON 格式

    Args:
        results: 评估结果
        output_path: 输出文件路径
        metadata: 附加元数据

    Returns:
        保存的文件路径
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }
    if metadata:
        data["metadata"] = metadata

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return output_path


def load_results_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    加载 JSON 格式的结果

    Args:
        file_path: JSON 文件路径

    Returns:
        结果字典
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
