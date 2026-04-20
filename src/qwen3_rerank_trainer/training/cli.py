#!/usr/bin/env python3
"""
Qwen3-Reranker SFT 训练命令行工具

使用示例:
    # 基本训练
    qwen3-rerank-train --model /path/to/model --data train.jsonl --output outputs/sft

    # 使用 LoRA
    qwen3-rerank-train --model /path/to/model --data train.jsonl --output outputs/sft \
        --lora --lora-r 8 --lora-alpha 16

    # 自定义参数
    qwen3-rerank-train --model /path/to/model --data train.jsonl --output outputs/sft \
        --n-docs 8 --n-pos 1 --batch-size 4 --learning-rate 1e-5
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='Qwen3-Reranker SFT 训练工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本训练
  %(prog)s --model /path/to/Qwen3-Reranker-4B --data train.jsonl --output outputs/sft

  # 使用 LoRA 微调
  %(prog)s --model /path/to/model --data train.jsonl --output outputs/sft \\
           --lora --lora-r 8 --lora-alpha 16

  # 完整参数示例
  %(prog)s --model /path/to/model --data train.jsonl --output outputs/sft \\
           --n-docs 8 --n-pos 1 --batch-size 4 --gradient-accumulation 8 \\
           --learning-rate 1e-5 --epochs 3 --chunk-size 16
        """
    )

    # 模型和数据
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='模型路径或 HuggingFace 模型名称'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='训练数据文件路径 (jsonl 格式)'
    )
    parser.add_argument(
        '--val-data',
        type=str,
        default=None,
        help='验证数据文件路径 (可选)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='outputs/sft',
        help='输出目录 (默认: outputs/sft)'
    )

    # 数据采样
    parser.add_argument(
        '--n-docs',
        type=int,
        default=8,
        help='每组文档数 (默认: 8)'
    )
    parser.add_argument(
        '--n-pos',
        type=int,
        default=1,
        help='固定正例数，0 表示动态分配 (默认: 1)'
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=4096,
        help='最大序列长度 (默认: 4096)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=0,
        help='最大样本数，0 表示不限制 (默认: 0)'
    )
    parser.add_argument(
        '--filter-overlength',
        action='store_true',
        help='过滤超过 max_length 的样本（默认关闭，减少重复 tokenization）'
    )

    # 训练参数
    parser.add_argument(
        '--batch-size',
        type=int,
        default=4,
        help='批次大小 (默认: 4)'
    )
    parser.add_argument(
        '--gradient-accumulation',
        type=int,
        default=8,
        help='梯度累积步数 (默认: 8)'
    )
    parser.add_argument(
        '--learning-rate', '--lr',
        type=float,
        default=1e-5,
        help='学习率 (默认: 1e-5)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='训练轮数 (默认: 3)'
    )
    parser.add_argument(
        '--warmup-ratio',
        type=float,
        default=0.1,
        help='Warmup 比例 (默认: 0.1)'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=0,
        help='分块处理大小，0 表示不分块 (默认: 0)'
    )

    # 损失函数参数
    parser.add_argument(
        '--loss-type',
        type=str,
        default='bce',
        choices=['bce', 'infonce', 'list_mle', 'lambda_loss', 'ranknet'],
        help='损失函数类型: bce(默认), infonce, list_mle, lambda_loss, ranknet'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.05,
        help='InfoNCE 温度参数 (默认: 0.05)'
    )
    parser.add_argument(
        '--infonce-mode',
        type=str,
        default='single',
        choices=['single', 'posset', 'avgpos'],
        help='InfoNCE 正例策略 (默认: single)'
    )
    parser.add_argument(
        '--lambda-metric',
        type=str,
        default='ndcg',
        choices=['ndcg', 'map', 'mrr'],
        help='LambdaLoss 目标指标 (默认: ndcg)'
    )
    parser.add_argument(
        '--ranknet-max-pairs',
        type=int,
        default=2000000,
        help='RankNet 最大 pair 数，超出后分块计算 (默认: 2000000)'
    )

    # LoRA 参数
    parser.add_argument(
        '--lora',
        action='store_true',
        help='使用 LoRA 微调'
    )
    parser.add_argument(
        '--lora-r',
        type=int,
        default=8,
        help='LoRA rank (默认: 8)'
    )
    parser.add_argument(
        '--lora-alpha',
        type=int,
        default=16,
        help='LoRA alpha (默认: 16)'
    )
    parser.add_argument(
        '--lora-dropout',
        type=float,
        default=0.05,
        help='LoRA dropout (默认: 0.05)'
    )

    # 其他参数
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子 (默认: 42)'
    )
    parser.add_argument(
        '--logging-steps',
        type=int,
        default=10,
        help='日志记录步数 (默认: 10)'
    )
    parser.add_argument(
        '--save-steps',
        type=int,
        default=500,
        help='保存检查点步数 (默认: 500)'
    )
    parser.add_argument(
        '--bf16',
        action='store_true',
        help='使用 bf16 训练'
    )
    parser.add_argument(
        '--fp16',
        action='store_true',
        help='使用 fp16 训练'
    )
    parser.add_argument(
        '--report-to',
        type=str,
        default='none',
        choices=['none', 'wandb', 'tensorboard'],
        help='报告工具 (默认: none)'
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查依赖
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    except ImportError:
        logger.error("请安装 transformers: pip install transformers>=4.40.0")
        return 1

    from ..training import RerankDataset, RerankCollator, ContrastiveSFTTrainer
    from ..training.sft_trainer import get_yes_no_token_ids

    # 设置随机种子
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # 检查数据文件
    if not os.path.exists(args.data):
        logger.error(f"数据文件不存在: {args.data}")
        return 1

    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Qwen3-Reranker SFT 训练")
    logger.info("=" * 60)
    logger.info(f"模型: {args.model}")
    logger.info(f"数据: {args.data}")
    logger.info(f"输出: {args.output}")
    logger.info(f"n_docs: {args.n_docs}, n_pos: {args.n_pos}")
    logger.info(f"filter_overlength: {args.filter_overlength}")
    logger.info(f"batch_size: {args.batch_size}, gradient_accumulation: {args.gradient_accumulation}")
    logger.info(f"learning_rate: {args.learning_rate}, epochs: {args.epochs}")
    logger.info(f"损失函数: {args.loss_type}")
    if args.loss_type == 'infonce':
        logger.info(f"  temperature: {args.temperature}")
        logger.info(f"  mode: {args.infonce_mode}")
    elif args.loss_type == 'lambda_loss':
        logger.info(f"  目标指标: {args.lambda_metric}")
    elif args.loss_type == 'ranknet':
        logger.info(f"  max_pairs_per_batch: {args.ranknet_max_pairs}")
    if args.lora:
        logger.info(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    logger.info("=" * 60)

    # 加载 tokenizer
    logger.info("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side='left',
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    logger.info("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32),
        device_map="auto",
    )

    # 应用 LoRA
    if args.lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            logger.error("请安装 peft: pip install peft>=0.10.0")
            return 1

        logger.info("应用 LoRA...")
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # 获取 yes/no token IDs
    yes_id, no_id = get_yes_no_token_ids(tokenizer)
    logger.info(f"yes_token_id: {yes_id}, no_token_id: {no_id}")

    # 创建数据集
    logger.info("创建数据集...")
    train_dataset = RerankDataset(
        args.data,
        tokenizer=tokenizer,
        n_docs=args.n_docs,
        n_pos=args.n_pos,
        max_length=args.max_length,
        max_samples=args.max_samples,  # 0 表示不限制
        seed=args.seed,
        filter_overlength=args.filter_overlength,
    )
    logger.info(f"训练样本数: {len(train_dataset)}")

    val_dataset = None
    if args.val_data and os.path.exists(args.val_data):
        val_dataset = RerankDataset(
            args.val_data,
            tokenizer=tokenizer,
            n_docs=args.n_docs,
            n_pos=args.n_pos,
            max_length=args.max_length,
            seed=args.seed,
            filter_overlength=args.filter_overlength,
        )
        logger.info(f"验证样本数: {len(val_dataset)}")

    # 创建 collator
    collator = RerankCollator(tokenizer, max_length=args.max_length)

    # 训练参数
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=args.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=args.seed,
    )

    # 创建 Trainer
    trainer = ContrastiveSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        yes_token_id=yes_id,
        no_token_id=no_id,
        chunk_size=args.chunk_size if args.chunk_size > 0 else 0,
        loss_type=args.loss_type,
        temperature=args.temperature,
        infonce_mode=args.infonce_mode,
        lambda_metric=args.lambda_metric,
        ranknet_max_pairs_per_batch=args.ranknet_max_pairs,
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()

    # 保存模型
    logger.info(f"保存模型到: {output_dir}")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    logger.info("训练完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
