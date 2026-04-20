#!/usr/bin/env python3
"""
RL Training CLI

命令行工具，用于运行 Doc-level REINFORCE 训练。

用法:
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --output outputs/rl

    # 使用所有文档（n_docs=0）
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --n_docs 0 --max_docs 50

    # 分块前向传播（节省显存）
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --chunk_size 8
"""
import os
import sys
import json
import torch
import logging
import argparse
import random
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType

from .rl_dataset import RLRerankDataset, RLCollator
from .rl_trainer import RLTrainer, load_sft_model

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MODEL = "Qwen/Qwen3-Reranker-4B"
YES_TOKEN = "yes"
NO_TOKEN = "no"


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-Reranker RL 训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 基础训练
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl

    # 使用所有文档（推荐用于大规模数据）
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --n_docs 0 --max_docs 50

    # 分块前向传播（节省显存）
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --chunk_size 8

    # DAPO 训练
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dapo

    # Dr. GRPO 训练
    qwen3-rerank-train-rl --sft_model outputs/sft/final --data train.jsonl --loss_type dr_grpo
        """
    )

    # 模型参数
    model_group = parser.add_argument_group("模型参数")
    model_group.add_argument("--base_model", type=str, default=DEFAULT_MODEL,
                             help="基座模型路径")
    model_group.add_argument("--sft_model", type=str, required=True,
                             help="SFT 模型路径（第一阶段训练结果）")
    model_group.add_argument("--use_base_model", action="store_true",
                             help="直接使用基础模型（跳过 SFT 模型加载）")

    # 数据参数
    data_group = parser.add_argument_group("数据参数")
    data_group.add_argument("--data", type=str, required=True,
                            help="训练数据文件 (JSONL)")
    data_group.add_argument("--val_data", type=str, default=None,
                            help="验证数据文件 (JSONL)")
    data_group.add_argument("--max_samples", type=int, default=0,
                            help="最大训练样本数 (0=不限制)")
    data_group.add_argument("--max_length", type=int, default=4096,
                            help="最大序列长度")
    data_group.add_argument("--filter-overlength", action="store_true",
                            help="过滤超过 max_length 的样本（默认关闭，减少重复 tokenization）")

    # 正负例采样参数
    sampling_group = parser.add_argument_group("正负例采样")
    sampling_group.add_argument("--n_docs", type=int, default=8,
                                help="每组文档数 (0=使用所有文档)")
    sampling_group.add_argument("--n_pos", type=int, default=0,
                                help="固定正例数 (0=按原始比例动态分配)")
    sampling_group.add_argument("--min_pos", type=int, default=1,
                                help="每组最少正例数")
    sampling_group.add_argument("--min_neg", type=int, default=1,
                                help="每组最少负例数")
    sampling_group.add_argument("--max_docs", type=int, default=0,
                                help="单样本最大文档数 (0=不限制，用于 n_docs=0 时避免极端样本 OOM)")

    # RL 训练参数
    rl_group = parser.add_argument_group("RL 训练参数")
    rl_group.add_argument("--kl_coef", type=float, default=0.1,
                          help="KL 惩罚系数")
    rl_group.add_argument("--clip_range", type=float, default=0.2,
                          help="PPO 风格的 ratio clipping")
    rl_group.add_argument("--reward_type", type=str, default="rank_based",
                          choices=["rank_based", "score_based", "ndcg_based", "recall_based"],
                          help="奖励类型")
    rl_group.add_argument("--reward_k", type=int, default=10,
                          help="ndcg_based/recall_based 的阈值 k")
    rl_group.add_argument("--scale_rewards", action="store_true",
                          help="是否标准化奖励")
    rl_group.add_argument("--loss_type", type=str, default="dapo",
                          choices=["grpo", "dapo", "dr_grpo", "dpo"],
                          help="损失类型")
    rl_group.add_argument("--dpo_beta", type=float, default=0.1,
                          help="DPO beta")
    rl_group.add_argument("--dpo_reference_free", action="store_true",
                          help="DPO reference-free（不使用参考模型）")
    rl_group.add_argument("--num_iterations", type=int, default=1,
                          help="每个 batch 的更新次数")
    rl_group.add_argument("--chunk_size", type=int, default=0,
                          help="分块前向传播大小 (0=不分块)")

    # LoRA 参数
    lora_group = parser.add_argument_group("LoRA 参数")
    lora_group.add_argument("--lora_r", type=int, default=8,
                            help="LoRA rank")
    lora_group.add_argument("--lora_alpha", type=int, default=16,
                            help="LoRA alpha")

    # 训练参数
    train_group = parser.add_argument_group("训练参数")
    train_group.add_argument("--output", type=str, default="outputs/rl",
                             help="输出目录")
    train_group.add_argument("--batch_size", type=int, default=2,
                             help="每个设备的 batch size (query 数)")
    train_group.add_argument("--gradient_accumulation_steps", type=int, default=8,
                             help="梯度累积步数")
    train_group.add_argument("--learning_rate", type=float, default=5e-6,
                             help="学习率")
    train_group.add_argument("--epochs", type=int, default=3,
                             help="训练轮数")
    train_group.add_argument("--warmup_ratio", type=float, default=0.1,
                             help="warmup 比例")
    train_group.add_argument("--seed", type=int, default=42,
                             help="随机种子")
    train_group.add_argument("--logging_steps", type=int, default=1,
                             help="日志记录步数")
    train_group.add_argument("--report_to", type=str, default="none",
                             choices=["none", "tensorboard", "wandb", "swanlab"],
                             help="日志报告工具")

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 参数校验
    if args.n_docs > 0 and args.n_pos > 0 and args.n_pos >= args.n_docs:
        logger.error(f"--n_pos ({args.n_pos}) 必须小于 --n_docs ({args.n_docs})")
        sys.exit(1)
    if args.n_docs == 0 and args.n_pos > 0:
        logger.error("--n_docs=0（使用所有文档）时不能指定 --n_pos")
        sys.exit(1)

    # 设置随机种子
    set_seed(args.seed)

    # 解析路径
    output_dir = Path(args.output)
    sft_model_path = Path(args.sft_model)
    train_file = Path(args.data)
    val_file = Path(args.val_data) if args.val_data else None

    if not train_file.exists():
        logger.error(f"训练文件不存在: {train_file}")
        sys.exit(1)

    if not args.use_base_model and not sft_model_path.exists():
        logger.error(f"SFT 模型不存在: {sft_model_path}")
        logger.error("请先运行 SFT 训练，或使用 --use_base_model 直接测试")
        sys.exit(1)

    # 打印配置
    logger.info("=" * 60)
    logger.info("RL 训练配置")
    logger.info("=" * 60)
    if args.use_base_model:
        logger.info(f"使用基础模型: {args.base_model}")
    else:
        logger.info(f"SFT 模型: {sft_model_path}")
    logger.info(f"训练数据: {train_file}")
    if val_file:
        logger.info(f"验证数据: {val_file}")
    logger.info(f"filter_overlength: {args.filter_overlength}")
    logger.info(f"最大样本数: {args.max_samples if args.max_samples > 0 else '不限制'}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"KL 系数: {args.kl_coef}")
    logger.info(f"Clip 范围: {args.clip_range}")
    logger.info(f"奖励类型: {args.reward_type}")
    logger.info(f"损失类型: {args.loss_type}")
    logger.info(f"num_iterations: {args.num_iterations}")

    if args.chunk_size > 0:
        logger.info(f"分块前向传播: chunk_size={args.chunk_size}")
    else:
        logger.info(f"分块前向传播: 关闭")

    if args.n_docs == 0:
        max_docs_str = f", max_docs={args.max_docs}" if args.max_docs > 0 else ""
        logger.info(f"正负例采样: 使用所有文档 (min_pos={args.min_pos}, min_neg={args.min_neg}{max_docs_str})")
    elif args.n_pos > 0:
        logger.info(f"正负例采样: 固定 {args.n_pos} 正 {args.n_docs - args.n_pos} 负")
    else:
        logger.info(f"正负例采样: 动态比例 (min_pos={args.min_pos}, min_neg={args.min_neg})")

    if args.loss_type == "dpo":
        logger.info(f"DPO beta: {args.dpo_beta}, reference_free: {args.dpo_reference_free}")
    logger.info("=" * 60)

    # 加载 tokenizer（left padding 确保 logits[:, -1, :] 取到正确位置）
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 获取 yes/no token ID
    yes_token_id = tokenizer.convert_tokens_to_ids(YES_TOKEN)
    no_token_id = tokenizer.convert_tokens_to_ids(NO_TOKEN)
    logger.info(f"yes token ID: {yes_token_id}, no token ID: {no_token_id}")

    # 加载模型
    if args.use_base_model:
        logger.info(f"加载基础模型: {args.base_model}")
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
    else:
        model = load_sft_model(str(sft_model_path), args.base_model)

    # 添加新的 LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    # 准备数据集
    train_dataset = RLRerankDataset(
        str(train_file),
        tokenizer=tokenizer,
        max_length=args.max_length,
        n_docs=args.n_docs,
        n_pos=args.n_pos,
        min_pos=args.min_pos,
        min_neg=args.min_neg,
        max_docs=args.max_docs,
        seed=args.seed,
        max_samples=args.max_samples,
        filter_overlength=args.filter_overlength,
    )

    val_dataset = None
    if val_file and val_file.exists():
        val_dataset = RLRerankDataset(
            str(val_file),
            tokenizer=tokenizer,
            max_length=args.max_length,
            n_docs=args.n_docs,
            n_pos=args.n_pos,
            min_pos=args.min_pos,
            min_neg=args.min_neg,
            max_docs=args.max_docs,
            seed=args.seed,
            max_samples=0,
            filter_overlength=args.filter_overlength,
        )

    collator = RLCollator(tokenizer, max_length=args.max_length)

    # 训练参数
    run_name = output_dir.name
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        logging_steps=args.logging_steps,
        eval_strategy="epoch" if val_dataset else "no",
        save_strategy="epoch",
        save_total_limit=None,
        load_best_model_at_end=val_dataset is not None,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        report_to=args.report_to if args.report_to != "none" else None,
        seed=args.seed,
    )

    # 创建训练器
    trainer = RLTrainer(
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        kl_coef=args.kl_coef,
        beta=args.dpo_beta,
        reference_free=args.dpo_reference_free,
        reward_type=args.reward_type,
        reward_k=args.reward_k,
        clip_range=args.clip_range,
        scale_rewards=args.scale_rewards,
        loss_type=args.loss_type,
        num_iterations=args.num_iterations,
        chunk_size=args.chunk_size,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # 开始训练
    logger.info("开始 RL 训练...")
    trainer.train()

    # 保存最终模型
    final_output_dir = output_dir / "final"
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 保存 LoRA adapter
    lora_output_dir = final_output_dir / "lora_adapter"
    lora_output_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(lora_output_dir))
    tokenizer.save_pretrained(str(lora_output_dir))
    logger.info(f"LoRA adapter 已保存: {lora_output_dir}")

    # 2. Merge LoRA 并保存完整模型
    logger.info("Merging LoRA adapter with base model...")
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(str(final_output_dir), torch_dtype=torch.bfloat16)
    tokenizer.save_pretrained(str(final_output_dir))
    logger.info(f"Merged 模型已保存: {final_output_dir}")

    # 保存训练配置
    config = vars(args)
    config["yes_token_id"] = yes_token_id
    config["no_token_id"] = no_token_id
    with open(final_output_dir / "training_config.json", 'w') as f:
        json.dump(config, f, indent=2, default=str)

    logger.info("RL 训练完成!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
