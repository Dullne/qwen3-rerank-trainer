"""
RL Trainer for Reranker

提供 Doc-level REINFORCE 训练器：
- RLTrainer: 继承自 transformers.Trainer，实现 GRPO/DAPO/Dr.GRPO 损失
- load_sft_model: 加载 SFT 模型并合并 LoRA
"""
import torch
import logging
from pathlib import Path
from typing import Tuple, Optional
from transformers import Trainer, AutoModelForCausalLM

logger = logging.getLogger(__name__)


class RLTrainer(Trainer):
    """RL 训练器（Doc-level REINFORCE）

    支持分块前向传播，允许使用很大的 n_docs 而不会 OOM。
    显存占用与 chunk_size 相关，与 n_docs 无关。

    Args:
        yes_token_id: "yes" token 的 ID
        no_token_id: "no" token 的 ID
        kl_coef: KL 惩罚系数（防止策略偏离参考模型）
        reward_type: 奖励类型
            - "rank_based": ERANK 风格
            - "score_based": 基于分数
            - "ndcg_based": NDCG 奖励
            - "recall_based": Recall 奖励
        reward_k: ndcg_based/recall_based 的阈值 k
        clip_range: PPO 风格的 ratio clipping
        scale_rewards: 是否标准化奖励
        loss_type: 损失类型 ("grpo", "dapo", "dr_grpo")
        num_iterations: 每个 batch 的更新次数（论文中的 μ）
        chunk_size: 分块大小（0 表示不分块）
        *args, **kwargs: 传递给 Trainer 的其他参数

    Example:
        >>> trainer = RLTrainer(
        ...     yes_token_id=tokenizer.convert_tokens_to_ids("yes"),
        ...     no_token_id=tokenizer.convert_tokens_to_ids("no"),
        ...     kl_coef=0.1,
        ...     reward_type="rank_based",
        ...     loss_type="dapo",
        ...     chunk_size=8,  # 分块处理，节省显存
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     data_collator=collator,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        yes_token_id: int,
        no_token_id: int,
        kl_coef: float = 0.1,
        reward_type: str = "rank_based",
        reward_k: int = 10,
        clip_range: float = 0.2,
        scale_rewards: bool = False,
        loss_type: str = "dapo",
        num_iterations: int = 1,
        chunk_size: int = 0,
        beta: float = 0.1,
        reference_free: bool = False,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.kl_coef = kl_coef
        self.reward_type = reward_type
        self.reward_k = reward_k
        self.clip_range = clip_range
        self.scale_rewards = scale_rewards
        self.loss_type = loss_type.lower()
        self.num_iterations = num_iterations
        self.chunk_size = chunk_size
        self.beta = beta
        self.reference_free = reference_free

        # 延迟导入损失函数
        from ..rl import reinforce_loss, dpo_loss
        self._reinforce_loss = reinforce_loss
        self._dpo_loss = dpo_loss

    def _requires_reference(self) -> bool:
        return self.kl_coef > 0 or (self.loss_type == "dpo" and not self.reference_free)

    def _get_logits(self, model, input_ids, attention_mask) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取 yes/no logits"""
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 获取最后一个 token 的 logits（使用 left padding，所以直接取 -1 位置）
        last_token_logits = logits[:, -1, :]

        yes_logits = last_token_logits[:, self.yes_token_id]
        no_logits = last_token_logits[:, self.no_token_id]

        return yes_logits, no_logits

    def _get_group_logits_chunked(
        self,
        model,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        group_size: int,
        offset: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """分块获取一个 query 组的所有 logits

        Args:
            model: 模型
            input_ids: 完整 batch 的 input_ids
            attention_mask: 完整 batch 的 attention_mask
            group_size: 当前组的文档数
            offset: 当前组在 batch 中的起始位置

        Returns:
            (yes_logits, no_logits, ref_yes_logits, ref_no_logits)
            每个 tensor 的形状为 [group_size]
        """
        group_yes_logits = []
        group_no_logits = []
        group_ref_yes = []
        group_ref_no = []

        # 参考模型前向（无梯度）仅在支持 adapter 切换时启用
        base_model = model.module if hasattr(model, 'module') else model
        need_ref = self._requires_reference()
        has_adapter_toggle = (
            need_ref
            and hasattr(base_model, "disable_adapter_layers")
            and hasattr(base_model, "enable_adapter_layers")
        )
        # 分块处理当前组
        for chunk_start in range(0, group_size, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, group_size)

            # 提取当前 chunk 的数据
            chunk_ids = input_ids[offset + chunk_start:offset + chunk_end]
            chunk_mask = attention_mask[offset + chunk_start:offset + chunk_end]

            # 策略模型前向（需要梯度）
            yes_logits, no_logits = self._get_logits(model, chunk_ids, chunk_mask)
            group_yes_logits.append(yes_logits)
            group_no_logits.append(no_logits)

            # 参考模型前向（无梯度）
            if has_adapter_toggle:
                with torch.no_grad():
                    base_model.disable_adapter_layers()
                    ref_yes, ref_no = self._get_logits(model, chunk_ids, chunk_mask)
                    base_model.enable_adapter_layers()
                    group_ref_yes.append(ref_yes)
                    group_ref_no.append(ref_no)

        # 拼接当前组的所有 logits
        yes_logits = torch.cat(group_yes_logits, dim=0)
        no_logits = torch.cat(group_no_logits, dim=0)
        ref_yes = torch.cat(group_ref_yes, dim=0) if group_ref_yes else None
        ref_no = torch.cat(group_ref_no, dim=0) if group_ref_no else None

        return yes_logits, no_logits, ref_yes, ref_no

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        group_sizes = inputs["group_sizes"]

        # 判断是否使用分块模式
        use_chunked = self.chunk_size > 0
        need_ref = self._requires_reference()

        if not use_chunked:
            # 原有逻辑：一次性前向传播所有样本
            yes_logits, no_logits = self._get_logits(model, input_ids, attention_mask)

            # 参考模型前向传播（关闭 adapter）
            ref_yes_logits, ref_no_logits = None, None
            if need_ref:
                base_model = model.module if hasattr(model, 'module') else model
                if hasattr(base_model, "disable_adapter_layers") and hasattr(base_model, "enable_adapter_layers"):
                    with torch.no_grad():
                        base_model.disable_adapter_layers()
                        ref_yes_logits, ref_no_logits = self._get_logits(model, input_ids, attention_mask)
                        base_model.enable_adapter_layers()
                elif self.loss_type == "dpo" and not self.reference_free:
                    raise ValueError(
                        "DPO requires reference logits but adapter layers are not available. "
                        "Enable LoRA adapter or set reference_free=True."
                    )

        # 按组计算损失
        total_loss = 0.0
        total_kl = 0.0
        all_rewards = []
        all_advantages = []
        all_pos_scores = []
        all_neg_scores = []
        used_groups = 0
        offset = 0

        for group_size in group_sizes:
            if use_chunked:
                # 分块模式：逐组分块前向传播
                group_yes, group_no, group_ref_yes, group_ref_no = self._get_group_logits_chunked(
                    model, input_ids, attention_mask, group_size, offset
                )
            else:
                # 原有模式：从完整 logits 中提取当前组
                group_yes = yes_logits[offset:offset + group_size]
                group_no = no_logits[offset:offset + group_size]
                group_ref_yes = ref_yes_logits[offset:offset + group_size] if ref_yes_logits is not None else None
                group_ref_no = ref_no_logits[offset:offset + group_size] if ref_no_logits is not None else None

            group_labels = labels[offset:offset + group_size]

            if self.loss_type == "dpo":
                pos_mask = group_labels > 0.5
                neg_mask = ~pos_mask
                if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                    offset += group_size
                    continue

                if not self.reference_free and (group_ref_yes is None or group_ref_no is None):
                    raise ValueError(
                        "DPO requires reference logits but none were provided. "
                        "Enable LoRA adapter or set reference_free=True."
                    )

                loss, pos_score, neg_score = self._dpo_loss(
                    pos_yes_logits=group_yes[pos_mask],
                    pos_no_logits=group_no[pos_mask],
                    neg_yes_logits=group_yes[neg_mask],
                    neg_no_logits=group_no[neg_mask],
                    beta=self.beta,
                    ref_pos_yes_logits=group_ref_yes[pos_mask] if group_ref_yes is not None else None,
                    ref_pos_no_logits=group_ref_no[pos_mask] if group_ref_no is not None else None,
                    ref_neg_yes_logits=group_ref_yes[neg_mask] if group_ref_yes is not None else None,
                    ref_neg_no_logits=group_ref_no[neg_mask] if group_ref_no is not None else None,
                    reference_free=self.reference_free,
                )
                total_loss += loss
                all_pos_scores.append(pos_score)
                all_neg_scores.append(neg_score)
                used_groups += 1
            else:
                # 计算 REINFORCE 损失
                loss, advantages, rewards, kl = self._reinforce_loss(
                    yes_logits=group_yes,
                    no_logits=group_no,
                    labels=group_labels,
                    ref_yes_logits=group_ref_yes,
                    ref_no_logits=group_ref_no,
                    kl_coef=self.kl_coef,
                    reward_type=self.reward_type,
                    reward_k=self.reward_k,
                    clip_range=self.clip_range,
                    scale_rewards=self.scale_rewards,
                    loss_type=self.loss_type,
                )

                total_loss += loss
                total_kl += kl
                all_rewards.append(rewards)
                all_advantages.append(advantages)
                used_groups += 1

            offset += group_size

        if used_groups == 0:
            raise ValueError("No valid groups for loss computation (check labels).")

        # 平均损失
        avg_loss = total_loss / used_groups

        # DPO 分支日志
        if self.loss_type == "dpo":
            pos_score_mean = torch.stack(all_pos_scores).mean()
            neg_score_mean = torch.stack(all_neg_scores).mean()
            metrics = {
                "train/loss": avg_loss.item(),
                "train/pos_score": pos_score_mean.item(),
                "train/neg_score": neg_score_mean.item(),
            }
            self.log(metrics)

            if self.state.global_step % self.args.logging_steps == 0:
                logger.info(
                    f"Step {self.state.global_step}: loss={avg_loss.item():.4f}, "
                    f"pos_score={pos_score_mean.item():.4f}, neg_score={neg_score_mean.item():.4f}"
                )
        else:
            avg_kl = total_kl / used_groups
            all_rewards = torch.cat(all_rewards)
            all_advantages = torch.cat(all_advantages)
            reward_mean = all_rewards.mean()
            reward_std = all_rewards.std()
            advantage_mean = all_advantages.mean()
            advantage_std = all_advantages.std()

            metrics = {
                "train/loss": avg_loss.item(),
                "train/kl": avg_kl.item(),
                "train/reward_mean": reward_mean.item(),
                "train/reward_std": reward_std.item(),
                "train/advantage_mean": advantage_mean.item(),
                "train/advantage_std": advantage_std.item(),
            }
            self.log(metrics)

            if self.state.global_step % self.args.logging_steps == 0:
                logger.info(
                    f"Step {self.state.global_step}: loss={avg_loss.item():.4f}, "
                    f"kl={avg_kl.item():.4f}, reward={reward_mean.item():.4f}±{reward_std.item():.4f}, "
                    f"adv={advantage_mean.item():.4f}±{advantage_std.item():.4f}"
                )

        if return_outputs:
            return avg_loss, metrics
        return avg_loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        """支持 num_iterations：对同一 batch 多次更新"""
        model.train()
        inputs = self._prepare_inputs(inputs)

        total_loss = 0.0
        for _ in range(self.num_iterations):
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

            # 缩放 loss（考虑 gradient accumulation）
            if self.args.n_gpu > 1:
                loss = loss.mean()

            # 反向传播
            self.accelerator.backward(loss)
            total_loss += loss.detach()

        # 返回平均 loss（用于日志）
        return total_loss / self.num_iterations


def load_sft_model(
    sft_model_path: str,
    base_model_path: str,
    torch_dtype=torch.bfloat16,
    trust_remote_code: bool = True,
):
    """加载 SFT 模型并合并 LoRA

    Args:
        sft_model_path: SFT 模型路径（可能是 LoRA adapter 或完整模型）
        base_model_path: 基座模型路径
        torch_dtype: 模型数据类型
        trust_remote_code: 是否信任远程代码

    Returns:
        合并后的模型

    Example:
        >>> model = load_sft_model(
        ...     "outputs/sft/final",
        ...     "Qwen/Qwen3-Reranker-4B"
        ... )
    """
    from peft import PeftModel

    logger.info(f"加载 SFT 模型: {sft_model_path}")

    # 检查是否是 LoRA 模型
    adapter_config_path = Path(sft_model_path) / "adapter_config.json"

    if adapter_config_path.exists():
        # 加载基座模型
        logger.info(f"加载基座模型: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        # 加载 LoRA adapter
        model = PeftModel.from_pretrained(model, sft_model_path)
        # 合并 LoRA
        logger.info("合并 LoRA adapter...")
        model = model.merge_and_unload()
    else:
        # 直接加载合并后的模型
        model = AutoModelForCausalLM.from_pretrained(
            sft_model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )

    return model
