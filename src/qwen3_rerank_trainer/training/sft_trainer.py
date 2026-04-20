"""
Contrastive SFT Trainer for Qwen3-Reranker

基于 HuggingFace Trainer 的对比学习训练器。
支持多种损失函数：BCE、InfoNCE、ListMLE、LambdaLoss、RankNet。
"""
import torch
import torch.nn.functional as F
from transformers import Trainer
from typing import Dict, Optional, Any, List


# 支持的损失类型
SUPPORTED_LOSS_TYPES = ["bce", "infonce", "list_mle", "lambda_loss", "ranknet"]


class ContrastiveSFTTrainer(Trainer):
    """对比学习 SFT 训练器

    支持多种损失函数训练 Qwen3-Reranker。
    支持分块前向传播，允许使用很大的 n_docs 而不会 OOM。

    Args:
        yes_token_id: "yes" token 的 ID
        no_token_id: "no" token 的 ID
        chunk_size: 分块大小（0 表示不分块）
        loss_type: 损失函数类型
            - "bce": Binary Cross-Entropy (默认)
            - "infonce": InfoNCE 对比学习损失
            - "list_mle": ListMLE 排序损失
            - "lambda_loss": LambdaLoss (NDCG 优化)
            - "ranknet": RankNet 成对排序损失
        temperature: InfoNCE 温度参数 (仅 infonce 使用)
        infonce_mode: InfoNCE 正例策略 (仅 infonce 使用)
            - "single": 单正例（默认）
            - "posset": 多正例正例集
            - "avgpos": 多正例逐正例对比
        lambda_metric: LambdaLoss 目标指标 (仅 lambda_loss 使用)
            - "ndcg" (默认)
            - "map"
            - "mrr"
        *args, **kwargs: 传递给 Trainer 的参数

    Example:
        >>> trainer = ContrastiveSFTTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=train_dataset,
        ...     data_collator=collator,
        ...     yes_token_id=yes_id,
        ...     no_token_id=no_id,
        ...     loss_type="infonce",  # 使用 InfoNCE 损失
        ...     temperature=0.05,
        ...     chunk_size=16,
        ... )
        >>> trainer.train()
    """

    def __init__(
        self,
        yes_token_id: int,
        no_token_id: int,
        chunk_size: int = 0,
        loss_type: str = "bce",
        temperature: float = 0.05,
        infonce_mode: str = "single",
        lambda_metric: str = "ndcg",
        ranknet_max_pairs_per_batch: int = 2000000,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.chunk_size = chunk_size
        self.loss_type = loss_type.lower()
        self.temperature = temperature
        self.infonce_mode = infonce_mode.lower()
        if self.infonce_mode not in {"single", "posset", "avgpos"}:
            raise ValueError(f"Unsupported infonce_mode: {self.infonce_mode}")
        self.lambda_metric = lambda_metric
        self.ranknet_max_pairs_per_batch = int(ranknet_max_pairs_per_batch)

        if self.loss_type not in SUPPORTED_LOSS_TYPES:
            raise ValueError(
                f"不支持的损失类型: {loss_type}. "
                f"支持的类型: {SUPPORTED_LOSS_TYPES}"
            )

        # 延迟导入损失函数
        self._loss_fn = None

    def _get_logits(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """获取 yes/no logits 差值

        Args:
            model: 模型
            input_ids: 输入 token IDs
            attention_mask: 注意力掩码

        Returns:
            tensor: yes_logits - no_logits，形状为 (batch_size,)
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # 获取最后一个 token 的 logits（使用 left padding，所以直接取 -1 位置）
        last_token_logits = logits[:, -1, :]

        # 提取 yes/no logits
        yes_logits = last_token_logits[:, self.yes_token_id]
        no_logits = last_token_logits[:, self.no_token_id]

        return yes_logits - no_logits

    def compute_loss(
        self,
        model,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: Optional[int] = None,
    ):
        """计算损失

        支持多种损失函数：
        - BCE: Binary Cross-Entropy
        - InfoNCE: 对比学习损失
        - ListMLE: 概率排序损失
        - LambdaLoss: NDCG 优化损失
        - RankNet: 成对排序损失

        Args:
            model: 模型
            inputs: 输入字典，包含 input_ids, attention_mask, labels, group_sizes
            return_outputs: 是否返回输出
            num_items_in_batch: batch 中的样本数（可选）

        Returns:
            loss 或 (loss, outputs) 元组
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        group_sizes = inputs.get("group_sizes", None)

        # 判断是否使用分块模式
        use_chunked = self.chunk_size > 0 and input_ids.size(0) > self.chunk_size

        if use_chunked:
            # 分块前向传播
            all_logit_diffs = []
            total_samples = input_ids.size(0)

            for chunk_start in range(0, total_samples, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_samples)
                chunk_ids = input_ids[chunk_start:chunk_end]
                chunk_mask = attention_mask[chunk_start:chunk_end]

                logit_diff = self._get_logits(model, chunk_ids, chunk_mask)
                all_logit_diffs.append(logit_diff)

            logit_diff = torch.cat(all_logit_diffs, dim=0)
        else:
            # 一次性前向传播
            logit_diff = self._get_logits(model, input_ids, attention_mask)

        # 根据损失类型计算损失
        if self.loss_type == "bce":
            loss = self._compute_bce_loss(logit_diff, labels)
        elif self.loss_type == "infonce":
            loss = self._compute_infonce_loss(logit_diff, labels, group_sizes)
        elif self.loss_type == "list_mle":
            loss = self._compute_list_mle_loss(logit_diff, labels, group_sizes)
        elif self.loss_type == "lambda_loss":
            loss = self._compute_lambda_loss(logit_diff, labels, group_sizes)
        elif self.loss_type == "ranknet":
            loss = self._compute_ranknet_loss(logit_diff, labels, group_sizes)
        else:
            raise ValueError(f"未知损失类型: {self.loss_type}")

        if return_outputs:
            return loss, {"logits": logit_diff}
        return loss

    def _compute_bce_loss(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """BCE 损失"""
        return F.binary_cross_entropy_with_logits(
            scores,
            labels.float(),
            reduction='mean'
        )

    def _compute_infonce_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        group_sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """InfoNCE 损失（组内对比学习）

        infonce_loss expects:
        - scores: [B, M] - scores for B groups, each with M candidates
        - pos_mask: [B, M] - boolean mask for positives
        """
        from ..losses import infonce_loss

        pos_mask = labels > 0.5
        if group_sizes is None:
            # 无分组信息，整个 batch 作为一组
            return infonce_loss(
                scores.unsqueeze(0),  # [1, M]
                pos_mask=pos_mask.unsqueeze(0),
                scale=1.0 / self.temperature,  # infonce_loss uses scale (1/temperature)
                mode=self.infonce_mode,
            )

        # 按组计算损失
        total_loss = 0.0
        offset = 0
        for group_size in group_sizes:
            group_scores = scores[offset:offset + group_size].unsqueeze(0)  # [1, group_size]
            group_labels = labels[offset:offset + group_size]
            pos_mask = group_labels > 0.5
            loss = infonce_loss(
                group_scores,
                pos_mask=pos_mask.unsqueeze(0),
                scale=1.0 / self.temperature,  # infonce_loss uses scale (1/temperature)
                mode=self.infonce_mode,
            )
            total_loss += loss
            offset += group_size

        return total_loss / len(group_sizes)

    def _compute_list_mle_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        group_sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """ListMLE 损失"""
        from ..losses import list_mle

        if group_sizes is None:
            return list_mle(
                scores.unsqueeze(0),
                labels.float().unsqueeze(0)
            )

        # 按组计算损失
        total_loss = 0.0
        offset = 0
        for group_size in group_sizes:
            group_scores = scores[offset:offset + group_size].unsqueeze(0)
            group_labels = labels[offset:offset + group_size].float().unsqueeze(0)

            loss = list_mle(group_scores, group_labels)
            total_loss += loss
            offset += group_size

        return total_loss / len(group_sizes)

    def _compute_lambda_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        group_sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """LambdaLoss 损失"""
        from ..losses import lambda_loss

        if group_sizes is None:
            return lambda_loss(
                scores.unsqueeze(0),
                labels.float().unsqueeze(0),
                metric=self.lambda_metric
            )

        # 按组计算损失
        total_loss = 0.0
        offset = 0
        for group_size in group_sizes:
            group_scores = scores[offset:offset + group_size].unsqueeze(0)
            group_labels = labels[offset:offset + group_size].float().unsqueeze(0)

            loss = lambda_loss(
                group_scores,
                group_labels,
                metric=self.lambda_metric
            )
            total_loss += loss
            offset += group_size

        return total_loss / len(group_sizes)

    def _compute_ranknet_loss(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        group_sizes: Optional[List[int]] = None
    ) -> torch.Tensor:
        """RankNet 损失"""
        from ..losses import ranknet_loss

        if group_sizes is None:
            return ranknet_loss(
                scores.unsqueeze(0),
                labels.float().unsqueeze(0),
                max_pairs_per_batch=self.ranknet_max_pairs_per_batch,
            )

        # 按组计算损失
        total_loss = 0.0
        offset = 0
        for group_size in group_sizes:
            group_scores = scores[offset:offset + group_size].unsqueeze(0)
            group_labels = labels[offset:offset + group_size].float().unsqueeze(0)

            loss = ranknet_loss(
                group_scores,
                group_labels,
                max_pairs_per_batch=self.ranknet_max_pairs_per_batch,
            )
            total_loss += loss
            offset += group_size

        return total_loss / len(group_sizes)


def get_yes_no_token_ids(tokenizer, yes_token: str = "yes", no_token: str = "no"):
    """获取 yes/no token 的 ID

    Args:
        tokenizer: HuggingFace tokenizer
        yes_token: "yes" token 字符串
        no_token: "no" token 字符串

    Returns:
        tuple: (yes_token_id, no_token_id)

    Raises:
        ValueError: 如果 token 不在词表中
    """
    yes_token_id = tokenizer.convert_tokens_to_ids(yes_token)
    no_token_id = tokenizer.convert_tokens_to_ids(no_token)

    if yes_token_id == tokenizer.unk_token_id:
        raise ValueError(f"'{yes_token}' token 未在词表中找到")
    if no_token_id == tokenizer.unk_token_id:
        raise ValueError(f"'{no_token}' token 未在词表中找到")

    return yes_token_id, no_token_id
