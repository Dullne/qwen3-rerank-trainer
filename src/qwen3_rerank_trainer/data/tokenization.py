"""
Tokenization 模块

提供 Qwen3-Reranker 格式的 tokenization 函数。
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import torch

from .formatting import PREFIX, SUFFIX, format_input


def tokenize_for_training(
    tokenizer,
    query: str,
    docs: List[str],
    labels: Optional[List[int]] = None,
    instruction: Optional[str] = None,
    max_length: int = 8192,
) -> Dict[str, torch.Tensor]:
    """Tokenize query + docs 用于训练

    Args:
        tokenizer: 分词器
        query: 查询
        docs: 文档列表
        labels: 标签列表（0/1）
        instruction: 任务指令
        max_length: 最大序列长度

    Returns:
        dict with:
            - input_ids: [N, L]
            - attention_mask: [N, L]
            - labels: [N] (如果提供了 labels)
            - yes_token_id: int
            - no_token_id: int
    """
    prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)

    texts = [format_input(query, doc, instruction) for doc in docs]

    max_content_length = max_length - len(prefix_tokens) - len(suffix_tokens)
    encoded = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_content_length,
        return_attention_mask=False,
        add_special_tokens=False,
    )

    for i, input_ids in enumerate(encoded['input_ids']):
        encoded['input_ids'][i] = prefix_tokens + input_ids + suffix_tokens

    tokenizer.padding_side = 'left'
    inputs = tokenizer.pad(
        {'input_ids': encoded['input_ids']},
        padding=True,
        return_tensors='pt',
        max_length=max_length,
    )

    yes_token_id = tokenizer.convert_tokens_to_ids("yes")
    no_token_id = tokenizer.convert_tokens_to_ids("no")

    result = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'yes_token_id': yes_token_id,
        'no_token_id': no_token_id,
    }

    if labels is not None:
        result['labels'] = torch.tensor(labels, dtype=torch.long)

    return result


def extract_yes_no_logits(
    logits: torch.Tensor,
    yes_token_id: int,
    no_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """从模型输出中提取 yes/no logits

    Args:
        logits: [N, L, V] 模型输出
        yes_token_id: yes token 的 id
        no_token_id: no token 的 id

    Returns:
        yes_logits: [N]
        no_logits: [N]
    """
    last_logits = logits[:, -1, :]

    yes_logits = last_logits[:, yes_token_id]
    no_logits = last_logits[:, no_token_id]

    return yes_logits, no_logits


def compute_scores(
    yes_logits: torch.Tensor,
    no_logits: torch.Tensor,
) -> torch.Tensor:
    """计算 P(yes) 分数

    使用 sigmoid(yes - no)，等价于 softmax([no, yes])[1]

    Args:
        yes_logits: [N]
        no_logits: [N]

    Returns:
        scores: [N] P(yes) 分数，范围 [0, 1]
    """
    return torch.sigmoid(yes_logits - no_logits)


def forward_and_get_logits(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    yes_token_id: int,
    no_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """前向传播并提取 yes/no logits 和分数

    Args:
        model: 语言模型
        input_ids: [N, L]
        attention_mask: [N, L]
        yes_token_id: yes token id
        no_token_id: no token id

    Returns:
        yes_logits: [N]
        no_logits: [N]
        scores: [N] P(yes)
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    yes_logits, no_logits = extract_yes_no_logits(
        outputs.logits, yes_token_id, no_token_id
    )
    scores = compute_scores(yes_logits, no_logits)

    return yes_logits, no_logits, scores
