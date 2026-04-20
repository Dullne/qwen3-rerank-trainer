"""
LambdaLoss 框架 + 权重方案

支持可插拔的权重方案，可以优化不同的排序指标：
- NDCGLoss2PPScheme: 优化 NDCG（默认，推荐）
- MAPScheme: 优化 MAP（二元相关性）
- MRRScheme: 优化 MRR（只关心首个结果）
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import torch
import torch.nn.functional as F


class BaseWeightingScheme(ABC):
    """LambdaLoss 权重方案基类

    权重方案定义了如何计算 pairwise ranking loss 的权重。
    不同的权重方案对应不同的优化目标（NDCG、MAP、MRR 等）。
    """

    @abstractmethod
    def __call__(
        self,
        gain: torch.Tensor,
        discount: torch.Tensor,
        labels_sorted: torch.Tensor,
    ) -> torch.Tensor:
        """计算 pairwise 权重矩阵

        Args:
            gain: [B, M] 归一化收益 (2^label - 1) / maxDCG
            discount: [1, M] 位置折扣 log2(rank + 1)
            labels_sorted: [B, M] 按预测分数排序后的标签

        Returns:
            weights: [B, M, M] pairwise 权重矩阵
        """
        raise NotImplementedError


class NoWeightingScheme(BaseWeightingScheme):
    """无权重方案：所有文档对权重相等

    等价于普通的 pairwise ranking loss，不针对特定指标优化。
    """

    def __call__(self, gain, discount, labels_sorted):
        return gain.new_tensor(1.0)


class NDCGLoss1Scheme(BaseWeightingScheme):
    """NDCGLoss1 权重方案

    weight = gain / discount
    简单但界限较松，原论文不推荐。
    """

    def __call__(self, gain, discount, labels_sorted):
        return (gain / discount)[:, :, None]


class NDCGLoss2Scheme(BaseWeightingScheme):
    """NDCGLoss2 权重方案

    weight = |Δdiscount(|i-j|)| × |gain_i - gain_j|
    其中 Δdiscount = |1/log2(|i-j|) - 1/log2(|i-j|+1)|

    考虑了交换两个文档后 discount 的变化，比 Loss1 更紧的 NDCG 上界。
    """

    def __call__(self, gain, discount, labels_sorted):
        B, M = gain.shape
        device = gain.device

        pos_idxs = torch.arange(1, M + 1, device=device)
        delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])

        delta_idxs_safe = delta_idxs.clamp(min=1)
        delta_discount = torch.abs(
            torch.pow(discount[0, delta_idxs_safe - 1].abs(), -1.0)
            - torch.pow(discount[0, delta_idxs].abs(), -1.0)
        )
        delta_discount.diagonal().zero_()

        gain_diff = torch.abs(gain[:, :, None] - gain[:, None, :])

        return delta_discount[None, :, :] * gain_diff


class LambdaRankScheme(BaseWeightingScheme):
    """LambdaRank 权重方案

    weight = |1/discount_i - 1/discount_j| × |gain_i - gain_j|

    直接使用两个位置的 discount 差，计算简单但界限比 Loss2 松。
    """

    def __call__(self, gain, discount, labels_sorted):
        discount_diff = torch.abs(
            torch.pow(discount[:, :, None], -1.0)
            - torch.pow(discount[:, None, :], -1.0)
        )

        gain_diff = torch.abs(gain[:, :, None] - gain[:, None, :])

        return discount_diff * gain_diff


class NDCGLoss2PPScheme(BaseWeightingScheme):
    """NDCGLoss2++ 权重方案（推荐，性能最优）

    weight = μ × NDCGLoss2_weight + LambdaRank_weight

    结合了 Loss2 的紧界限和 LambdaRank 的高效性。
    原论文 (CIKM 2018) 实验证明性能最强。
    """

    def __init__(self, mu: float = 10.0):
        """
        Args:
            mu: NDCGLoss2 权重的缩放因子，默认 10.0
        """
        self.mu = mu
        self._ndcg_loss2 = NDCGLoss2Scheme()
        self._lambda_rank = LambdaRankScheme()

    def __call__(self, gain, discount, labels_sorted):
        loss2_weight = self._ndcg_loss2(gain, discount, labels_sorted)
        lambda_weight = self._lambda_rank(gain, discount, labels_sorted)
        return self.mu * loss2_weight + lambda_weight


class MAPScheme(BaseWeightingScheme):
    """MAP (Mean Average Precision) 权重方案

    适用于二元相关性标签（0/1）。

    权重计算：|ΔAP| ≈ |rel_i - rel_j| × |P@rank_i - P@rank_j|
    其中 P@rank = cumsum(rel) / rank

    对于 AP 优化，权重反映交换两个文档对 AP 的影响。
    """

    def __call__(self, gain, discount, labels_sorted):
        B, M = labels_sorted.shape
        device = labels_sorted.device

        binary_labels = (labels_sorted > 0).float()

        cumsum_rel = torch.cumsum(binary_labels, dim=1)
        ranks = torch.arange(1, M + 1, device=device, dtype=gain.dtype)
        precision_at_k = cumsum_rel / ranks

        precision_diff = torch.abs(
            precision_at_k[:, :, None] - precision_at_k[:, None, :]
        )

        rel_diff = torch.abs(
            binary_labels[:, :, None] - binary_labels[:, None, :]
        )

        return precision_diff * rel_diff


class MRRScheme(BaseWeightingScheme):
    """MRR (Mean Reciprocal Rank) 权重方案

    只关心第一个正例的位置。

    权重计算：
    - 对于包含"第一个正例"的文档对：weight = |1/rank_i - 1/rank_j|
    - 其他文档对：weight = fallback_weight（默认 0.1）

    适用于只关心首个结果的场景（如问答、导航搜索）。
    """

    def __init__(self, fallback_weight: float = 0.1):
        """
        Args:
            fallback_weight: 非首正例对的权重（0 表示完全忽略）
        """
        self.fallback_weight = fallback_weight

    def __call__(self, gain, discount, labels_sorted):
        B, M = labels_sorted.shape
        device = labels_sorted.device

        binary_labels = (labels_sorted > 0).float()

        has_pos = binary_labels.any(dim=1)
        first_pos_idx = binary_labels.argmax(dim=1)
        first_pos_mask = torch.zeros_like(binary_labels)
        if has_pos.any():
            batch_idx = torch.arange(B, device=device)[has_pos]
            first_pos_mask[batch_idx, first_pos_idx[has_pos]] = 1.0

        ranks = torch.arange(1, M + 1, device=device, dtype=gain.dtype)
        rr = 1.0 / ranks
        rr_diff = torch.abs(rr[:, None] - rr[None, :])

        involves_first_pos = (
            first_pos_mask[:, :, None] + first_pos_mask[:, None, :]
        )

        weights = torch.where(
            involves_first_pos > 0,
            rr_diff[None, :, :].expand(B, -1, -1),
            torch.full((B, M, M), self.fallback_weight, device=device, dtype=gain.dtype)
        )

        rel_diff = torch.abs(
            binary_labels[:, :, None] - binary_labels[:, None, :]
        )

        return weights * rel_diff


# ============================================================================
# 便捷的权重方案注册表
# ============================================================================

WEIGHTING_SCHEMES = {
    'no_weighting': NoWeightingScheme,
    'ndcg_loss1': NDCGLoss1Scheme,
    'ndcg_loss2': NDCGLoss2Scheme,
    'lambda_rank': LambdaRankScheme,
    'ndcg_loss2pp': NDCGLoss2PPScheme,
    'map': MAPScheme,
    'mrr': MRRScheme,
}


def get_weighting_scheme(name: str, **kwargs) -> BaseWeightingScheme:
    """根据名称获取权重方案实例

    Args:
        name: 方案名称，支持 'no_weighting', 'ndcg_loss1', 'ndcg_loss2',
              'lambda_rank', 'ndcg_loss2pp', 'map', 'mrr'
        **kwargs: 传递给方案构造函数的参数

    Returns:
        权重方案实例
    """
    if name not in WEIGHTING_SCHEMES:
        raise ValueError(f"Unknown weighting scheme: {name!r}, "
                        f"available: {list(WEIGHTING_SCHEMES.keys())}")
    return WEIGHTING_SCHEMES[name](**kwargs)


# ============================================================================
# LambdaLoss 主函数
# ============================================================================

def lambda_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    weighting_scheme: BaseWeightingScheme | str | None = None,
    metric: str | None = None,
    sigma: float = 1.0,
    activation: str = "sigmoid",
    eps: float = 1e-10,
    k: int | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    LambdaLoss Framework for Ranking Metric Optimization.

    支持可插拔的权重方案，可以优化不同的排序指标：
      - NDCGLoss2PPScheme: 优化 NDCG（默认，推荐）
      - MAPScheme: 优化 MAP（二元相关性）
      - MRRScheme: 优化 MRR（只关心首个结果）

    Args:
        scores: shape [B, M], 每个 query 的 M 个候选的模型分数
        labels: shape [B, M], 相关性标签，支持 {0, 1} 二元或 graded labels
        weighting_scheme: 权重方案，可以是：
            - BaseWeightingScheme 实例（推荐）
            - 字符串名称：'ndcg_loss2pp', 'map', 'mrr', 'ndcg_loss2', 'lambda_rank', 'no_weighting'
            - None: 使用 metric 指定的方案（默认 NDCG）
        metric: 简化入口，支持 'ndcg'/'map'/'mrr'（也可直接传方案名）
            仅在 weighting_scheme 为空时生效
        sigma: 分数差的缩放因子
        activation: 激活函数 "sigmoid" | "tanh_scaled" | "arctan_scaled"
        eps: 数值稳定性常数
        k: 只在 top-k 范围内计算损失，默认 None 使用全部文档
        reduction: "none" | "mean" | "sum"

    Returns:
        loss: LambdaLoss

    References:
        - The LambdaLoss Framework for Ranking Metric Optimization (CIKM 2018)
    """
    if scores.ndim != 2 or labels.ndim != 2:
        raise ValueError(f"scores/labels must be 2D, got scores={scores.ndim}D labels={labels.ndim}D")
    if scores.shape != labels.shape:
        raise ValueError(f"scores shape {tuple(scores.shape)} != labels shape {tuple(labels.shape)}")

    if weighting_scheme is None:
        if metric is None:
            weighting_scheme = NDCGLoss2PPScheme()
        else:
            metric_key = metric.lower()
            metric_map = {
                "ndcg": "ndcg_loss2pp",
                "map": "map",
                "mrr": "mrr",
            }
            scheme_name = metric_map.get(metric_key, metric_key)
            weighting_scheme = get_weighting_scheme(scheme_name)
    elif isinstance(weighting_scheme, str):
        weighting_scheme = get_weighting_scheme(weighting_scheme)

    B, M = scores.shape
    device = scores.device

    scores_sorted, indices_pred = scores.sort(descending=True, dim=-1)
    labels_sorted_by_pred = torch.gather(labels, dim=1, index=indices_pred)
    labels_sorted_true, _ = labels.sort(descending=True, dim=-1)

    true_diffs = labels_sorted_by_pred.unsqueeze(2) - labels_sorted_by_pred.unsqueeze(1)
    padded_pairs_mask = torch.isfinite(true_diffs)
    valid_pairs_mask = padded_pairs_mask & (true_diffs > 0)

    labels_sorted_by_pred_clamped = labels_sorted_by_pred.clamp(min=0.0)
    labels_sorted_true_clamped = labels_sorted_true.clamp(min=0.0)
    gain_pred = torch.pow(2.0, labels_sorted_by_pred_clamped) - 1.0

    positions = torch.arange(1, M + 1, dtype=scores.dtype, device=device)
    discount = torch.log2(1.0 + positions).unsqueeze(0)

    k_val = k or M
    gain_true = torch.pow(2.0, labels_sorted_true_clamped) - 1.0
    max_dcg = (gain_true[:, :k_val] / discount[:, :k_val]).sum(dim=-1).clamp(min=eps)

    gain_normalized = gain_pred / max_dcg.unsqueeze(1)

    weights = weighting_scheme(gain_normalized, discount, labels_sorted_by_pred)

    score_diffs = scores_sorted.unsqueeze(2) - scores_sorted.unsqueeze(1)
    score_diffs = score_diffs.clamp(min=-1e8, max=1e8)
    score_diffs.masked_fill_(torch.isnan(score_diffs), 0.0)

    activation_fns = {
        'sigmoid': torch.sigmoid,
        'tanh_scaled': lambda x: (torch.tanh(x) + 1) / 2,
        'arctan_scaled': lambda x: torch.atan(x) / torch.pi + 0.5,
    }
    activation_fn = activation_fns.get(activation, torch.sigmoid)

    weighted_probas = activation_fn(sigma * score_diffs).clamp(min=eps)
    losses = -(weights * torch.log2(weighted_probas))

    k_val = k or M
    ndcg_at_k_mask = torch.zeros((M, M), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k_val, :k_val] = True
    final_mask = valid_pairs_mask & ndcg_at_k_mask.unsqueeze(0)

    masked_losses = losses[final_mask]
    if masked_losses.numel() == 0:
        loss_val = torch.tensor(0.0, device=device, requires_grad=True)
    else:
        loss_val = masked_losses.mean()

    if reduction == "none":
        loss_per_query = torch.zeros(B, device=device)
        for i in range(B):
            query_mask = final_mask[i]
            if query_mask.any():
                loss_per_query[i] = losses[i][query_mask].mean()
        return loss_per_query
    elif reduction == "mean":
        return loss_val
    elif reduction == "sum":
        return loss_val * masked_losses.numel()
    else:
        raise ValueError(f"Unsupported reduction: {reduction!r}")
