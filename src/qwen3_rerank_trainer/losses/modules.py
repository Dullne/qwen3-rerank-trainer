"""nn.Module wrappers for qwen3 reranker ranking losses."""

from __future__ import annotations

import torch
import torch.nn as nn

from .lambda_loss import (
    BaseWeightingScheme,
    lambda_loss,
)
from .listwise import list_mle, listwise_softmax_ce, p_list_mle
from .pairwise import ranknet_loss
from .pointwise import pointwise_ce_from_yes_no_logits


class LambdaLoss(nn.Module):
    """LambdaLoss framework wrapper."""

    def __init__(
        self,
        sigma: float = 1.0,
        metric: str | None = "ndcg",
        weighting_scheme: BaseWeightingScheme | str | None = None,
        activation: str = "sigmoid",
        eps: float = 1e-10,
        k: int | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.sigma = sigma
        self.metric = metric
        self.weighting_scheme = weighting_scheme
        self.activation = activation
        self.eps = eps
        self.k = k
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return lambda_loss(
            scores,
            labels,
            weighting_scheme=self.weighting_scheme,
            metric=self.metric,
            sigma=self.sigma,
            activation=self.activation,
            eps=self.eps,
            k=self.k,
            reduction=self.reduction,
        )


class LambdaLossNDCG(LambdaLoss):
    """LambdaLoss tuned for NDCG."""

    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "ndcg")
        super().__init__(**kwargs)


class LambdaLossMAP(LambdaLoss):
    """LambdaLoss tuned for MAP."""

    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "map")
        super().__init__(**kwargs)


class LambdaLossMRR(LambdaLoss):
    """LambdaLoss tuned for MRR."""

    def __init__(self, **kwargs):
        kwargs.setdefault("metric", "mrr")
        super().__init__(**kwargs)


class ListwiseLoss(nn.Module):
    """ListNet-style listwise softmax cross-entropy loss."""

    def __init__(
        self,
        temperature: float = 1.0,
        target_temperature: float | None = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.target_temperature = (
            temperature if target_temperature is None else target_temperature
        )
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return listwise_softmax_ce(
            scores,
            labels,
            score_temperature=self.temperature,
            target_temperature=self.target_temperature,
            reduction=self.reduction,
        )


class ListNetLoss(ListwiseLoss):
    """Alias for the ListNet-style listwise loss."""


class ListMLELoss(nn.Module):
    """ListMLE loss."""

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-10,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return list_mle(
            scores,
            labels,
            temperature=self.temperature,
            eps=self.eps,
            reduction=self.reduction,
        )


class PositionAwareListMLELoss(nn.Module):
    """Position-aware ListMLE loss."""

    def __init__(
        self,
        temperature: float = 1.0,
        eps: float = 1e-10,
        reduction: str = "mean",
    ):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return p_list_mle(
            scores,
            labels,
            temperature=self.temperature,
            eps=self.eps,
            reduction=self.reduction,
        )


class RankNetLoss(nn.Module):
    """RankNet loss over listwise scores/labels."""

    def __init__(self, sigma: float = 1.0, max_pairs_per_batch: int = 2_000_000):
        super().__init__()
        self.sigma = sigma
        self.max_pairs_per_batch = max_pairs_per_batch

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return ranknet_loss(
            scores,
            labels,
            sigma=self.sigma,
            max_pairs_per_batch=self.max_pairs_per_batch,
        )


class PointwiseCELoss(nn.Module):
    """Pointwise cross-entropy loss over yes/no logits."""

    def forward(
        self,
        yes_logits: torch.Tensor,
        no_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        return pointwise_ce_from_yes_no_logits(yes_logits, no_logits, labels)
