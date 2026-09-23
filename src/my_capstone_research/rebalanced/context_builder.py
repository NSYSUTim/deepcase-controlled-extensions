from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.builder_base import BaseResearchContextBuilder
from .config import RebalancedConfig
from .losses import sigmoid_focal_loss, weighted_binary_loss


class RebalancedContextBuilder(BaseResearchContextBuilder):
    """完整可訓練的 joint-objective Rebalanced ContextBuilder。"""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        max_length: int = 10,
        padding_idx: int = 0,
        loss_type: str = "focal",
        focal_gamma: float = 2.0,
        positive_class_weight: float = 3.0,
        negative_class_weight: float = 1.0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            max_length=max_length,
            padding_idx=padding_idx,
        )
        self.loss_type = str(loss_type)
        self.focal_gamma = float(focal_gamma)
        self.positive_class_weight = float(positive_class_weight)
        self.negative_class_weight = float(negative_class_weight)

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)
        self.encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.attention_score = nn.Linear(hidden_size, 1)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.triage_head = nn.Linear(hidden_size, 1)

    def _encode(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        states, _ = self.encoder(self.embedding(X))
        return {"states": states}

    def _attention_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.attention_score(encoded["states"]).squeeze(-1)

    def _decode_from_attention(
        self,
        encoded: dict[str, torch.Tensor],
        attention: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pooled = torch.bmm(attention.unsqueeze(1), encoded["states"]).squeeze(1)
        hidden = F.relu(self.hidden(pooled))
        logits = self.out(hidden)
        triage_logits = self.triage_head(hidden).squeeze(-1)
        return {
            "log_probs": F.log_softmax(logits, dim=1),
            "pooled_context": hidden,
            "triage_logits": triage_logits,
        }

    def _compute_loss(
        self,
        *,
        outputs: dict[str, torch.Tensor],
        event_targets: torch.Tensor,
        extra_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        next_event_loss = nn.NLLLoss()(outputs["log_probs"], event_targets)
        triage_targets = extra_batch.get("triage_targets")
        if triage_targets is None:
            return next_event_loss

        triage_targets = triage_targets.to(torch.float32)
        if self.loss_type == "focal":
            triage_loss = sigmoid_focal_loss(
                outputs["triage_logits"],
                triage_targets,
                gamma=self.focal_gamma,
            )
        else:
            triage_loss = weighted_binary_loss(
                outputs["triage_logits"],
                triage_targets,
                positive_weight=self.positive_class_weight,
                negative_weight=self.negative_class_weight,
            )
        return next_event_loss + triage_loss

    def _export_init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "padding_idx": self.padding_idx,
            "loss_type": self.loss_type,
            "focal_gamma": self.focal_gamma,
            "positive_class_weight": self.positive_class_weight,
            "negative_class_weight": self.negative_class_weight,
        }


def build_rebalanced_model_change_summary(config: RebalancedConfig) -> dict[str, Any]:
    return {
        "preserved": [
            "保留 DeepCASE 的 GRU + attention + L1 + DBSCAN 主幹",
            "仍以 attention fingerprint 餵給 clustering",
        ],
        "changed": [
            "next-event only 改成 next-event + supervised triage joint objective",
            f"loss_type = {config.loss_type}",
            "後段 prediction 使用 calibrated posterior + reject-aware triage",
        ],
        "why": "這個版本把資料不平衡真正拉進 representation learning 與決策校準，而不是只改輸出表格。",
    }
