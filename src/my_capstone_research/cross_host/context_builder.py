from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.builder_base import BaseResearchContextBuilder
from .config import CrossHostConfig


class CrossHostContextBuilder(BaseResearchContextBuilder):
    """完整可訓練的跨主機雙流 GRU ContextBuilder。"""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        max_length: int = 20,
        local_context_length: int = 10,
        companion_context_length: int = 10,
        padding_idx: int = 0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            max_length=max_length,
            padding_idx=padding_idx,
        )
        self.local_context_length = int(local_context_length)
        self.companion_context_length = int(companion_context_length)

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)
        self.local_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.companion_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.local_pool = nn.Linear(hidden_size, 1)
        self.companion_pool = nn.Linear(hidden_size, 1)
        self.fusion_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def _split_context(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        local = X[:, : self.local_context_length]
        companion = X[:, self.local_context_length :]
        return local, companion

    def _pool_stream(
        self,
        states: torch.Tensor,
        scorer: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(scorer(states).squeeze(-1), dim=1)
        pooled = torch.sum(states * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def _encode(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        local_context, companion_context = self._split_context(X)

        local_states, _ = self.local_encoder(self.embedding(local_context))
        companion_states, _ = self.companion_encoder(self.embedding(companion_context))

        local_summary, local_pool_attention = self._pool_stream(
            local_states,
            self.local_pool,
        )
        companion_summary, companion_pool_attention = self._pool_stream(
            companion_states,
            self.companion_pool,
        )

        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([local_summary, companion_summary], dim=1))
        )
        fused_summary = gate * local_summary + (1.0 - gate) * companion_summary
        gated_local = local_states * gate.unsqueeze(1)
        gated_companion = companion_states * (1.0 - gate).unsqueeze(1)
        combined_states = torch.cat([gated_local, gated_companion], dim=1)

        return {
            "combined_states": combined_states,
            "fused_summary": fused_summary,
            "local_pool_attention": local_pool_attention,
            "companion_pool_attention": companion_pool_attention,
            "fusion_gate": gate,
        }

    def _attention_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.attention_score(encoded["combined_states"]).squeeze(-1)

    def _decode_from_attention(
        self,
        encoded: dict[str, torch.Tensor],
        attention: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        attended = torch.bmm(
            attention.unsqueeze(1),
            encoded["combined_states"],
        ).squeeze(1)
        fused = attended + encoded["fused_summary"]
        logits = self.out(F.relu(self.hidden(fused)))
        return {
            "log_probs": F.log_softmax(logits, dim=1),
            "pooled_context": fused,
            "local_pool_attention": encoded["local_pool_attention"],
            "companion_pool_attention": encoded["companion_pool_attention"],
            "fusion_gate": encoded["fusion_gate"],
        }

    def _export_init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "local_context_length": self.local_context_length,
            "companion_context_length": self.companion_context_length,
            "padding_idx": self.padding_idx,
        }


def build_cross_host_model_change_summary(config: CrossHostConfig) -> dict[str, Any]:
    return {
        "preserved": [
            "保留 DeepCASE 的 GRU + attention + L1 + DBSCAN 主幹",
            "仍由 Interpreter 進行 query / clustering / reject 流程",
        ],
        "changed": [
            "context 從單一 same-host 序列，改成 local stream + companion stream",
            "以雙流 GRU 分別編碼本機與跨主機前文",
            f"fusion strategy = {config.fusion_strategy}",
        ],
        "why": "這個版本真正把跨主機 companion context 接進 ContextBuilder，而不是只在前處理階段拼接字串。",
    }
