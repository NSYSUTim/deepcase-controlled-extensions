from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..common.builder_base import BaseResearchContextBuilder
from .config import HierarchicalContextConfig


class HierarchicalContextBuilder(BaseResearchContextBuilder):
    """完整可訓練的短期/長期雙分支 GRU ContextBuilder。"""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int = 128,
        max_length: int = 42,
        short_context_length: int = 10,
        long_memory_length: int = 32,
        padding_idx: int = 0,
    ) -> None:
        super().__init__(
            input_size=input_size,
            output_size=output_size,
            hidden_size=hidden_size,
            max_length=max_length,
            padding_idx=padding_idx,
        )
        self.short_context_length = int(short_context_length)
        self.long_memory_length = int(long_memory_length)

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=padding_idx)
        self.short_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.long_encoder = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.short_pool = nn.Linear(hidden_size, 1)
        self.long_pool = nn.Linear(hidden_size, 1)
        self.fusion_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.attention_score = nn.Linear(hidden_size, 1)
        self.hidden = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def _split_context(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        long_memory = X[:, : self.long_memory_length]
        short_context = X[:, self.long_memory_length :]
        return short_context, long_memory

    def _pool_stream(
        self,
        states: torch.Tensor,
        scorer: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(scorer(states).squeeze(-1), dim=1)
        pooled = torch.sum(states * weights.unsqueeze(-1), dim=1)
        return pooled, weights

    def _encode(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        short_context, long_memory = self._split_context(X)
        short_states, _ = self.short_encoder(self.embedding(short_context))
        long_states, _ = self.long_encoder(self.embedding(long_memory))

        short_summary, short_pool_attention = self._pool_stream(
            short_states,
            self.short_pool,
        )
        long_summary, long_pool_attention = self._pool_stream(
            long_states,
            self.long_pool,
        )

        gate = torch.sigmoid(
            self.fusion_gate(torch.cat([short_summary, long_summary], dim=1))
        )
        fused_summary = gate * short_summary + (1.0 - gate) * long_summary
        gated_long = long_states * (1.0 - gate).unsqueeze(1)
        gated_short = short_states * gate.unsqueeze(1)
        combined_states = torch.cat([gated_long, gated_short], dim=1)

        return {
            "combined_states": combined_states,
            "fused_summary": fused_summary,
            "short_pool_attention": short_pool_attention,
            "long_pool_attention": long_pool_attention,
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
            "short_pool_attention": encoded["short_pool_attention"],
            "long_pool_attention": encoded["long_pool_attention"],
            "fusion_gate": encoded["fusion_gate"],
        }

    def _export_init_kwargs(self) -> dict[str, Any]:
        return {
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "max_length": self.max_length,
            "short_context_length": self.short_context_length,
            "long_memory_length": self.long_memory_length,
            "padding_idx": self.padding_idx,
        }


def build_hierarchical_model_change_summary(
    config: HierarchicalContextConfig,
) -> dict[str, Any]:
    return {
        "preserved": [
            "保留 DeepCASE 的 GRU + attention + L1 + DBSCAN 主幹",
            "仍由 Interpreter 做 query / clustering / reject",
        ],
        "changed": [
            "context 由固定短窗改成 short-term context + long-term memory",
            "以雙分支 GRU 分別編碼短期與長期記憶",
            f"landmark strategy = {config.landmark_strategy}",
        ],
        "why": "這個版本真正讓 short-term 與 long-term 訊號在模型內部分流，再由 gate 融合，而不是只在前處理階段把長度拉長。",
    }
