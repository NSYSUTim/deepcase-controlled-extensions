from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from deepcase.context_builder.utils import unique_2d
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class BaseResearchContextBuilder(nn.Module, ABC):
    """研究版 ContextBuilder 的完整介面骨架。

    這個基底類別保留 DeepCASE `fit / predict / query / save / load` 的使用方式，
    讓研究版 builder 可以直接接到原版 Interpreter。
    """

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        max_length: int,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.hidden_size = int(hidden_size)
        self.max_length = int(max_length)
        self.padding_idx = int(padding_idx)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @abstractmethod
    def _encode(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        """把輸入 context 轉成可供 attention 使用的隱向量狀態。"""

    @abstractmethod
    def _attention_logits(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor:
        """回傳尚未 softmax 的 attention logits。"""

    @abstractmethod
    def _decode_from_attention(
        self,
        encoded: dict[str, torch.Tensor],
        attention: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """從指定的 attention 分布解碼出事件分布與其他研究輸出。"""

    @abstractmethod
    def _export_init_kwargs(self) -> dict[str, Any]:
        """保存 / 載入模型時需要的初始化參數。"""

    def _normalize_targets(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 2 and y.shape[1] == 1:
            return y.squeeze(1)
        return y.reshape(-1)

    def _prepare_extra_tensor(self, value: Any) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device)
        if tensor.dtype.is_floating_point:
            return tensor.to(torch.float32)
        return tensor.to(torch.long)

    def _compute_loss(
        self,
        *,
        outputs: dict[str, torch.Tensor],
        event_targets: torch.Tensor,
        extra_batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        return nn.NLLLoss()(outputs["log_probs"], event_targets)

    def _forward_details(self, X: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self._encode(X)
        attention = torch.softmax(self._attention_logits(encoded), dim=1)
        decoded = self._decode_from_attention(encoded, attention)
        decoded["attention"] = attention
        return decoded

    def forward(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None = None,
        steps: int = 1,
        teach_ratio: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del y, teach_ratio
        outputs = self._forward_details(X)
        confidence = outputs["log_probs"].exp().unsqueeze(1).repeat(1, steps, 1)
        attention = outputs["attention"].unsqueeze(1).repeat(1, steps, 1)
        return confidence, attention

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        epochs: int = 10,
        batch_size: int = 128,
        learning_rate: float = 0.01,
        optimizer: type[optim.Optimizer] = optim.Adam,
        teach_ratio: float = 0.5,
        verbose: bool = True,
        **extra_tensors: Any,
    ) -> BaseResearchContextBuilder:
        del teach_ratio
        logger.info("fit %s samples", X.shape[0])
        mode = self.training
        self.train()

        X_tensor = torch.as_tensor(X, dtype=torch.int64, device=self.device)
        y_tensor = self._normalize_targets(
            torch.as_tensor(y, dtype=torch.int64, device=self.device)
        )

        ordered_extra: list[tuple[str, torch.Tensor]] = []
        for key, value in extra_tensors.items():
            if value is None:
                continue
            ordered_extra.append((key, self._prepare_extra_tensor(value)))

        dataset_tensors = [X_tensor, y_tensor, *[tensor for _, tensor in ordered_extra]]
        data_loader = DataLoader(
            TensorDataset(*dataset_tensors),
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer_instance = optimizer(self.parameters(), lr=learning_rate)

        for epoch in range(1, epochs + 1):
            iterator = data_loader
            if verbose:
                iterator = tqdm(
                    data_loader,
                    desc=f"[Epoch {epoch}/{epochs} loss=0.0000]",
                )

            total_loss = 0.0
            total_items = 0
            for batch in iterator:
                batch_X = batch[0]
                batch_y = batch[1]
                extra_batch = {
                    key: batch[index + 2] for index, (key, _tensor) in enumerate(ordered_extra)
                }

                optimizer_instance.zero_grad()
                outputs = self._forward_details(batch_X)
                loss = self._compute_loss(
                    outputs=outputs,
                    event_targets=batch_y,
                    extra_batch=extra_batch,
                )
                loss.backward()
                optimizer_instance.step()

                total_loss += float(loss.item()) * int(batch_X.shape[0])
                total_items += int(batch_X.shape[0])
                if verbose and hasattr(iterator, "set_description"):
                    iterator.set_description(
                        f"[Epoch {epoch}/{epochs} loss={total_loss / max(total_items, 1):.4f}]"
                    )

        self.train(mode)
        return self

    def predict(
        self,
        X: torch.Tensor,
        y: torch.Tensor | None = None,
        steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del y
        logger.info("predict %s samples", X.shape[0])
        mode = self.training
        self.eval()

        X_tensor = torch.as_tensor(X, dtype=torch.int64, device=self.device)
        X_unique, inverse = torch.unique(X_tensor, dim=0, return_inverse=True)

        with torch.no_grad():
            confidence, attention = self.forward(X_unique, steps=steps)

        self.train(mode)
        return confidence[inverse], attention[inverse]

    def query(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        *,
        iterations: int = 0,
        batch_size: int = 1024,
        ignore: int | None = None,
        return_optimization: float | None = None,
        verbose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        del ignore
        X_tensor = torch.as_tensor(X, dtype=torch.int64, device=self.device)
        y_tensor = torch.as_tensor(y, dtype=torch.int64, device=self.device)
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)

        X_unique, y_unique, inverse = unique_2d(X_tensor, y_tensor)
        y_unique = self._normalize_targets(y_unique)

        mode = self.training
        self.eval()

        result_confidence: list[torch.Tensor] = []
        result_attention: list[torch.Tensor] = []
        confidence_orig: list[torch.Tensor] = []
        confidence_optim: list[torch.Tensor] = []

        data_loader = DataLoader(
            TensorDataset(X_unique, y_unique),
            batch_size=batch_size,
            shuffle=False,
        )
        progress = None
        if verbose:
            progress = tqdm(
                total=max(int(iterations), 1) * len(data_loader),
                desc="最佳化 research query",
            )

        for X_batch, y_batch in data_loader:
            with torch.no_grad():
                encoded = self._encode(X_batch)
                initial_logits = self._attention_logits(encoded)
                initial_attention = torch.softmax(initial_logits, dim=1)
                initial_outputs = self._decode_from_attention(encoded, initial_attention)
                initial_log_probs = initial_outputs["log_probs"]
                initial_target = initial_log_probs.exp()[
                    torch.arange(y_batch.shape[0], device=y_batch.device),
                    y_batch,
                ]

            if return_optimization is not None:
                confidence_orig.append(initial_target >= return_optimization)

            raw_attention = Variable(initial_logits.detach().clone(), requires_grad=True)
            optimizer_instance = optim.Adam([raw_attention], lr=0.1)
            criterion = nn.NLLLoss()

            for _ in range(int(iterations)):
                optimizer_instance.zero_grad()
                optimized_attention = torch.softmax(raw_attention, dim=1)
                optimized_outputs = self._decode_from_attention(encoded, optimized_attention)
                loss = criterion(optimized_outputs["log_probs"], y_batch)
                loss.backward()
                optimizer_instance.step()
                if progress is not None:
                    progress.update()

            with torch.no_grad():
                if iterations > 0:
                    optimized_attention = torch.softmax(raw_attention, dim=1)
                    optimized_outputs = self._decode_from_attention(encoded, optimized_attention)
                    optimized_log_probs = optimized_outputs["log_probs"]
                    optimized_target = optimized_log_probs.exp()[
                        torch.arange(y_batch.shape[0], device=y_batch.device),
                        y_batch,
                    ]
                    use_optimized = optimized_target > initial_target
                    final_attention = initial_attention.clone()
                    final_attention[use_optimized] = optimized_attention[use_optimized]
                    final_log_probs = initial_log_probs.clone()
                    final_log_probs[use_optimized] = optimized_log_probs[use_optimized]
                else:
                    final_attention = initial_attention
                    final_log_probs = initial_log_probs

                if return_optimization is not None:
                    final_target = final_log_probs.exp()[
                        torch.arange(y_batch.shape[0], device=y_batch.device),
                        y_batch,
                    ]
                    confidence_optim.append(final_target >= return_optimization)

            result_confidence.append(final_log_probs.exp().cpu())
            result_attention.append(final_attention.cpu())

        if progress is not None:
            progress.close()

        self.train(mode)

        confidence = torch.cat(result_confidence, dim=0).to(self.device)
        attention = torch.cat(result_attention, dim=0).to(self.device)
        inverse = inverse.to(self.device)

        if return_optimization is not None:
            return (
                confidence,
                attention,
                inverse,
                torch.cat(confidence_orig, dim=0),
                torch.cat(confidence_optim, dim=0),
            )
        return confidence, attention, inverse

    def save(self, outfile: str | Any) -> None:
        payload = {
            "class_name": self.__class__.__name__,
            "init_kwargs": self._export_init_kwargs(),
            "state_dict": self.state_dict(),
        }
        torch.save(payload, outfile)

    @classmethod
    def load(
        cls,
        infile: str | Any,
        device: str | torch.device | None = None,
    ) -> BaseResearchContextBuilder:
        payload = torch.load(infile, map_location=device, weights_only=False)
        if "init_kwargs" not in payload:
            raise ValueError("研究版 builder 存檔缺少 init_kwargs，無法載入。")
        result = cls(**payload["init_kwargs"])
        if device is not None:
            result = result.to(device)
        result.load_state_dict(payload["state_dict"])
        return result
