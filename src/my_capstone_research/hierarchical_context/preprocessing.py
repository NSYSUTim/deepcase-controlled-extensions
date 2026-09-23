from __future__ import annotations

from collections import deque
from typing import Any

from my_capstone.pipeline import SequenceBundle
from tqdm import tqdm

from ..common.execution import NO_EVENT_ORIGINAL_ID, build_sequence_bundle
from .memory_selector import select_landmarks
from .config import HierarchicalContextConfig


def build_hierarchical_preprocessing_plan(config: HierarchicalContextConfig) -> dict[str, Any]:
    return {
        "goal": "把 baseline 的固定短窗 context，拆成 short-term context 與 long-term memory。",
        "short_term_context": {
            "context_length": config.short_context_length,
            "purpose": "保留最近事件的局部行為訊號",
        },
        "long_term_memory": {
            "memory_length": config.long_memory_length,
            "landmark_strategy": config.landmark_strategy,
            "time_window_seconds": config.landmark_time_window_seconds,
            "purpose": "保留較遠但可能是前導事件的長距離訊號",
        },
    }


def build_hierarchical_sequence_bundle(
    *,
    frame,
    config: HierarchicalContextConfig,
    token_to_internal: dict[int, int] | None = None,
    allow_new_tokens: bool = True,
) -> SequenceBundle:
    pad_token = 0
    max_history_events = config.short_context_length + config.landmark_candidate_limit
    if token_to_internal is None:
        token_to_internal = {}
    internal_to_original: dict[int, int] = {pad_token: NO_EVENT_ORIGINAL_ID}
    for original_event, token_id in token_to_internal.items():
        internal_to_original[int(token_id)] = int(original_event)

    machine_histories: dict[str, deque[tuple[float, int]]] = {}
    encoded_events: list[int] = []
    encoded_contexts: list[list[int]] = []
    encoded_labels: list[int] = []

    for row_index, row in enumerate(
        tqdm(
            frame.itertuples(index=False),
            total=len(frame),
            desc="hierarchical_context: building sequence bundle",
            unit="events",
        ),
        start=1,
    ):
        machine = str(row.machine)
        timestamp = float(row.timestamp)
        original_event = int(row.event)
        label = int(row.label)

        token_id = token_to_internal.get(original_event)
        if token_id is None:
            if not allow_new_tokens:
                raise ValueError(
                    "hierarchical_context predict 遇到訓練期間未見過的新 event："
                    f"{original_event}。這會讓 internal event id 與已訓練模型不一致，"
                    "請先重訓此方法或只使用與訓練分佈一致的事件集合。"
                )
            token_id = len(token_to_internal) + 1
            token_to_internal[original_event] = token_id
            internal_to_original[token_id] = original_event

        history = machine_histories.setdefault(machine, deque(maxlen=max_history_events))
        while history and timestamp - history[0][0] > config.landmark_time_window_seconds:
            history.popleft()

        previous_tokens = [token for _ts, token in history]
        short_context = previous_tokens[-config.short_context_length :]
        long_candidates = previous_tokens[: max(0, len(previous_tokens) - len(short_context))]
        long_memory = select_landmarks(
            long_candidates,
            top_k=config.long_memory_length,
            strategy=config.landmark_strategy,
        )

        padded_long = [pad_token] * (config.long_memory_length - len(long_memory)) + long_memory
        padded_short = [pad_token] * (config.short_context_length - len(short_context)) + short_context
        context = padded_long + padded_short

        encoded_events.append(token_id)
        encoded_contexts.append(context)
        encoded_labels.append(label)

        history.append((timestamp, token_id))

        if row_index % 50000 == 0:
            print(
                "[hierarchical_context] bundle progress: "
                f"{row_index}/{len(frame)} rows, "
                f"tokens={len(token_to_internal)}, "
                f"active_machines={len(machine_histories)}, "
                f"history_limit={max_history_events}"
            )

    return build_sequence_bundle(
        events=encoded_events,
        contexts=encoded_contexts,
        labels=encoded_labels,
        mapping=internal_to_original,
    )
