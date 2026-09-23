from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from my_capstone.pipeline import SequenceBundle
from tqdm import tqdm

from ..common.execution import NO_EVENT_ORIGINAL_ID, build_sequence_bundle
from .config import CrossHostConfig

NO_EVENT_TOKEN_KEY = "__NO_EVENT__"


def _cross_host_token_key(*, machine: str, original_event: int, token_scope: str) -> str:
    if token_scope == "event":
        return str(int(original_event))
    if token_scope == "machine_event":
        return f"{machine}|{int(original_event)}"
    raise ValueError("cross_host token_scope must be 'event' or 'machine_event'.")


def _token_key_to_original_event(token_key: str) -> int:
    if token_key == NO_EVENT_TOKEN_KEY:
        return NO_EVENT_ORIGINAL_ID
    if "|" in token_key:
        _machine, original_event = token_key.split("|", 1)
        return int(original_event)
    return int(token_key)


def build_cross_host_preprocessing_plan(config: CrossHostConfig) -> dict[str, Any]:
    return {
        "token_scope": config.token_scope,
        "goal": "把原版 same-host only 的 context 建構，擴成 local context + cross-host companion context。",
        "local_context_definition": {
            "group_key": "machine",
            "context_length": config.local_context_length,
        },
        "companion_context_definition": {
            "group_key": "scenario",
            "exclude_same_machine": True,
            "context_length": config.companion_context_length,
            "time_window_seconds": config.companion_time_window_seconds,
        },
        "difference_from_baseline": [
            "baseline 只看同一台主機的前序事件",
            "cross_host 版本額外收集同 scenario、時間相近、但來自其他主機的 companion events",
        ],
    }


def build_cross_host_sequence_bundle(
    *,
    frame,
    config: CrossHostConfig,
    token_to_internal: dict[str, int] | None = None,
    allow_new_tokens: bool = True,
) -> SequenceBundle:
    total_context_length = config.local_context_length + config.companion_context_length
    pad_token = 0
    if token_to_internal is None:
        token_to_internal = {NO_EVENT_TOKEN_KEY: pad_token}
    else:
        token_to_internal.setdefault(NO_EVENT_TOKEN_KEY, pad_token)

    internal_to_original: dict[int, int] = {pad_token: NO_EVENT_ORIGINAL_ID}
    for token_key, token_id in token_to_internal.items():
        internal_to_original[int(token_id)] = _token_key_to_original_event(token_key)

    local_buffers: dict[str, deque[int]] = defaultdict(
        lambda: deque(maxlen=config.local_context_length)
    )
    scenario_machine_buffers: dict[str, dict[str, deque[tuple[float, int]]]] = defaultdict(dict)

    encoded_events: list[int] = []
    encoded_contexts: list[list[int]] = []
    encoded_labels: list[int] = []

    for row_index, row in enumerate(
        tqdm(
            frame.itertuples(index=False),
            total=len(frame),
            desc="cross_host: building sequence bundle",
            unit="events",
        ),
        start=1,
    ):
        scenario = str(row.scenario)
        machine = str(row.machine)
        timestamp = float(row.timestamp)
        original_event = int(row.event)
        label = int(row.label)

        token_key = _cross_host_token_key(
            machine=machine,
            original_event=original_event,
            token_scope=config.token_scope,
        )
        token_id = token_to_internal.get(token_key)
        if token_id is None:
            if not allow_new_tokens:
                raise ValueError(
                    "cross_host predict 遇到訓練期間未見過的新 token："
                    f"{token_key}。這會讓 internal event id 與已訓練模型不一致，"
                    "請先重訓此方法或只使用與訓練分佈一致的事件集合。"
                )
            token_id = len(token_to_internal)
            token_to_internal[token_key] = token_id
            internal_to_original[token_id] = original_event

        local_context = list(local_buffers[machine])
        machine_buffers = scenario_machine_buffers[scenario]
        current_machine_buffer = machine_buffers.setdefault(
            machine,
            deque(maxlen=config.companion_context_length),
        )
        companion_candidates: list[tuple[float, int]] = []
        for recent_machine, recent_events in machine_buffers.items():
            while (
                recent_events
                and timestamp - recent_events[0][0] > config.companion_time_window_seconds
            ):
                recent_events.popleft()
            if recent_machine != machine:
                companion_candidates.extend(recent_events)
        companion_candidates.sort(key=lambda item: item[0])
        companion_context = [
            recent_token
            for _recent_ts, recent_token in companion_candidates[
                -config.companion_context_length :
            ]
        ]

        padded_local = [pad_token] * (config.local_context_length - len(local_context)) + local_context
        padded_companion = [pad_token] * (
            config.companion_context_length - len(companion_context)
        ) + companion_context
        context = padded_local + padded_companion
        if len(context) != total_context_length:
            raise ValueError("cross_host context 長度與設定不一致。")

        encoded_events.append(token_id)
        encoded_contexts.append(context)
        encoded_labels.append(label)

        local_buffers[machine].append(token_id)
        current_machine_buffer.append((timestamp, token_id))

        if row_index % 250000 == 0:
            print(
                "[cross_host] bundle progress: "
                f"{row_index}/{len(frame)} rows, "
                f"tokens={len(token_to_internal)}, "
                f"scenario_buffers={len(scenario_machine_buffers)}"
            )

    return build_sequence_bundle(
        events=encoded_events,
        contexts=encoded_contexts,
        labels=encoded_labels,
        mapping=internal_to_original,
    )
