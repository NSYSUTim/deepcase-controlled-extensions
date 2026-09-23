from __future__ import annotations

from collections import Counter
import numpy as np


def event_rarity_scores(event_ids: list[int]) -> dict[int, float]:
    counts = Counter(event_ids)
    total = max(sum(counts.values()), 1)
    return {event_id: 1.0 - (count / total) for event_id, count in counts.items()}


def select_landmarks(
    event_ids: list[int],
    *,
    top_k: int,
    strategy: str = "recent_distinct",
) -> list[int]:
    if top_k <= 0 or not event_ids:
        return []

    if strategy == "recent_distinct":
        result: list[int] = []
        seen: set[int] = set()
        for event_id in reversed(event_ids):
            if event_id in seen:
                continue
            seen.add(event_id)
            result.append(event_id)
            if len(result) >= top_k:
                break
        result.reverse()
        return result

    if strategy == "uniform":
        if len(event_ids) <= top_k:
            return list(event_ids)
        indices = np.linspace(0, len(event_ids) - 1, num=top_k, dtype=int)
        return [event_ids[int(index)] for index in indices]

    rarity = event_rarity_scores(event_ids)
    ranked = sorted(
        enumerate(event_ids),
        key=lambda item: (rarity.get(item[1], 0.0), item[0]),
        reverse=True,
    )
    selected_positions = sorted(index for index, _event_id in ranked[:top_k])
    return [event_ids[index] for index in selected_positions]
