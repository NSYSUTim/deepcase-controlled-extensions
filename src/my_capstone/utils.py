from __future__ import annotations

import json
import sys
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_dir(path: Path | str) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_path(value: Path | str, *, base_dir: Path = PROJECT_ROOT) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def slugify(value: str) -> str:
    pieces: list[str] = []
    for char in value.lower():
        if char.isalnum():
            pieces.append(char)
        elif char in {" ", "-", "_"}:
            pieces.append("-")
    slug = "".join(pieces).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "run"


def current_timestamp() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d_%H-%M-%S")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _json_ready(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def save_json(path: Path | str, payload: Any) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path


def load_json(path: Path | str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_torch(path: Path | str, payload: Any) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    torch.save(payload, output_path)
    return output_path


def load_torch(path: Path | str, *, map_location: str | torch.device = "cpu") -> Any:
    return torch.load(path, map_location=map_location, weights_only=False)


def tensor_to_cpu(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().cpu()


def save_annotated_csv(
    path: Path | str,
    frame: pd.DataFrame,
    comments: list[str],
) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        for comment in comments:
            output_file.write(f"# {comment}\n")
        frame.to_csv(output_file, index=False)
    return output_path


class TeeStream:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


@contextmanager
def capture_output(log_path: Path | str) -> Iterator[None]:
    output_path = Path(log_path)
    ensure_dir(output_path.parent)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with output_path.open("w", encoding="utf-8") as log_file:
        sys.stdout = TeeStream(original_stdout, log_file)
        sys.stderr = TeeStream(original_stderr, log_file)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
