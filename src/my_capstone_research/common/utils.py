from __future__ import annotations

from pathlib import Path

from my_capstone.utils import (
    PROJECT_ROOT,
    current_timestamp,
    ensure_dir,
    load_json,
    resolve_path,
    save_json,
    slugify,
)


def build_research_run_id(method_name: str, variant_name: str) -> str:
    return f"{current_timestamp()}_{slugify(method_name)}_{slugify(variant_name)}"


def write_text(path: Path | str, content: str) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    output_path.write_text(content, encoding="utf-8")
    return output_path


__all__ = [
    "PROJECT_ROOT",
    "build_research_run_id",
    "ensure_dir",
    "load_json",
    "resolve_path",
    "save_json",
    "slugify",
    "write_text",
]
