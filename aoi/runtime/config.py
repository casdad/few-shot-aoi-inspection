from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path).resolve(strict=True)
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"YAML root must be a mapping: {p}")
    payload["_config_path"] = str(p)
    return payload


def project_root(config: dict[str, Any]) -> Path:
    return Path(config["_config_path"]).parent.parent


def resolve_path(config: dict[str, Any], value: str) -> Path:
    p = Path(value)
    return p.resolve() if p.is_absolute() else (project_root(config) / p).resolve()


def resolve_device(value: str) -> str:
    value = str(value).strip().lower()
    if value == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return value
