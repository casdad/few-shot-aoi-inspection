from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def payload_hash(payload: Any) -> str:
    text = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def write_yaml(path: str | Path, payload: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return p
