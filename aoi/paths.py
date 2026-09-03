from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _legacy_candidate(path: Path) -> Path | None:
    parts = path.parts
    lowered = [part.casefold() for part in parts]
    for prefix in (("aoi_train", "outputs"), ("aoi_code", "outputs")):
        for index in range(len(parts) - 1):
            if tuple(lowered[index : index + 2]) == prefix:
                return PROJECT_ROOT / "outputs" / Path(*parts[index + 2 :])
    for anchor in ("assets", "data", "outputs", "audit", "configs"):
        if anchor in lowered:
            index = lowered.index(anchor)
            return PROJECT_ROOT / Path(*parts[index:])
    return None


def resolve_artifact(
    value: str | Path,
    *,
    base: str | Path | None = None,
    strict: bool = True,
) -> Path:
    """Resolve current relative manifests and archived absolute-path manifests."""
    path = Path(value)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        if base is not None:
            candidates.append(Path(base) / path)
        candidates.append(PROJECT_ROOT / path)

    legacy = _legacy_candidate(path)
    if legacy is not None:
        candidates.append(legacy)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
    if strict:
        attempted = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"artifact not found; tried: {attempted}")
    return candidates[0].resolve()


def portable_path(value: str | Path) -> str:
    path = Path(value).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)
