from __future__ import annotations

from pathlib import Path
from typing import Iterator

_KINDS = ("assetbundles", "resources")


def asset_bucket(name: str) -> str:
    i = name.find("_")
    return name[:i] if i >= 0 else "_"


def asset_rel(kind: str, name: str) -> Path:
    return Path(kind, asset_bucket(name), name)


def kind_rel(src: Path) -> Path:
    parts = src.parts
    for i, p in enumerate(parts):
        if p in _KINDS:
            tail = parts[i:-1]
            if len(tail) == 1:
                return Path(tail[0], asset_bucket(src.name))
            return Path(*tail)
    return Path(asset_bucket(src.name))


def io_pairs(indir: Path, outdir: Path) -> Iterator[tuple[Path, Path]]:
    if indir.is_file():
        dest = outdir / indir.name if outdir.is_dir() or not outdir.suffix else outdir
        yield indir, dest
        return
    for src in sorted(indir.rglob("*")):
        if src.is_file():
            yield src, outdir / src.relative_to(indir)
