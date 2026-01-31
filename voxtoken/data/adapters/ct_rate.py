from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class CTRateRow:
    case_id: str
    report_path: str
    volume_path: str | None = None


def iter_ct_rate_rows(root: str) -> Iterator[CTRateRow]:
    """
    Minimal adapter placeholder.

    The repo skeleton uses `voxtoken.data.ingest` as the runnable interface and may inline
    CT-RATE parsing logic there. This module exists to match the file layout described in
    `docs/plan.md` section 10.
    """
    root_path = Path(root)
    if not root_path.exists():
        return iter(())
    return iter(())


def ensure_paths_exist(rows: Iterable[CTRateRow]) -> None:
    for r in rows:
        if not Path(r.report_path).exists():
            raise FileNotFoundError(r.report_path)
        if r.volume_path and not Path(r.volume_path).exists():
            raise FileNotFoundError(r.volume_path)

