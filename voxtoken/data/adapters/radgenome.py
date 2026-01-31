from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class RadGenomeRow:
    case_id: str
    report_path: str
    mask_path: str | None = None


def iter_radgenome_rows(root: str) -> Iterator[RadGenomeRow]:
    """
    Minimal adapter placeholder.

    RadGenome grounding is proposal-level in this repo skeleton and is not implemented as
    a runnable experiment yet. This module exists to match the file layout described in
    `docs/plan.md` section 10.
    """
    root_path = Path(root)
    if not root_path.exists():
        return iter(())
    return iter(())

