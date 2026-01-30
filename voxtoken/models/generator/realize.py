from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...schemas import Citation, ReportPlan


class Realizer:
    """Plan -> report text + per-sentence citations."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def realize(self, plan: ReportPlan) -> Tuple[str, List[Citation]]:
        """Returns report text and per-sentence citations."""
        raise NotImplementedError


__all__ = [
    "Realizer",
]

