from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import Citation, Issue, ReportPlan


class Verifier:
    """Programmatic verification: missing-slot / inconsistency / overclaim / unsupported."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def verify(self, report_text: str, citations: List[Citation], plan: ReportPlan) -> Tuple[float, List[Issue]]:
        """
        1) parse report -> slots (or use the plan as canonical)
        2) check missing_slot / inconsistency / overclaim / unsupported

        Returns:
            (score, issues)
        """
        raise NotImplementedError


__all__ = [
    "Verifier",
]

