from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ..schemas import Citation, Issue, ReportPlan
from .rules import check_inconsistency, check_missing_slots, check_overclaim, check_unsupported
from .scorer import score_from_issues


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
        missing = check_missing_slots(plan, report_text)
        inconsistency = check_inconsistency(plan, report_text)
        overclaim = check_overclaim(plan, report_text)
        unsupported = check_unsupported(report_text, citations, plan)

        issues = [*missing, *inconsistency, *overclaim, *unsupported]

        slot_f1 = 0.0
        if plan.facts:
            missing_ratio = float(len(missing)) / float(len(plan.facts))
            slot_f1 = max(0.0, 1.0 - missing_ratio)

        weights = self.cfg.get("weights", {})
        score = score_from_issues(
            slot_f1,
            issues,
            alpha=float(weights.get("missing_slot", 1.0)),
            beta=float(weights.get("inconsistency", 1.0)),
            gamma=float(weights.get("overclaim", 1.0)),
            delta=float(weights.get("unsupported", 1.0)),
        )

        return float(score), issues


__all__ = [
    "Verifier",
]
