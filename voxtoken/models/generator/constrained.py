from __future__ import annotations

from typing import List, Tuple

from ...schemas import Citation, ReportPlan


def enforce_plan_constraints(draft_text: str, plan: ReportPlan) -> Tuple[str, List[str]]:
    """
    Enforce slot/value constraints so that report facts come from `plan`.

    Returns:
        (fixed_text, violations)
    """
    raise NotImplementedError


def require_citations(sentences: List[str], citations: List[Citation]) -> List[str]:
    """
    Minimal gate: every sentence must have a citation entry with non-empty token ids.

    Returns:
        violations (empty means pass)
    """
    raise NotImplementedError

