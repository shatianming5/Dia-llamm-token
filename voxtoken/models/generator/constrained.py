from __future__ import annotations

from typing import List, Tuple

from ...schemas import Citation, ReportPlan


def enforce_plan_constraints(draft_text: str, plan: ReportPlan) -> Tuple[str, List[str]]:
    """
    Enforce slot/value constraints so that report facts come from `plan`.

    Returns:
        (fixed_text, violations)
    """
    violations: List[str] = []
    if not plan.facts:
        return draft_text, violations

    lower = draft_text.lower()
    for fact in plan.facts:
        key = str(fact.finding_type).strip().lower()
        if not key:
            continue
        if key not in lower:
            violations.append(f"missing finding_type in realized text: {fact.finding_type}")

    return draft_text, violations


def require_citations(sentences: List[str], citations: List[Citation]) -> List[str]:
    """
    Minimal gate: every sentence must have a citation entry with non-empty token ids.

    Returns:
        violations (empty means pass)
    """
    cited_by_sent = {int(c.sent_id): list(c.cited_token_ids) for c in citations}
    violations: List[str] = []
    for sent_id in range(len(sentences)):
        token_ids = cited_by_sent.get(sent_id, [])
        if not token_ids:
            violations.append(f"sentence {sent_id} missing citation")
    return violations
