from __future__ import annotations

from typing import List

from ..schemas import Citation, Issue, ReportPlan


def check_missing_slots(plan: ReportPlan, report_text: str) -> List[Issue]:
    if not plan.facts:
        return []

    text = report_text.lower()
    issues: List[Issue] = []
    for fact in plan.facts:
        key = str(fact.finding_type).strip().lower()
        if not key:
            continue
        if key not in text:
            issues.append(
                Issue(
                    type="missing_slot",
                    span=(0, 0),
                    reason=f"missing finding_type in report: {fact.finding_type}",
                    related_tokens=list(fact.supported_token_ids),
                )
            )
    return issues


def check_inconsistency(plan: ReportPlan, report_text: str) -> List[Issue]:
    return []


def check_overclaim(plan: ReportPlan, report_text: str) -> List[Issue]:
    return []


def check_unsupported(report_text: str, citations: List[Citation], plan: ReportPlan) -> List[Issue]:
    sentences = [s.strip() for s in report_text.splitlines() if s.strip()]
    if not sentences:
        return []

    cited_by_sent = {int(c.sent_id): list(c.cited_token_ids) for c in citations}
    issues: List[Issue] = []
    for sent_id in range(len(sentences)):
        token_ids = cited_by_sent.get(sent_id, [])
        if not token_ids:
            issues.append(
                Issue(
                    type="unsupported",
                    span=(sent_id, sent_id),
                    reason="missing or empty citation for sentence",
                )
            )
    return issues
