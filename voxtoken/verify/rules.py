from __future__ import annotations

from typing import Dict, List, Set

from ..schemas import Citation, Issue, ReportPlan
from .extract_slots import extract_slots_from_report


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
    if not plan.facts:
        return []

    pred_slots = extract_slots_from_report(report_text)
    if not pred_slots:
        return []

    def norm(x: object) -> str:
        return str(x).strip()

    issues: List[Issue] = []
    n = min(len(plan.facts), len(pred_slots))
    for i in range(int(n)):
        pf = plan.facts[int(i)]
        rf = pred_slots[int(i)]

        plan_tuple = (
            norm(getattr(pf, "finding_type", "")),
            norm(getattr(pf, "side", "U")),
            norm(getattr(pf, "location", "U")),
            norm(getattr(pf, "size_bin", "U")),
            norm(getattr(pf, "certainty", "U")),
        )
        pred_tuple = (
            norm(getattr(rf, "finding_type", "")),
            norm(getattr(rf, "side", "U")),
            norm(getattr(rf, "location", "U")),
            norm(getattr(rf, "size_bin", "U")),
            norm(getattr(rf, "certainty", "U")),
        )

        if plan_tuple != pred_tuple:
            issues.append(
                Issue(
                    type="inconsistency",
                    span=(int(i), int(i)),
                    reason=f"plan-vs-report mismatch at sent_id={int(i)}: plan={plan_tuple} pred={pred_tuple}",
                    related_tokens=list(getattr(pf, "supported_token_ids", []) or []),
                )
            )
    return issues


def check_overclaim(plan: ReportPlan, report_text: str) -> List[Issue]:
    allowed = {str(f.finding_type).strip().lower() for f in (plan.facts or []) if str(f.finding_type).strip()}
    if not allowed:
        return []

    sentences = [s.strip() for s in report_text.splitlines() if s.strip()]
    issues: List[Issue] = []
    for sent_id, sent in enumerate(sentences):
        # Heuristic parser for the repo skeleton: take the leading token before "(" or whitespace.
        s = sent.strip()
        if not s:
            continue
        cut = len(s)
        for ch in ["(", " "]:
            pos = s.find(ch)
            if pos != -1:
                cut = min(cut, pos)
        finding = s[:cut].strip().lower()
        if not finding:
            continue
        if finding not in allowed:
            issues.append(
                Issue(
                    type="overclaim",
                    span=(sent_id, sent_id),
                    reason=f"finding_type '{finding}' not present in plan",
                )
            )
    return issues


def check_unsupported(report_text: str, citations: List[Citation], plan: ReportPlan) -> List[Issue]:
    sentences = [s.strip() for s in report_text.splitlines() if s.strip()]
    if not sentences:
        return []

    cited_by_sent = {int(c.sent_id): list(c.cited_token_ids) for c in citations}
    supported_by_finding: Dict[str, Set[int]] = {}
    for fact in (plan.facts or []):
        ft = str(getattr(fact, "finding_type", "")).strip().lower()
        if not ft:
            continue
        tids = set(int(x) for x in (getattr(fact, "supported_token_ids", []) or []))
        if not tids:
            continue
        supported_by_finding.setdefault(ft, set()).update(tids)

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
            continue

        # Slot-level support check: cited tokens must overlap the plan-support for this sentence's finding_type.
        #
        # Report format is controlled by the Realizer in this repo skeleton; treat the text before the first "("
        # as the finding key. If the plan has no facts (e.g., "No findings."), skip this gate.
        if supported_by_finding:
            sent = sentences[sent_id]
            finding = sent.split("(", 1)[0].strip().lower()
            if finding and finding in supported_by_finding:
                cited = set(int(x) for x in token_ids)
                if not (cited & supported_by_finding[finding]):
                    issues.append(
                        Issue(
                            type="unsupported",
                            span=(sent_id, sent_id),
                            reason="citation does not support the sentence slot (no overlap with supported_token_ids)",
                            related_tokens=list(cited),
                        )
                    )
    return issues
