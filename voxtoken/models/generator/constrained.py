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
        fixed = "No findings.\n"
        if [s for s in str(draft_text).splitlines() if s.strip()] and "no findings" not in str(draft_text).lower():
            violations.append("plan has no facts but report is non-empty")
        return fixed, violations

    def _sentences(text: str) -> List[str]:
        return [s.strip() for s in str(text).splitlines() if s.strip()]

    def _render_fact(fact: object) -> str:
        finding = str(getattr(fact, "finding_type", "")).strip() or "U"
        side = str(getattr(fact, "side", "U")).strip() or "U"
        location = str(getattr(fact, "location", "U")).strip() or "U"
        size_bin = str(getattr(fact, "size_bin", "U")).strip() or "U"
        certainty = str(getattr(fact, "certainty", "U")).strip() or "U"
        return f"{finding} (side={side}, location={location}, size={size_bin}, certainty={certainty})."

    def _parse_sentence_tuple(sent: str) -> Tuple[str, str, str, str, str] | None:
        s = str(sent).strip()
        if not s:
            return None
        if s.lower().startswith("no findings"):
            return None

        finding = s
        side = "U"
        location = "U"
        size_bin = "U"
        certainty = "U"

        open_i = s.find("(")
        close_i = s.find(")")
        if open_i >= 0:
            finding = s[:open_i].strip()
            inside = s[open_i + 1 : close_i if close_i > open_i else len(s)]
            parts = [p.strip() for p in inside.split(",") if p.strip()]
            kv = {}
            for p in parts:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                kv[k.strip()] = v.strip().strip(".")

            side = str(kv.get("side", side))
            location = str(kv.get("location", location))
            size_bin = str(kv.get("size", kv.get("size_bin", size_bin)))
            certainty = str(kv.get("certainty", certainty))

        return (
            str(finding).strip(),
            str(side).strip(),
            str(location).strip(),
            str(size_bin).strip(),
            str(certainty).strip(),
        )

    sent_in = _sentences(draft_text)
    facts = list(plan.facts or [])
    if not sent_in:
        violations.append("empty report text")
        return draft_text, violations

    fixed: List[str] = []
    n = min(len(sent_in), len(facts))
    for i in range(int(n)):
        plan_sent = _render_fact(facts[i])
        pred_tuple = _parse_sentence_tuple(sent_in[i])
        plan_tuple = _parse_sentence_tuple(plan_sent)
        if pred_tuple != plan_tuple:
            violations.append(f"rewrite sent_id={i}: plan={plan_tuple} pred={pred_tuple}")
            fixed.append(plan_sent)
        else:
            fixed.append(sent_in[i])

    if len(sent_in) > len(facts):
        violations.append(f"drop extra sentences: {len(sent_in) - len(facts)}")

    if len(sent_in) < len(facts):
        violations.append(f"missing sentences: expected {len(facts)} got {len(sent_in)}")

    return "\n".join(fixed) + ("\n" if fixed else ""), violations


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
