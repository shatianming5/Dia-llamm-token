from __future__ import annotations

import json
import sys

from ..schemas import Citation, FactSlot, ReportPlan
from ..verify.verifier import Verifier


def main() -> None:
    plan_fact = FactSlot(
        finding_type="nodule",
        side="R",
        location="U",
        size_bin="U",
        certainty="U",
        supported_token_ids=[1, 2],
    )
    plan = ReportPlan(facts=[plan_fact], impression=[plan_fact])

    # Deliberate mismatch vs plan_fact.side ("R" -> "L") to trigger inconsistency.
    report = "nodule (side=L, location=U, size=U, certainty=U).\n"
    citations = [Citation(sent_id=0, cited_token_ids=[1])]

    verifier = Verifier({"weights": {"missing_slot": 1.0, "inconsistency": 1.0, "overclaim": 1.0, "unsupported": 1.0}})
    _score, issues = verifier.verify(report, citations, plan)

    ok = any(getattr(it, "type", "") == "inconsistency" for it in issues)
    if not ok:
        print("[ERR] expected an inconsistency issue but none was found", file=sys.stderr)
        for it in issues:
            try:
                print(f"[ERR] issue: {it}", file=sys.stderr)
            except Exception:
                continue
        sys.exit(1)

    print(json.dumps({"ok": True}, ensure_ascii=False))


if __name__ == "__main__":
    main()

