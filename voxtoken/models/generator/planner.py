from __future__ import annotations

from typing import Any, Dict, List

from ...schemas import EvidenceNode, FactSlot, ReportPlan


class Planner:
    """Evidence -> report plan (deterministic M0 or learned M2+)."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def build_plan(self, evidence: List[EvidenceNode]) -> ReportPlan:
        """
        Deterministic baseline suggestion:
          - sort by severity/certainty
          - group by anatomy
          - cap per organ
        """
        max_facts = int(self.cfg.get("max_facts", 8))
        ev_sorted = sorted(evidence, key=lambda e: str(e.eid))

        facts: List[FactSlot] = []
        for ev in ev_sorted:
            if len(facts) >= max_facts:
                break
            attrs = dict(ev.attrs or {})
            facts.append(
                FactSlot(
                    finding_type=str(ev.finding_type),
                    side=str(attrs.get("side", "U")),
                    location=str(attrs.get("location", "U")),
                    size_bin=str(attrs.get("size_bin", "U")),
                    certainty=str(attrs.get("certainty", "U")),
                    supported_token_ids=list(ev.supported_token_ids),
                )
            )

        impression = facts[: int(self.cfg.get("max_impression", 4))]
        return ReportPlan(facts=facts, impression=impression)


__all__ = [
    "Planner",
]
