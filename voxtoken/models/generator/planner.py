from __future__ import annotations

from typing import Any, Dict, List

from ...schemas import EvidenceNode, ReportPlan


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
        raise NotImplementedError


__all__ = [
    "Planner",
]

