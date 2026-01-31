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

        # Repo-skeleton grounding support: group evidence by a stable "root token id" so that
        # refinement (splitting a token into children) does not reorder sentences. This keeps
        # sentence IDs aligned with GT boxes in RadGenome-style manifests.
        groups: Dict[int, List[EvidenceNode]] = {}
        for ev in list(evidence or []):
            attrs = ev.attrs or {}
            root = None
            if isinstance(attrs, dict) and "root_token_id" in attrs:
                root = attrs.get("root_token_id", None)
            if root is None:
                # Fallback: use the first supported token id if present.
                ids = list(ev.supported_token_ids or [])
                root = ids[0] if ids else 0
            try:
                rid = int(root)
            except Exception:
                rid = 0
            groups.setdefault(int(rid), []).append(ev)

        facts: List[FactSlot] = []
        for rid in sorted(int(x) for x in groups.keys()):
            if len(facts) >= max_facts:
                break
            evs = groups[int(rid)]
            # Merge supported token ids within the group.
            supported: List[int] = []
            seen: set[int] = set()
            for ev in evs:
                for tid in list(ev.supported_token_ids or []):
                    try:
                        tid_i = int(tid)
                    except Exception:
                        continue
                    if tid_i in seen:
                        continue
                    seen.add(tid_i)
                    supported.append(int(tid_i))
            supported.sort()

            # Pick a representative node for slot attrs / finding type.
            rep = sorted(evs, key=lambda e: str(e.eid))[0]
            rep_attrs: Dict[str, Any] = dict(rep.attrs or {}) if isinstance(rep.attrs, dict) else {}

            facts.append(
                FactSlot(
                    finding_type=str(rep.finding_type),
                    side=str(rep_attrs.get("side", "U")),
                    location=str(rep_attrs.get("location", "U")),
                    size_bin=str(rep_attrs.get("size_bin", "U")),
                    certainty=str(rep_attrs.get("certainty", "U")),
                    supported_token_ids=supported,
                )
            )

        impression = facts[: int(self.cfg.get("max_impression", 4))]
        return ReportPlan(facts=facts, impression=impression)


__all__ = [
    "Planner",
]
