from __future__ import annotations

from typing import Any, Dict, List, Tuple

from ...schemas import Citation, ReportPlan


class Realizer:
    """Plan -> report text + per-sentence citations."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg

    def realize(self, plan: ReportPlan) -> Tuple[str, List[Citation]]:
        """Returns report text and per-sentence citations."""
        sentences: List[str] = []
        citations: List[Citation] = []

        facts = list(plan.facts or [])
        if not facts:
            sentences = ["No findings."]
            citations = [Citation(sent_id=0, cited_token_ids=[0])]
            return "\n".join(sentences) + "\n", citations

        for sent_id, fact in enumerate(facts):
            side = str(fact.side)
            location = str(fact.location)
            size_bin = str(fact.size_bin)
            certainty = str(fact.certainty)
            finding = str(fact.finding_type)
            sentences.append(f"{finding} (side={side}, location={location}, size={size_bin}, certainty={certainty}).")

            token_ids = list(fact.supported_token_ids or [])
            if not token_ids:
                token_ids = [0]
            citations.append(Citation(sent_id=sent_id, cited_token_ids=token_ids))

        return "\n".join(sentences) + "\n", citations


__all__ = [
    "Realizer",
]
