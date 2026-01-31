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
        emit_citations = bool(self.cfg.get("emit_citations", True))
        overclaim = bool(self.cfg.get("overclaim_extra_sentence", False))
        overclaim_finding = str(self.cfg.get("overclaim_finding_type", "hallucination")).strip() or "hallucination"

        facts = list(plan.facts or [])
        if not facts:
            sentences = ["No findings."]
            citations = [Citation(sent_id=0, cited_token_ids=[0])] if emit_citations else []
            return "\n".join(sentences) + "\n", citations

        for sent_id, fact in enumerate(facts):
            side = str(fact.side)
            location = str(fact.location)
            size_bin = str(fact.size_bin)
            certainty = str(fact.certainty)
            finding = str(fact.finding_type)
            sentences.append(f"{finding} (side={side}, location={location}, size={size_bin}, certainty={certainty}).")

            if emit_citations:
                token_ids = list(fact.supported_token_ids or [])
                if not token_ids:
                    token_ids = [0]
                citations.append(Citation(sent_id=sent_id, cited_token_ids=token_ids))

        if overclaim:
            # Append a fact not present in the plan to exercise "no-constrained" ablation.
            sent_id = len(sentences)
            sentences.append(f"{overclaim_finding} (unconstrained).")
            if emit_citations:
                citations.append(Citation(sent_id=sent_id, cited_token_ids=[0]))

        return "\n".join(sentences) + "\n", citations


__all__ = [
    "Realizer",
]
