from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Sequence, Tuple

from ..schemas import Citation, EvidenceNode, Issue, ReportPlan, Token, TokenFeatures, TraceStep
from ..torch_compat import Tensor, no_grad
from ..verify.verifier import Verifier
from ..models.evidence_head import EvidenceHead
from ..models.generator.planner import Planner
from ..models.generator.realize import Realizer
from ..models.policy import SplitPolicy
from ..models.tokenizer import TokenPyramid, Tokenizer3D


class RefineRunner:
    def __init__(
        self,
        tokenizer: Tokenizer3D,
        evidence_head: EvidenceHead,
        planner: Planner,
        realizer: Realizer,
        verifier: Verifier,
        policy: SplitPolicy,
        cfg: Dict[str, Any],
    ):
        self.tokenizer = tokenizer
        self.evidence_head = evidence_head
        self.planner = planner
        self.realizer = realizer
        self.verifier = verifier
        self.policy = policy
        self.cfg = cfg

    @no_grad()
    def run_case(self, volume: Tensor, budget_B: int) -> Dict[str, Any]:
        pyramid = self.tokenizer.build_pyramid(volume)  # Path A
        tk = self.tokenizer.select_tokens(pyramid, active_nodes=[], budget_B=budget_B)

        plan, report, cites, score, issues = self._gen_verify(tk, pyramid)
        trace: List[TraceStep] = []

        max_rounds = int(self.cfg.get("refine", {}).get("max_rounds", 0))
        for k in range(max_rounds):
            feats = self._featurize_tokens(tk, pyramid, cites, issues)
            split_ids = self._select_splits(feats, budget_left=budget_B - len(tk))
            tk2 = self._refine_select(pyramid, tk, split_ids, budget_B)

            plan2, report2, cites2, score2, issues2 = self._gen_verify(tk2, pyramid)
            trace.append(
                TraceStep(
                    k=k,
                    budget_total=budget_B,
                    budget_used=len(tk2),
                    split_token_ids=list(split_ids),
                    added_token_ids=[t.token_id for t in tk2 if t.token_id not in {x.token_id for x in tk}],
                    verifier_score_before=float(score),
                    verifier_score_after=float(score2),
                    latency_ms={},
                )
            )

            if self._stop(score, score2, len(tk), len(tk2)):
                break
            tk, plan, report, cites, score, issues = tk2, plan2, report2, cites2, score2, issues2

        return self._dump_artifacts(report, cites, plan, trace, issues)

    def _gen_verify(
        self, tokens: List[Token], pyramid: TokenPyramid
    ) -> Tuple[ReportPlan, str, List[Citation], float, List[Issue]]:
        raise NotImplementedError

    def _featurize_tokens(
        self, tokens: List[Token], pyramid: TokenPyramid, citations: List[Citation], issues: List[Issue]
    ) -> List[TokenFeatures]:
        raise NotImplementedError

    def _select_splits(self, feats: List[TokenFeatures], budget_left: int) -> Sequence[int]:
        raise NotImplementedError

    def _refine_select(
        self, pyramid: TokenPyramid, tokens: List[Token], split_ids: Sequence[int], budget_B: int
    ) -> List[Token]:
        raise NotImplementedError

    def _stop(self, score_before: float, score_after: float, tokens_before: int, tokens_after: int) -> bool:
        raise NotImplementedError

    def _dump_artifacts(
        self,
        report: str,
        citations: List[Citation],
        plan: ReportPlan,
        trace: List[TraceStep],
        issues: List[Issue],
    ) -> Dict[str, Any]:
        return {
            "report": report,
            "citations": [asdict(c) for c in citations],
            "plan": asdict(plan),
            "trace": [asdict(t) for t in trace],
            "issues": [asdict(i) for i in issues],
        }


__all__ = [
    "RefineRunner",
]

