from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ..schemas import Citation, EvidenceNode, Issue, ReportPlan, Token, TokenFeatures, TraceStep
from ..torch_compat import Tensor, no_grad
from ..verify.verifier import Verifier
from ..models.evidence_head import EvidenceHead
from ..models.generator.constrained import enforce_plan_constraints, require_citations
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

        return self._dump_artifacts(
            budget_B=int(budget_B),
            tokens_used=int(len(tk)),
            verifier_score=float(score),
            report=report,
            citations=cites,
            plan=plan,
            trace=trace,
            issues=issues,
        )

    def _gen_verify(
        self, tokens: List[Token], pyramid: TokenPyramid
    ) -> Tuple[ReportPlan, str, List[Citation], float, List[Issue]]:
        evidence: List[EvidenceNode] = self.evidence_head.forward(tokens, None)
        plan = self.planner.build_plan(evidence)
        report, citations = self.realizer.realize(plan)

        fixed_text, plan_violations = enforce_plan_constraints(report, plan)
        cite_violations = require_citations([s for s in fixed_text.splitlines() if s.strip()], citations)

        score, issues = self.verifier.verify(fixed_text, citations, plan)

        if plan_violations:
            for v in plan_violations:
                issues.append(Issue(type="overclaim", span=(0, 0), reason=str(v)))
            score = float(score) - float(len(plan_violations))

        if cite_violations:
            for v in cite_violations:
                issues.append(Issue(type="unsupported", span=(0, 0), reason=str(v)))
            score = float(score) - float(len(cite_violations))

        return plan, fixed_text, citations, float(score), issues

    def _featurize_tokens(
        self, tokens: List[Token], pyramid: TokenPyramid, citations: List[Citation], issues: List[Issue]
    ) -> List[TokenFeatures]:
        pressure: Dict[int, float] = {}
        for c in citations:
            for tid in c.cited_token_ids:
                pressure[int(tid)] = pressure.get(int(tid), 0.0) + 1.0

        feats: List[TokenFeatures] = []
        for t in tokens:
            feats.append(
                TokenFeatures(
                    token_id=int(t.token_id),
                    level=int(t.level),
                    recon_error=0.0,
                    evidence_entropy=0.0,
                    citation_pressure=float(pressure.get(int(t.token_id), 0.0)),
                    history_splits=int(len(t.children_ids)),
                )
            )
        return feats

    def _select_splits(self, feats: List[TokenFeatures], budget_left: int) -> Sequence[int]:
        if budget_left <= 0:
            return []

        refine_cfg = dict(self.cfg.get("refine", {}))
        max_splits = int(refine_cfg.get("max_splits_per_round", budget_left))
        max_splits = max(0, min(int(budget_left), int(max_splits)))
        if max_splits <= 0:
            return []

        scored = self.policy.score(feats)
        return [tid for tid, _ in scored[:max_splits]]

    def _refine_select(
        self, pyramid: TokenPyramid, tokens: List[Token], split_ids: Sequence[int], budget_B: int
    ) -> List[Token]:
        current_ids = {int(t.token_id) for t in tokens}
        if len(current_ids) >= int(budget_B):
            return list(tokens)

        all_tokens = list(pyramid.tokens_by_level.get(0, []))
        all_tokens.sort(key=lambda t: int(t.token_id))
        out = list(tokens)

        for t in all_tokens:
            if len(out) >= int(budget_B):
                break
            if int(t.token_id) in current_ids:
                continue
            out.append(t)
            current_ids.add(int(t.token_id))

        out.sort(key=lambda t: int(t.token_id))
        return out

    def _stop(self, score_before: float, score_after: float, tokens_before: int, tokens_after: int) -> bool:
        if tokens_after == tokens_before and score_after <= score_before:
            return True
        min_delta = float(self.cfg.get("refine", {}).get("min_score_delta", 1e-6))
        if (score_after - score_before) < min_delta:
            return True
        return False

    def _dump_artifacts(
        self,
        *,
        budget_B: int,
        tokens_used: int,
        verifier_score: float,
        report: str,
        citations: List[Citation],
        plan: ReportPlan,
        trace: List[TraceStep],
        issues: List[Issue],
    ) -> Dict[str, Any]:
        return {
            "budget_B": int(budget_B),
            "tokens_used": int(tokens_used),
            "verifier_score": float(verifier_score),
            "report": report,
            "citations": [asdict(c) for c in citations],
            "plan": asdict(plan),
            "trace": [asdict(t) for t in trace],
            "issues": [asdict(i) for i in issues],
        }


__all__ = [
    "RefineRunner",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})


def _make_dummy_volume(shape_cdhw: Sequence[int]) -> List[List[List[List[float]]]]:
    c, d, h, w = [int(x) for x in shape_cdhw]
    return [[[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(d)] for _ in range(c)]


def run_infer_refine(cfg: Dict[str, Any], *, budget_B: int) -> Dict[str, Any]:
    tokenizer = Tokenizer3D(dict(cfg.get("tokenizer", {})))
    evidence_head = EvidenceHead(dict(cfg.get("evidence", {})))
    planner = Planner(dict(cfg.get("planner", {})))
    realizer = Realizer(dict(cfg.get("realizer", {})))
    verifier = Verifier(dict(cfg.get("verifier", {})))
    policy = SplitPolicy(dict(cfg.get("policy", {})))

    runner = RefineRunner(tokenizer, evidence_head, planner, realizer, verifier, policy, cfg)

    vol_cfg = dict(cfg.get("volume", {}))
    shape = vol_cfg.get("shape_cdhw", [1, 8, 8, 8])
    volume = _make_dummy_volume(shape)

    return runner.run_case(volume, budget_B=int(budget_B))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference refine loop (minimal baseline).")
    parser.add_argument("--out", required=True, help="Output directory (writes run.json/summary.json)")
    parser.add_argument("--budget", type=int, default=16, help="Token budget B")
    parser.add_argument("--config", default="voxtoken/configs/inference.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    cfg = dict(cfg)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_infer_refine(cfg, budget_B=int(args.budget))

    # Inject minimal meta summary for audit.
    summary = {
        "timestamp_utc": _utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(cfg_path),
        "budget_B": int(args.budget),
    }

    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "run_path": str(out_dir / "run.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
