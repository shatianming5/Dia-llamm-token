from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ..schemas import Citation, EvidenceNode, Issue, ReportPlan, Token, TokenFeatures, TraceStep
from ..torch_compat import Tensor, no_grad
from ..verify.verifier import Verifier
from ..models.evidence_head import EvidenceHead
from ..generation.constrained import enforce_plan_constraints, require_citations
from ..generation.planner import Planner
from ..generation.realizer import Realizer
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
        t_start = time.perf_counter()
        t0 = t_start
        pyramid = self.tokenizer.build_pyramid(volume)  # Path A
        tk = self.tokenizer.select_tokens(pyramid, active_nodes=[], budget_B=budget_B)
        t1 = time.perf_counter()

        evidence, plan, report, cites, score, issues, lat_gv = self._gen_verify(tk, pyramid)
        trace: List[TraceStep] = []

        base_latency = {
            "tokenize": float((t1 - t0) * 1000.0),
            "generate": float(lat_gv.get("generate", 0.0)),
            "verify": float(lat_gv.get("verify", 0.0)),
        }
        base_latency["total"] = float(base_latency["tokenize"] + base_latency["generate"] + base_latency["verify"])

        max_rounds = int(self.cfg.get("refine", {}).get("max_rounds", 0))
        for k in range(max_rounds):
            tk0 = time.perf_counter()
            feats = self._featurize_tokens(tk, pyramid, volume, cites, issues)
            split_ids = self._select_splits(feats, budget_left=budget_B - len(tk))
            tk2, executed_split_ids = self._refine_select(pyramid, tk, split_ids, budget_B)
            tk1 = time.perf_counter()

            evidence2, plan2, report2, cites2, score2, issues2, lat_gv2 = self._gen_verify(tk2, pyramid)
            step_latency = {
                "tokenize": float((tk1 - tk0) * 1000.0),
                "generate": float(lat_gv2.get("generate", 0.0)),
                "verify": float(lat_gv2.get("verify", 0.0)),
            }
            step_latency["total"] = float(step_latency["tokenize"] + step_latency["generate"] + step_latency["verify"])
            trace.append(
                TraceStep(
                    k=k,
                    budget_total=budget_B,
                    budget_used=len(tk2),
                    split_token_ids=list(executed_split_ids),
                    added_token_ids=[t.token_id for t in tk2 if t.token_id not in {x.token_id for x in tk}],
                    verifier_score_before=float(score),
                    verifier_score_after=float(score2),
                    latency_ms=step_latency,
                )
            )

            if self._stop(score, score2, len(tk), len(tk2)):
                break
            tk, evidence, plan, report, cites, score, issues = tk2, evidence2, plan2, report2, cites2, score2, issues2

        t_end = time.perf_counter()
        # Use overall runtime as the authoritative latency total (includes refine rounds).
        base_latency["total"] = float((t_end - t_start) * 1000.0)

        return self._dump_artifacts(
            budget_B=int(budget_B),
            tokens_used=int(len(tk)),
            verifier_score=float(score),
            tokens=list(tk),
            report=report,
            citations=cites,
            plan=plan,
            evidence_nodes=evidence,
            trace=trace,
            issues=issues,
            latency_ms=base_latency,
        )

    def _gen_verify(
        self, tokens: List[Token], pyramid: TokenPyramid
    ) -> Tuple[List[EvidenceNode], ReportPlan, str, List[Citation], float, List[Issue], Dict[str, float]]:
        t0 = time.perf_counter()
        evidence: List[EvidenceNode] = self.evidence_head.forward(tokens, None)
        plan = self.planner.build_plan(evidence)
        report, citations = self.realizer.realize(plan)

        gates = dict(self.cfg.get("gates", {}))
        do_plan_constraints = bool(gates.get("enforce_plan_constraints", True))
        do_require_citations = bool(gates.get("require_citations", True))

        fixed_text = report
        plan_violations: List[str] = []
        if do_plan_constraints:
            fixed_text, plan_violations = enforce_plan_constraints(report, plan)

        cite_violations: List[str] = []
        if do_require_citations:
            cite_violations = require_citations([s for s in fixed_text.splitlines() if s.strip()], citations)

        t1 = time.perf_counter()
        score, issues = self.verifier.verify(fixed_text, citations, plan)
        t2 = time.perf_counter()

        if do_require_citations and cite_violations:
            for v in cite_violations:
                # Parse "sentence {sent_id} missing citation" for span localization.
                span = (0, 0)
                try:
                    parts = str(v).strip().split()
                    if len(parts) >= 2 and parts[0].lower() == "sentence":
                        sid = int(parts[1])
                        span = (int(sid), int(sid))
                except Exception:
                    span = (0, 0)
                issues.append(Issue(type="unsupported", span=span, reason=str(v)))
            score = float(score) - float(len(cite_violations))

        latency = {
            "generate": float((t1 - t0) * 1000.0),
            "verify": float((t2 - t1) * 1000.0),
        }
        return evidence, plan, fixed_text, citations, float(score), issues, latency

    def _featurize_tokens(
        self, tokens: List[Token], pyramid: TokenPyramid, volume: Tensor, citations: List[Citation], issues: List[Issue]
    ) -> List[TokenFeatures]:
        # Feature proxies (stdlib-only): use per-token patch statistics as recon/entropy signals.
        sx, sy, sz = self.tokenizer.cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])  # type: ignore[assignment]
        if not (isinstance(sx, (int, float)) and isinstance(sy, (int, float)) and isinstance(sz, (int, float))):
            sx, sy, sz = 1.0, 1.0, 1.0
        sx, sy, sz = float(sx), float(sy), float(sz)

        def _box_mm_to_zyx_slices(box_mm: Tuple[float, float, float, float, float, float]) -> Tuple[int, int, int, int, int, int]:
            x0, x1, y0, y1, z0, z1 = box_mm
            ix0 = int(round(float(x0) / sx))
            ix1 = int(round(float(x1) / sx))
            iy0 = int(round(float(y0) / sy))
            iy1 = int(round(float(y1) / sy))
            iz0 = int(round(float(z0) / sz))
            iz1 = int(round(float(z1) / sz))
            return iz0, iz1, iy0, iy1, ix0, ix1

        def _patch_variance(token: Token) -> float:
            # Welford mean/variance over (C,D,H,W) patch.
            try:
                c = len(volume)  # type: ignore[arg-type]
                d = len(volume[0])  # type: ignore[index]
                h = len(volume[0][0])  # type: ignore[index]
                w = len(volume[0][0][0])  # type: ignore[index]
            except Exception:
                return 0.0, 0.0, 0.0

            z0, z1, y0, y1, x0, x1 = _box_mm_to_zyx_slices(token.omega_box_mm)
            z0 = max(0, min(int(z0), int(d)))
            z1 = max(0, min(int(z1), int(d)))
            y0 = max(0, min(int(y0), int(h)))
            y1 = max(0, min(int(y1), int(h)))
            x0 = max(0, min(int(x0), int(w)))
            x1 = max(0, min(int(x1), int(w)))
            if z1 <= z0 or y1 <= y0 or x1 <= x0:
                return 0.0, 0.0, 0.0

            n = 0
            mean = 0.0
            m2 = 0.0
            vmax = float("-inf")
            for cc in range(int(c)):
                for zz in range(int(z0), int(z1)):
                    for yy in range(int(y0), int(y1)):
                        row = volume[cc][zz][yy]  # type: ignore[index]
                        for xx in range(int(x0), int(x1)):
                            v = float(row[xx])
                            n += 1
                            delta = v - mean
                            mean += delta / float(n)
                            delta2 = v - mean
                            m2 += delta * delta2
                            if v > vmax:
                                vmax = float(v)
            if n <= 0:
                return 0.0, 0.0, 0.0
            var = float(m2 / float(n))
            if vmax == float("-inf"):
                vmax = 0.0
            return float(mean), float(var), float(vmax)

        pressure: Dict[int, float] = {}
        for c in citations:
            for tid in c.cited_token_ids:
                pressure[int(tid)] = pressure.get(int(tid), 0.0) + 1.0

        feats: List[TokenFeatures] = []
        for t in tokens:
            x0, x1, y0, y1, z0, z1 = [float(x) for x in t.omega_box_mm]
            mean_int, var, vmax = _patch_variance(t)
            feats.append(
                TokenFeatures(
                    token_id=int(t.token_id),
                    level=int(t.level),
                    recon_error=float(var),
                    evidence_entropy=float(math.log1p(max(0.0, float(var)))),
                    citation_pressure=float(pressure.get(int(t.token_id), 0.0)),
                    history_splits=int(len(t.children_ids)),
                    center_x_mm=float((x0 + x1) / 2.0),
                    center_y_mm=float((y0 + y1) / 2.0),
                    center_z_mm=float((z0 + z1) / 2.0),
                    mean_intensity=float(mean_int),
                    max_intensity=float(vmax),
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
    ) -> Tuple[List[Token], List[int]]:
        # Replace selected parent tokens with their children (1 level deeper), respecting the budget.
        out: List[Token] = list(tokens)
        active_ids = {int(t.token_id) for t in out}
        executed: List[int] = []

        for pid in split_ids:
            pid = int(pid)
            if pid not in active_ids:
                continue
            child_ids = list(pyramid.children_map.get(pid, []))
            if not child_ids:
                continue  # leaf

            child_tokens = [pyramid.token_by_id[cid] for cid in child_ids if cid in pyramid.token_by_id]
            if not child_tokens:
                continue

            # Net token increase: +len(children)-1 (parent removed).
            new_count = len(out) - 1 + len(child_tokens)
            if int(new_count) > int(budget_B):
                continue

            out = [t for t in out if int(t.token_id) != pid]
            active_ids.remove(pid)
            for ct in child_tokens:
                if int(ct.token_id) in active_ids:
                    continue
                out.append(ct)
                active_ids.add(int(ct.token_id))

            executed.append(pid)

        out.sort(key=lambda t: int(t.token_id))
        return out, executed

    def _stop(self, score_before: float, score_after: float, tokens_before: int, tokens_after: int) -> bool:
        if tokens_after == tokens_before and score_after <= score_before:
            return True

        refine_cfg = self.cfg.get("refine", {})
        if not isinstance(refine_cfg, dict):
            refine_cfg = {}

        score_delta = float(score_after) - float(score_before)
        token_delta = int(tokens_after) - int(tokens_before)

        # docs/plan.md §3.7: stop if marginal gain per added token is too small.
        tau_raw = refine_cfg.get("tau", None)
        if tau_raw is not None:
            try:
                tau = float(tau_raw)
            except Exception:
                tau = None
            if tau is not None:
                den = max(1, int(token_delta))
                marginal = float(score_delta) / float(den)
                if float(marginal) < float(tau):
                    return True

        # Secondary absolute guard (kept for backward-compat with existing configs).
        min_delta = float(refine_cfg.get("min_score_delta", 1e-6))
        if float(score_delta) < float(min_delta):
            return True
        return False

    def _dump_artifacts(
        self,
        *,
        budget_B: int,
        tokens_used: int,
        verifier_score: float,
        tokens: List[Token],
        report: str,
        citations: List[Citation],
        plan: ReportPlan,
        evidence_nodes: List[EvidenceNode],
        trace: List[TraceStep],
        issues: List[Issue],
        latency_ms: Dict[str, float],
    ) -> Dict[str, Any]:
        return {
            "meta": {
                "policy": getattr(self.policy, "export", lambda: {})(),
                "tokenizer": getattr(self.tokenizer, "export", lambda: {})(),
                "evidence": getattr(self.evidence_head, "export", lambda: {})(),
            },
            "budget_B": int(budget_B),
            "tokens_used": int(tokens_used),
            "verifier_score": float(verifier_score),
            "tokens": [asdict(t) for t in tokens],
            "latency_ms": dict(latency_ms),
            "report": report,
            "citations": [asdict(c) for c in citations],
            "evidence_nodes": [asdict(n) for n in (evidence_nodes or [])],
            "plan": asdict(plan),
            "trace": [asdict(t) for t in trace],
            "issues": [asdict(i) for i in issues],
        }


__all__ = [
    "RefineRunner",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")


def _write_sidecar_artifacts(out_dir: Path, run: Dict[str, Any]) -> None:
    """
    Plan A0.3 expects 3 "sidecar" artifacts in addition to run.json:
      - final_report.txt
      - evidence_graph.json
      - trace.jsonl

    We keep run.json as the primary contract (see docs/results_contract.md) and
    derive sidecars additively for downstream tooling / paper-facing exports.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    report = str(run.get("report", ""))
    (out_dir / "final_report.txt").write_text(report, encoding="utf-8")

    meta = run.get("meta", {})
    inp = meta.get("input", {}) if isinstance(meta, dict) else {}
    case_id = str(run.get("case_id", "")).strip() or str(inp.get("case_id", "")).strip() or "case-0000"

    tokens = run.get("tokens", [])
    token_omega_index: Dict[str, Any] = {}
    if isinstance(tokens, list):
        for t in tokens:
            if not isinstance(t, dict):
                continue
            tid = t.get("token_id", None)
            box = t.get("omega_box_mm", None)
            if tid is None:
                continue
            if not (isinstance(box, (list, tuple)) and len(box) == 6):
                continue
            token_omega_index[str(int(tid))] = [float(x) for x in box]

    evidence_graph = {
        "case_id": case_id,
        "tokens": tokens,
        "token_omega_index": token_omega_index,
        "evidence_nodes": run.get("evidence_nodes", []),
        "plan": run.get("plan", {}),
        "citations": run.get("citations", []),
    }
    (out_dir / "evidence_graph.json").write_text(
        json.dumps(evidence_graph, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    trace_rows = run.get("trace", [])
    if not isinstance(trace_rows, list):
        trace_rows = []
    (out_dir / "trace.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in trace_rows),
        encoding="utf-8",
    )


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})

def _normalize_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize paper-facing config templates (from docs/plan.md) into the keys used by
    this repo-skeleton runner, without breaking existing internal configs.
    """

    out = dict(cfg or {})

    # docs/plan.md §4.4: generation.require_citation
    gen = out.get("generation", None)
    if isinstance(gen, dict) and "require_citation" in gen:
        require = bool(gen.get("require_citation", True))

        gates = out.get("gates", {})
        if not isinstance(gates, dict):
            gates = {}
        gates = dict(gates)
        gates.setdefault("require_citations", require)
        out["gates"] = gates

        realizer = out.get("realizer", {})
        if not isinstance(realizer, dict):
            realizer = {}
        realizer = dict(realizer)
        realizer.setdefault("emit_citations", require)
        out["realizer"] = realizer

    # docs/plan.md §4.4: verifier alpha/beta/gamma/delta -> weights mapping.
    verifier = out.get("verifier", {})
    if isinstance(verifier, dict) and any(k in verifier for k in ["alpha", "beta", "gamma", "delta"]):
        v = dict(verifier)
        weights = v.get("weights", {})
        if not isinstance(weights, dict):
            weights = {}
        weights = dict(weights)
        if "missing_slot" not in weights and "alpha" in v:
            weights["missing_slot"] = v.get("alpha")
        if "inconsistency" not in weights and "beta" in v:
            weights["inconsistency"] = v.get("beta")
        if "overclaim" not in weights and "gamma" in v:
            weights["overclaim"] = v.get("gamma")
        if "unsupported" not in weights and "delta" in v:
            weights["unsupported"] = v.get("delta")
        v["weights"] = weights
        out["verifier"] = v

    return out


def _make_dummy_volume(shape_cdhw: Sequence[int]) -> List[List[List[List[float]]]]:
    return _make_dummy_volume_with_cfg(shape_cdhw, seed=0, pattern="zeros", pattern_cfg=None)


def _make_dummy_volume_with_cfg(
    shape_cdhw: Sequence[int],
    *,
    seed: int,
    pattern: str,
    pattern_cfg: Dict[str, Any] | None = None,
) -> List[List[List[List[float]]]]:
    c, d, h, w = [int(x) for x in shape_cdhw]
    pattern = str(pattern or "zeros").strip().lower()
    pattern_cfg = dict(pattern_cfg or {})

    if pattern == "noise":
        import random

        rng = random.Random(int(seed))
        return [[[[float(rng.random()) for _ in range(w)] for _ in range(h)] for _ in range(d)] for _ in range(c)]

    if pattern == "region_noise":
        import random

        rng = random.Random(int(seed))
        region_patch = int(pattern_cfg.get("region_patch", max(1, min(int(w), int(h), int(d)) // 2)))
        region_patch = max(1, int(region_patch))
        high = float(pattern_cfg.get("high_scale", 1.0))
        low = float(pattern_cfg.get("low_scale", 0.01))

        nx = max(1, (int(w) + int(region_patch) - 1) // int(region_patch))
        ny = max(1, (int(h) + int(region_patch) - 1) // int(region_patch))

        out: List[List[List[List[float]]]] = []
        for cc in range(int(c)):
            ch: List[List[List[float]]] = []
            for zz in range(int(d)):
                slab: List[List[float]] = []
                rz = int(zz) // int(region_patch)
                for yy in range(int(h)):
                    row: List[float] = []
                    ry = int(yy) // int(region_patch)
                    for xx in range(int(w)):
                        rx = int(xx) // int(region_patch)
                        rid = int(rz) * int(ny) * int(nx) + int(ry) * int(nx) + int(rx)
                        scale = float(high) if (int(rid) % 2 == 0) else float(low)
                        row.append(float(rid) + float(scale) * float(rng.random()))
                    slab.append(row)
                ch.append(slab)
            out.append(ch)
        return out

    if pattern == "gradient":
        out: List[List[List[List[float]]]] = []
        for cc in range(c):
            ch: List[List[List[float]]] = []
            for zz in range(d):
                slab: List[List[float]] = []
                for yy in range(h):
                    row: List[float] = []
                    for xx in range(w):
                        v = 0.0
                        if w > 1:
                            v += float(xx) / float(w - 1)
                        if h > 1:
                            v += float(yy) / float(h - 1)
                        if d > 1:
                            v += float(zz) / float(d - 1)
                        row.append(float(v))
                    slab.append(row)
                ch.append(slab)
            out.append(ch)
        return out

    # Default: zeros
    return [[[[0.0 for _ in range(w)] for _ in range(h)] for _ in range(d)] for _ in range(c)]


def _stable_int_hash(s: str) -> int:
    # Deterministic across processes (avoid Python's randomized hash()).
    h = 0
    for ch in str(s):
        h = (h * 131 + ord(ch)) % 2147483647
    return int(h)


def _load_manifest_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _select_manifest_row(rows: List[Dict[str, Any]], case_id: str | None) -> Dict[str, Any]:
    if case_id:
        want = str(case_id).strip()
        for row in rows:
            if str(row.get("case_id", "")).strip() == want:
                return row
        raise KeyError(f"case_id not found in manifest: {want}")
    if not rows:
        raise ValueError("manifest has no rows")
    return rows[0]


def _target_shape_cdhw(cfg: Dict[str, Any]) -> List[int]:
    vol_cfg = dict(cfg.get("volume", {}))
    target = vol_cfg.get("target_shape_cdhw") or vol_cfg.get("shape_cdhw") or [1, 8, 8, 8]
    try:
        c, d, h, w = [int(x) for x in target]
    except Exception:
        return [1, 8, 8, 8]
    return [max(1, c), max(1, d), max(1, h), max(1, w)]


def _load_nifti_volume_small_and_steps(path: Path, target_cdhw: Sequence[int]) -> Tuple[Any, List[int]]:
    """
    Load a very small (C,D,H,W) float32 volume from a potentially huge NIfTI.
    Uses strided slicing so we don't materialize the full array.
    """
    try:
        import nibabel as nib  # type: ignore
        import numpy as np  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("nibabel+numpy are required to load .nii.gz volumes") from exc

    img = nib.load(str(path))
    shape = tuple(int(x) for x in (img.shape[:3] if hasattr(img, "shape") else ()))
    if len(shape) != 3:
        raise ValueError(f"unsupported NIfTI shape: {getattr(img, 'shape', None)}")
    sx, sy, sz = shape  # (X,Y,Z)

    c_t, d_t, h_t, w_t = [int(x) for x in target_cdhw]
    if c_t != 1:
        # Repo skeleton is single-channel; keep it simple (pad/truncate to 1).
        c_t = 1

    # Compute simple strided sampling.
    step_x = max(1, sx // max(1, int(w_t)))
    step_y = max(1, sy // max(1, int(h_t)))
    step_z = max(1, sz // max(1, int(d_t)))
    slc = img.dataobj[0:sx:step_x, 0:sy:step_y, 0:sz:step_z]
    arr = np.asarray(slc, dtype=np.float32)

    # Pad/crop to exact target (X,Y,Z).
    out_xyz = np.zeros((int(w_t), int(h_t), int(d_t)), dtype=np.float32)
    x = min(int(w_t), int(arr.shape[0]))
    y = min(int(h_t), int(arr.shape[1]))
    z = min(int(d_t), int(arr.shape[2]))
    out_xyz[:x, :y, :z] = arr[:x, :y, :z]

    # Convert to (C,D,H,W) with D==Z, H==Y, W==X.
    out_dhw = np.transpose(out_xyz, (2, 1, 0))  # (Z,Y,X)
    vol = out_dhw[None, ...]  # (1,Z,Y,X)
    return vol, [int(step_x), int(step_y), int(step_z)]


def _load_nifti_volume_small(path: Path, target_cdhw: Sequence[int]) -> Any:
    vol, _steps = _load_nifti_volume_small_and_steps(path, target_cdhw)
    return vol


def _load_volume_for_manifest_row(row: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[Tensor, Dict[str, Any]]:
    case_id = str(row.get("case_id", "")).strip() or "case-0000"
    vol_path = str(row.get("volume_path", "")).strip()
    rpt_path = str(row.get("report_path", "")).strip()

    target = _target_shape_cdhw(cfg)
    vol_cfg = cfg.get("volume", {})
    if not isinstance(vol_cfg, dict):
        vol_cfg = {}
    vol_cfg = dict(vol_cfg)
    dummy_pattern = str(vol_cfg.get("pattern", "gradient") or "gradient").strip().lower()

    loader = "dummy"
    volume: Tensor
    volume_shape_xyz = None
    downsample_steps_xyz = None
    if vol_path and Path(vol_path).exists():
        loader = "nifti"
        try:
            volume, downsample_steps_xyz = _load_nifti_volume_small_and_steps(Path(vol_path), target)
            # Best-effort read original shape for audit.
            try:
                import nibabel as nib  # type: ignore

                volume_shape_xyz = list(getattr(nib.load(str(vol_path)), "shape", ())[:3])
            except Exception:
                volume_shape_xyz = None
        except Exception:
            # Fall back to dummy volume; still record the requested volume_path.
            loader = "dummy"
            volume_shape_xyz = None
            downsample_steps_xyz = None
            volume = _make_dummy_volume_with_cfg(target, seed=_stable_int_hash(case_id), pattern=dummy_pattern, pattern_cfg=vol_cfg)
    else:
        volume = _make_dummy_volume_with_cfg(target, seed=_stable_int_hash(case_id), pattern=dummy_pattern, pattern_cfg=vol_cfg)

    meta = {
        "case_id": case_id,
        "volume_path": vol_path,
        "report_path": rpt_path,
        "volume_loader": loader,
        "target_shape_cdhw": list(target),
        "volume_shape_xyz": volume_shape_xyz,
        "downsample_steps_xyz": downsample_steps_xyz,
    }
    return volume, meta


def run_infer_refine(cfg: Dict[str, Any], *, budget_B: int) -> Dict[str, Any]:
    tokenizer = Tokenizer3D(dict(cfg.get("tokenizer", {})))
    evidence_head = EvidenceHead(dict(cfg.get("evidence", {})))
    planner = Planner(dict(cfg.get("planner", {})))
    realizer = Realizer(dict(cfg.get("realizer", {})))
    verifier = Verifier(dict(cfg.get("verifier", {})))
    policy = SplitPolicy(dict(cfg.get("policy", {})))

    runner = RefineRunner(tokenizer, evidence_head, planner, realizer, verifier, policy, cfg)

    manifest_jsonl = cfg.get("_manifest_jsonl")
    manifest_case_id = cfg.get("_manifest_case_id")
    if manifest_jsonl:
        mp = Path(str(manifest_jsonl))
        rows = _load_manifest_jsonl(mp)
        row = _select_manifest_row(rows, str(manifest_case_id) if manifest_case_id else None)
        volume, input_meta = _load_volume_for_manifest_row(row, cfg)
        run = runner.run_case(volume, budget_B=int(budget_B))
        # Store selected case_id for downstream evaluation/debug.
        run["case_id"] = str(input_meta.get("case_id", "case-0000"))
        meta = run.get("meta", {})
        if isinstance(meta, dict):
            meta["input"] = dict(input_meta)
            run["meta"] = meta
        return run

    vol_cfg = dict(cfg.get("volume", {}))
    shape = vol_cfg.get("shape_cdhw", [1, 8, 8, 8])
    seed = int(vol_cfg.get("seed", 0))
    pattern = str(vol_cfg.get("pattern", "zeros"))
    volume = _make_dummy_volume_with_cfg(shape, seed=seed, pattern=pattern, pattern_cfg=vol_cfg)

    return runner.run_case(volume, budget_B=int(budget_B))


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference refine loop (minimal baseline).")
    parser.add_argument(
        "--out",
        default="",
        help="Output directory (default: <cfg.output.save_dir>/<run_id> or artifacts/runs/<timestamp>).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Token budget B (default: cfg.refine.budget_B or 16).",
    )
    parser.add_argument("--config", default="voxtoken/configs/inference.yaml", help="Path to YAML config")
    parser.add_argument("--manifest", default=None, help="Optional JSONL manifest; select a case to run")
    parser.add_argument("--case-id", default=None, help="Case ID to select from the manifest (defaults to first row)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = _load_yaml(cfg_path) if cfg_path.exists() else {}
    cfg = dict(cfg)
    cfg = _normalize_cfg(cfg)

    budget_B = args.budget
    if budget_B is None:
        refine_cfg = cfg.get("refine", {}) if isinstance(cfg.get("refine", {}), dict) else {}
        if isinstance(refine_cfg, dict) and "budget_B" in refine_cfg:
            try:
                budget_B = int(refine_cfg.get("budget_B", 16))
            except Exception:
                budget_B = 16
        else:
            budget_B = 16

    if args.manifest:
        cfg["_manifest_jsonl"] = str(Path(str(args.manifest)).expanduser())
        if args.case_id:
            cfg["_manifest_case_id"] = str(args.case_id).strip()

    out_arg = str(getattr(args, "out", "") or "").strip()
    if out_arg:
        out_dir = Path(out_arg)
    else:
        out_cfg = cfg.get("output", {}) if isinstance(cfg.get("output", {}), dict) else {}
        save_dir = str(out_cfg.get("save_dir", "") or "").strip() or "artifacts/runs"
        run_id = str(cfg.get("run_id", "") or "").strip() or _utc_now_compact()
        out_dir = Path(save_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    run = run_infer_refine(cfg, budget_B=int(budget_B))

    # Inject minimal meta summary for audit.
    summary = {
        "timestamp_utc": _utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "config_path": str(cfg_path),
        "budget_B": int(budget_B),
    }
    if isinstance(run, dict) and "case_id" in run:
        summary["case_id"] = str(run.get("case_id", ""))

    (out_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_sidecar_artifacts(out_dir, run)

    print(json.dumps({"out_dir": str(out_dir), "run_path": str(out_dir / "run.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
