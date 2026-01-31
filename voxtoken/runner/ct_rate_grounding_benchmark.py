from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence, Tuple

from ..models.policy import SplitPolicy
from ..models.tokenizer import TokenPyramid, Tokenizer3D
from ..schemas import Token, TokenFeatures
from .infer_refine import _load_volume_for_manifest_row


Box = Tuple[float, float, float, float, float, float]  # x0,x1,y0,y1,z0,z1 (mm)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_int_hash(s: str) -> int:
    h = 0
    for ch in str(s):
        h = (h * 131 + ord(ch)) % 2147483647
    return int(h)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml  # pyyaml

    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(cfg or {})


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError("manifest rows must be JSON objects")
        rows.append(obj)
    return rows


def _parse_gt_boxes_sent0(row: Dict[str, Any]) -> List[Box]:
    raw = row.get("grounding_boxes_by_sent_mm", {}) or row.get("gt_boxes_by_sent_mm", {})
    if not isinstance(raw, dict):
        return []
    v = raw.get("0", None) or raw.get(0, None)
    if not isinstance(v, list):
        return []
    out: List[Box] = []
    for b in v:
        if not isinstance(b, (list, tuple)) or len(b) != 6:
            continue
        try:
            out.append(tuple(float(x) for x in b))  # type: ignore[assignment]
        except Exception:
            continue
    return out


def _box_iou_3d(a: Box, b: Box) -> float:
    ax0, ax1, ay0, ay1, az0, az1 = [float(x) for x in a]
    bx0, bx1, by0, by1, bz0, bz1 = [float(x) for x in b]

    ix0 = max(ax0, bx0)
    ix1 = min(ax1, bx1)
    iy0 = max(ay0, by0)
    iy1 = min(ay1, by1)
    iz0 = max(az0, bz0)
    iz1 = min(az1, bz1)

    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0) * max(0.0, iz1 - iz0)
    if inter <= 0.0:
        return 0.0

    va = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0) * max(0.0, az1 - az0)
    vb = max(0.0, bx1 - bx0) * max(0.0, by1 - by0) * max(0.0, bz1 - bz0)
    union = va + vb - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _token_feature_vector(
    token: Token,
    *,
    volume: Any,
    voxel_spacing_mm: List[float],
) -> TokenFeatures:
    sx, sy, sz = [float(x) for x in (voxel_spacing_mm or [1.0, 1.0, 1.0])]

    def _shape_cdhw(vol: Any) -> Tuple[int, int, int, int]:
        try:
            c = len(vol)
            d = len(vol[0])
            h = len(vol[0][0])
            w = len(vol[0][0][0])
            return int(c), int(d), int(h), int(w)
        except Exception:
            return 1, 1, 1, 1

    def _box_mm_to_zyx_slices(box_mm: Box) -> Tuple[int, int, int, int, int, int]:
        x0, x1, y0, y1, z0, z1 = box_mm
        ix0 = int(round(float(x0) / sx))
        ix1 = int(round(float(x1) / sx))
        iy0 = int(round(float(y0) / sy))
        iy1 = int(round(float(y1) / sy))
        iz0 = int(round(float(z0) / sz))
        iz1 = int(round(float(z1) / sz))
        return iz0, iz1, iy0, iy1, ix0, ix1

    def _patch_variance() -> float:
        c, d, h, w = _shape_cdhw(volume)
        z0, z1, y0, y1, x0, x1 = _box_mm_to_zyx_slices(token.omega_box_mm)
        z0 = max(0, min(int(z0), int(d)))
        z1 = max(0, min(int(z1), int(d)))
        y0 = max(0, min(int(y0), int(h)))
        y1 = max(0, min(int(y1), int(h)))
        x0 = max(0, min(int(x0), int(w)))
        x1 = max(0, min(int(x1), int(w)))
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return 0.0

        n = 0
        mean = 0.0
        m2 = 0.0
        for cc in range(int(c)):
            for zz in range(int(z0), int(z1)):
                for yy in range(int(y0), int(y1)):
                    row = volume[cc][zz][yy]
                    for xx in range(int(x0), int(x1)):
                        v = float(row[xx])
                        n += 1
                        delta = v - mean
                        mean += delta / float(n)
                        delta2 = v - mean
                        m2 += delta * delta2
        if n <= 0:
            return 0.0
        return float(m2 / float(n))

    var = float(_patch_variance())
    return TokenFeatures(
        token_id=int(token.token_id),
        level=int(token.level),
        recon_error=float(var),
        evidence_entropy=float(math.log1p(max(0.0, float(var)))),
        citation_pressure=0.0,
        history_splits=int(len(token.children_ids)),
    )


def _select_splits(policy: SplitPolicy, feats: List[TokenFeatures], *, budget_left: int, max_splits_per_round: int) -> List[int]:
    if budget_left <= 0:
        return []
    max_splits = max(0, min(int(budget_left), int(max_splits_per_round)))
    if max_splits <= 0:
        return []
    scored = policy.score(feats)
    return [int(tid) for tid, _ in scored[:max_splits]]


def _select_splits_random(
    pyramid: TokenPyramid,
    tokens: List[Token],
    *,
    rng: random.Random,
    budget_left: int,
    max_splits_per_round: int,
) -> List[int]:
    if budget_left <= 0:
        return []
    max_splits = max(0, min(int(budget_left), int(max_splits_per_round)))
    if max_splits <= 0:
        return []
    cands = [int(t.token_id) for t in tokens if pyramid.children_map.get(int(t.token_id), [])]
    rng.shuffle(cands)
    return [int(x) for x in cands[:max_splits]]


def _refine_select(pyramid: TokenPyramid, tokens: List[Token], split_ids: Sequence[int], budget_B: int) -> Tuple[List[Token], List[int]]:
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
            continue

        child_tokens = [pyramid.token_by_id[cid] for cid in child_ids if cid in pyramid.token_by_id]
        if not child_tokens:
            continue

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


def _best_iou_and_token(tokens: List[Token], gt_boxes: List[Box]) -> Tuple[float, int | None, Box | None]:
    if not tokens or not gt_boxes:
        return 0.0, None, None

    # Deterministic: iterate tokens in order and keep the first best (no tie-breaking needed).
    best = -1.0
    best_tid: int | None = None
    best_box: Box | None = None
    for t in tokens:
        tb = t.omega_box_mm
        token_best = 0.0
        for gb in gt_boxes:
            token_best = max(token_best, float(_box_iou_3d(tb, gb)))
        if token_best > best:
            best = float(token_best)
            best_tid = int(t.token_id)
            best_box = tb
    return float(max(0.0, best)), best_tid, best_box


def _refine_select_oracle(
    pyramid: TokenPyramid,
    tokens: List[Token],
    *,
    gt_boxes: List[Box],
    budget_B: int,
    max_splits: int,
) -> Tuple[List[Token], List[int]]:
    tk = list(tokens)
    executed: List[int] = []

    for _i in range(max(0, int(max_splits))):
        base_iou, _tid, _box = _best_iou_and_token(tk, gt_boxes)
        best_pid: int | None = None
        best_impr = 0.0

        for t in tk:
            pid = int(t.token_id)
            child_ids = list(pyramid.children_map.get(pid, []))
            if not child_ids:
                continue

            new_count = int(len(tk) - 1 + len(child_ids))
            if new_count > int(budget_B):
                continue

            tk_candidate = [x for x in tk if int(x.token_id) != pid] + [pyramid.token_by_id[cid] for cid in child_ids if cid in pyramid.token_by_id]
            cand_iou, _ctid, _cbox = _best_iou_and_token(tk_candidate, gt_boxes)
            impr = float(cand_iou) - float(base_iou)

            if impr > best_impr + 1e-12:
                best_impr = float(impr)
                best_pid = int(pid)
            elif abs(impr - best_impr) <= 1e-12 and best_pid is not None and int(pid) < int(best_pid):
                best_pid = int(pid)

        if best_pid is None or float(best_impr) <= 0.0:
            break

        tk2, ex = _refine_select(pyramid, tk, [int(best_pid)], int(budget_B))
        if not ex or len(tk2) == len(tk):
            break
        executed.extend(list(ex))
        tk = tk2

    return tk, executed


def ct_rate_grounding_benchmark(
    *,
    manifest_jsonl: Path,
    out_dir: Path,
    cfg: Dict[str, Any],
    budgets: List[int],
    max_cases: int,
    split: str,
    policy_ckpt: str,
    seed: int,
) -> Dict[str, Any]:
    rows = _load_jsonl(manifest_jsonl)
    rows.sort(key=lambda r: str(r.get("case_id", "")).strip())
    split_norm = str(split or "").strip()
    if split_norm:
        rows = [r for r in rows if str(r.get("split", "")).strip() == split_norm]
    if max_cases > 0:
        rows = rows[: int(max_cases)]

    tokenizer_cfg = dict(cfg.get("tokenizer", {}))
    refine_cfg = cfg.get("refine", {})
    if not isinstance(refine_cfg, dict):
        refine_cfg = {}
    max_rounds = int(refine_cfg.get("max_rounds", 0))
    max_splits_per_round = int(refine_cfg.get("max_splits_per_round", 0)) or 0

    policy_cfg = cfg.get("policy", {})
    if not isinstance(policy_cfg, dict):
        policy_cfg = {}
    weights = dict(policy_cfg.get("weights", {}))

    budgets_sorted = sorted({int(b) for b in budgets if int(b) > 0})
    if not budgets_sorted:
        raise ValueError("no positive budgets provided")

    metrics_rows: List[Dict[str, Any]] = []

    for row in rows:
        case_id = str(row.get("case_id", "")).strip()
        if not case_id:
            continue
        gt_boxes = _parse_gt_boxes_sent0(row)
        if not gt_boxes:
            continue

        volume, input_meta = _load_volume_for_manifest_row(row, cfg)
        volume_loader = str((input_meta or {}).get("volume_loader", "unknown"))

        tokenizer = Tokenizer3D(tokenizer_cfg)
        spacing = tokenizer.cfg.get("voxel_spacing_mm", [1.0, 1.0, 1.0])
        if not isinstance(spacing, list) or len(spacing) != 3:
            spacing = [1.0, 1.0, 1.0]
        spacing = [float(x) for x in spacing]

        t_build0 = time.perf_counter()
        pyramid: TokenPyramid = tokenizer.build_pyramid(volume)
        t_build1 = time.perf_counter()
        build_ms = float((t_build1 - t_build0) * 1000.0)

        pol_heuristic = SplitPolicy({"mode": "heuristic", "weights": dict(weights)})
        ckpt_path = str(policy_ckpt).strip()
        pol_learned = SplitPolicy({"mode": "heuristic", "weights": dict(weights), "checkpoint_path": ckpt_path}) if ckpt_path else pol_heuristic

        for budget_B in budgets_sorted:
            base_tokens = tokenizer.select_tokens(pyramid, active_nodes=[], budget_B=int(budget_B))

            for method in ["fixed", "heuristic", "learned", "random", "oracle"]:
                t0 = time.perf_counter()
                tk = list(base_tokens)

                executed_split_ids: List[int] = []
                rng = (
                    random.Random(int(seed) + _stable_int_hash(f"{case_id}:{budget_B}:random"))
                    if method == "random"
                    else None
                )
                if max_rounds > 0:
                    for _k in range(int(max_rounds)):
                        budget_left = int(budget_B) - int(len(tk))
                        max_splits = int(max_splits_per_round or max(0, budget_left))

                        tk2: List[Token]
                        executed: List[int]
                        if method in {"heuristic", "learned"}:
                            pol = pol_learned if method == "learned" else pol_heuristic
                            feats = [_token_feature_vector(t, volume=volume, voxel_spacing_mm=spacing) for t in tk]
                            split_ids = _select_splits(
                                pol,
                                feats,
                                budget_left=int(budget_left),
                                max_splits_per_round=int(max_splits),
                            )
                            tk2, executed = _refine_select(pyramid, tk, split_ids, int(budget_B))
                        elif method == "random":
                            if rng is None:
                                break
                            split_ids = _select_splits_random(
                                pyramid,
                                tk,
                                rng=rng,
                                budget_left=int(budget_left),
                                max_splits_per_round=int(max_splits),
                            )
                            tk2, executed = _refine_select(pyramid, tk, split_ids, int(budget_B))
                        elif method == "oracle":
                            max_splits_eff = max(0, min(int(budget_left), int(max_splits)))
                            tk2, executed = _refine_select_oracle(
                                pyramid,
                                tk,
                                gt_boxes=list(gt_boxes),
                                budget_B=int(budget_B),
                                max_splits=int(max_splits_eff),
                            )
                        else:
                            tk2, executed = tk, []

                        executed_split_ids.extend(list(executed))
                        if len(tk2) == len(tk):
                            break
                        tk = tk2

                t1 = time.perf_counter()
                latency_ms_total = float(build_ms + (t1 - t0) * 1000.0)

                max_iou, best_tid, best_box = _best_iou_and_token(tk, gt_boxes)
                hit0 = int(bool(max_iou > 0.0))
                hit01 = int(bool(max_iou >= 0.1))

                metrics_rows.append(
                    {
                        "case_id": case_id,
                        "method": str(method),
                        "budget_B": int(budget_B),
                        "tokens_used": int(len(tk)),
                        "latency_ms": {"total": float(latency_ms_total)},
                        "ground_mean_iou": float(max_iou),
                        "ground_hit@0.0": int(hit0),
                        "ground_hit@0.1": int(hit01),
                        "gt_boxes_mm": [list(b) for b in gt_boxes],
                        "best_token_id": (int(best_tid) if best_tid is not None else None),
                        "best_token_box_mm": ([float(x) for x in best_box] if best_box is not None else None),
                        "executed_split_ids": list(executed_split_ids),
                        "volume_loader": volume_loader,
                        "seed": int(seed),
                    }
                )

    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in metrics_rows), encoding="utf-8")

    groups: DefaultDict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for r in metrics_rows:
        groups[(str(r.get("method", "")), int(r.get("budget_B", 0)))].append(r)

    def mean(vals: List[float]) -> float:
        return float(sum(vals) / float(len(vals))) if vals else 0.0

    group_rows: List[Dict[str, Any]] = []
    for (method, budget_B), rs in sorted(groups.items(), key=lambda x: (x[0][0], int(x[0][1]))):
        group_rows.append(
            {
                "method": str(method),
                "budget_B": int(budget_B),
                "n": int(len(rs)),
                "ground_mean_iou_mean": mean([float(x.get("ground_mean_iou", 0.0)) for x in rs]),
                "ground_hit@0.0_mean": mean([float(x.get("ground_hit@0.0", 0.0)) for x in rs]),
                "ground_hit@0.1_mean": mean([float(x.get("ground_hit@0.1", 0.0)) for x in rs]),
                "tokens_used_mean": mean([float(x.get("tokens_used", 0.0)) for x in rs]),
                "latency_ms.total_mean": mean([float((x.get("latency_ms") or {}).get("total", 0.0)) for x in rs]),
            }
        )

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "manifest": str(manifest_jsonl),
        "split": str(split_norm),
        "seed": int(seed),
        "out_dir": str(out_dir),
        "metrics_jsonl_path": str(metrics_path),
        "n_cases": int(len({str(r.get("case_id", "")) for r in metrics_rows if str(r.get("case_id", "")).strip()})),
        "groups": group_rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="CT-RATE TS grounding benchmark (tokenization vs GT boxes).")
    parser.add_argument("--manifest", required=True, help="Path to CT-RATE TS GT manifest.jsonl")
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--config", required=True, help="YAML config (ct_rate_ts_grounding_e0907.yaml)")
    parser.add_argument("--budgets", nargs="+", required=True, help="Budgets B (e.g., 8 16 32)")
    parser.add_argument("--max-cases", type=int, default=0, help="Max cases (0 = no limit)")
    parser.add_argument("--split", default="", help="Optional split filter (train|val|test)")
    parser.add_argument("--policy-ckpt", default="", help="Path to learned policy checkpoint.json")
    parser.add_argument("--seed", type=int, default=0, help="Random seed (affects random baseline and tie-breaking)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"[ERR] missing config: {cfg_path}", file=sys.stderr)
        sys.exit(2)
    cfg = _load_yaml(cfg_path)

    manifest = Path(args.manifest)
    if not manifest.exists():
        print(f"[ERR] missing manifest: {manifest}", file=sys.stderr)
        sys.exit(2)

    budgets = [int(x) for x in args.budgets]
    try:
        summary = ct_rate_grounding_benchmark(
            manifest_jsonl=manifest,
            out_dir=Path(args.out),
            cfg=cfg,
            budgets=budgets,
            max_cases=int(args.max_cases or 0),
            split=str(args.split),
            policy_ckpt=str(args.policy_ckpt),
            seed=int(args.seed),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[ERR] {exc}", file=sys.stderr)
        raise

    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
