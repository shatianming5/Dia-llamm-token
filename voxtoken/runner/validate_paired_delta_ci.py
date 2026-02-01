from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl_many(paths: List[Path]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _get_metric(row: Dict[str, Any], key: str) -> float:
    key = str(key)
    if "." not in key:
        try:
            return float(row.get(key, 0.0))
        except Exception:
            return 0.0
    cur: Any = row
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return 0.0
        cur = cur[part]
    try:
        return float(cur)
    except Exception:
        return 0.0


def _mean(vals: List[float]) -> float:
    return float(sum(vals) / float(len(vals))) if vals else 0.0


def _quantile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    q = float(q)
    q = 0.0 if q < 0.0 else (1.0 if q > 1.0 else q)
    idx = int(round(q * float(len(sorted_vals) - 1)))
    idx = max(0, min(int(idx), int(len(sorted_vals) - 1)))
    return float(sorted_vals[idx])


def _paired_deltas(
    rows: List[Dict[str, Any]],
    *,
    budget_B: int,
    lhs: str,
    rhs: str,
    metric: str,
) -> List[float]:
    by: Dict[Tuple[str, str, int], List[float]] = {}
    for r in rows:
        try:
            b = int(r.get("budget_B", 0) or 0)
        except Exception:
            continue
        if int(b) != int(budget_B):
            continue
        case_id = str(r.get("case_id", "")).strip()
        method = str(r.get("method", "")).strip()
        if not case_id or not method:
            continue
        v = float(_get_metric(r, metric))
        by.setdefault((case_id, method, int(budget_B)), []).append(float(v))

    cases = sorted({k[0] for k in by.keys()})
    deltas: List[float] = []
    for c in cases:
        lv = by.get((c, str(lhs), int(budget_B)))
        rv = by.get((c, str(rhs), int(budget_B)))
        if not lv or not rv:
            continue
        deltas.append(float(_mean(lv)) - float(_mean(rv)))
    return deltas


def _bootstrap_ci_mean(vals: List[float], *, n_boot: int, alpha: float, seed: int) -> Dict[str, float]:
    if not vals:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    n_boot = max(200, int(n_boot))
    alpha = float(alpha)
    alpha = 0.0 if alpha < 0.0 else (1.0 if alpha > 1.0 else alpha)

    n = int(len(vals))
    rng = random.Random(int(seed))
    boot: List[float] = []
    for _ in range(int(n_boot)):
        s = 0.0
        for _k in range(n):
            s += float(vals[rng.randrange(n)])
        boot.append(float(s) / float(n))
    boot.sort()
    return {
        "mean": float(_mean(vals)),
        "ci_low": float(_quantile(boot, float(alpha) / 2.0)),
        "ci_high": float(_quantile(boot, 1.0 - float(alpha) / 2.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a paired delta CI gate by bootstrapping over per-case deltas (lhs-rhs)."
    )
    parser.add_argument(
        "--in",
        dest="inputs",
        action="append",
        nargs="+",
        required=True,
        help="Input metrics.jsonl path(s). Repeatable, like paper_export: --in a.jsonl --in b.jsonl ...",
    )
    parser.add_argument("--budget", type=int, required=True, help="Budget_B to evaluate (e.g., 32)")
    parser.add_argument("--lhs", default="learned", help="Method A (default: learned)")
    parser.add_argument("--rhs", default="random", help="Method B (default: random)")
    parser.add_argument("--metric", default="ground_mean_iou", help="Metric key (supports dot paths)")
    parser.add_argument("--n-boot", type=int, default=4000, help="Bootstrap replicates (default: 4000)")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha for (1-alpha) CI (default: 0.05 -> 95%% CI)")
    parser.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed (default: 0)")
    parser.add_argument("--delta-ge", type=float, default=0.0, help="Require mean(delta) >= this threshold")
    parser.add_argument("--ci-low-ge", type=float, default=0.0, help="Require CI_low(delta) >= this threshold (default: 0.0)")
    args = parser.parse_args()

    raw_inputs = args.inputs or []
    flat_inputs: List[str] = []
    for group in raw_inputs:
        if isinstance(group, list):
            flat_inputs.extend([str(x) for x in group])
        else:
            flat_inputs.append(str(group))
    paths = [Path(str(p)).expanduser() for p in flat_inputs]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        print(f"[ERR] missing inputs: {missing}", file=sys.stderr)
        sys.exit(2)

    rows = _load_jsonl_many(paths)
    if not rows:
        print("[ERR] empty inputs (no JSONL rows)", file=sys.stderr)
        sys.exit(1)

    budget_B = int(args.budget)
    lhs = str(args.lhs).strip()
    rhs = str(args.rhs).strip()
    metric = str(args.metric).strip()

    deltas = _paired_deltas(rows, budget_B=int(budget_B), lhs=str(lhs), rhs=str(rhs), metric=str(metric))
    if not deltas:
        print("[ERR] no paired cases with both lhs and rhs present", file=sys.stderr)
        sys.exit(1)

    stats = _bootstrap_ci_mean(deltas, n_boot=int(args.n_boot), alpha=float(args.alpha), seed=int(args.seed))
    mean = float(stats["mean"])
    lo = float(stats["ci_low"])
    hi = float(stats["ci_high"])

    thr_delta = float(args.delta_ge)
    thr_lo = float(args.ci_low_ge)
    ok_delta = bool(mean >= thr_delta)
    ok_ci = bool(lo >= thr_lo)

    if not (ok_delta and ok_ci):
        reasons: list[str] = []
        if not ok_delta:
            reasons.append(f"mean_delta {mean:.6f} < {thr_delta:.6f}")
        if not ok_ci:
            reasons.append(f"delta_ci_low {lo:.6f} < {thr_lo:.6f}")
        print(
            "[ERR] paired delta CI gate failed: "
            f"budget_B={budget_B} metric={metric} lhs={lhs} rhs={rhs} "
            f"n_cases={len(deltas)} mean_delta={mean:.6f} ci=[{lo:.6f},{hi:.6f}] :: " + "; ".join(reasons),
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "[OK] paired delta CI gate passed: "
        f"budget_B={budget_B} metric={metric} lhs={lhs} rhs={rhs} "
        f"n_cases={len(deltas)} mean_delta={mean:.6f} ci=[{lo:.6f},{hi:.6f}]"
    )


if __name__ == "__main__":
    main()
