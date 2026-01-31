from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List


REQUIRED_METRICS_KEYS = [
    "case_id",
    "method",
    "budget_B",
    "tokens_used",
    "latency_ms",
    "ground_mean_iou",
    "ground_hit@0.0",
    "ground_hit@0.1",
]


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError("summary.json must be a JSON object")
    return obj


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise TypeError("metrics.jsonl rows must be JSON objects")
        rows.append(obj)
    return rows


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


def validate_grounding_benchmark(summary: Dict[str, Any], *, require_methods: List[str]) -> List[str]:
    errors: List[str] = []

    metrics_path = str(summary.get("metrics_jsonl_path", "")).strip()
    if not metrics_path:
        errors.append("summary.metrics_jsonl_path missing")
        return errors

    mp = Path(metrics_path)
    if not mp.exists():
        errors.append(f"metrics_jsonl_path not found: {mp}")
        return errors

    rows = _load_jsonl(mp)
    if not rows:
        errors.append("metrics.jsonl is empty")
        return errors

    # Required methods gate via summary.groups (more stable than scanning full metrics).
    groups = summary.get("groups", [])
    if not isinstance(groups, list):
        groups = []
    seen = {str(g.get("method", "")).strip() for g in groups if isinstance(g, dict)}
    missing = [m for m in require_methods if str(m).strip() and str(m).strip() not in seen]
    if missing:
        errors.append(f"missing required methods in summary.groups: {missing}")

    for i, r in enumerate(rows):
        for k in REQUIRED_METRICS_KEYS:
            if k not in r:
                errors.append(f"metrics[{i}] missing key: {k}")
        lat = r.get("latency_ms", None)
        if not isinstance(lat, dict) or "total" not in lat:
            errors.append(f"metrics[{i}] latency_ms.total missing")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate CT-RATE grounding benchmark outputs.")
    parser.add_argument("--in", dest="summary_json", required=True, help="Path to summary.json")
    parser.add_argument("--require-methods", nargs="+", default=None, help="Required methods (e.g., fixed heuristic learned)")
    parser.add_argument(
        "--require-improvement",
        nargs=5,
        default=None,
        metavar=("METHOD_A", "METHOD_B", "METRIC_KEY", "DELTA_GE", "ALPHA"),
        help="Require method A improves over method B by >= DELTA_GE with (1-ALPHA) bootstrap CI lower bound >= DELTA_GE, for at least one budget.",
    )
    parser.add_argument("--bootstrap-n", type=int, default=2000, help="Bootstrap samples for improvement gate")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for bootstrap determinism")
    args = parser.parse_args()

    summary = _load_json(Path(args.summary_json))
    require_methods = [str(x).strip() for x in (args.require_methods or []) if str(x).strip()]
    errors = validate_grounding_benchmark(summary, require_methods=require_methods)

    # Optional improvement gate (paper-grade).
    if args.require_improvement:
        method_a, method_b, metric_key, delta_ge_s, alpha_s = [str(x).strip() for x in args.require_improvement]
        try:
            delta_ge = float(delta_ge_s)
            alpha = float(alpha_s)
        except Exception:
            errors.append("require-improvement DELTA_GE/ALPHA must be numeric")
            delta_ge = 0.0
            alpha = 0.05

        metrics_path = str(summary.get("metrics_jsonl_path", "")).strip()
        if metrics_path and Path(metrics_path).exists():
            rows = _load_jsonl(Path(metrics_path))
            # Build per-budget case intersection.
            by_budget_case_method: Dict[tuple[int, str, str], float] = {}
            budgets: set[int] = set()
            for r in rows:
                try:
                    b = int(r.get("budget_B", 0) or 0)
                except Exception:
                    continue
                if b <= 0:
                    continue
                cid = str(r.get("case_id", "")).strip()
                m = str(r.get("method", "")).strip()
                if not cid or not m:
                    continue
                if metric_key not in r:
                    continue
                try:
                    v = float(r.get(metric_key, 0.0))
                except Exception:
                    continue
                budgets.add(int(b))
                by_budget_case_method[(int(b), cid, m)] = float(v)

            passed = False
            best: Dict[str, float] | None = None
            best_budget = None
            for b in sorted(budgets):
                # Cases where both methods exist.
                cases = {
                    cid
                    for (bb, cid, m) in by_budget_case_method.keys()
                    if int(bb) == int(b) and m in {method_a, method_b}
                }
                diffs: List[float] = []
                for cid in sorted(cases):
                    ka = (int(b), str(cid), str(method_a))
                    kb = (int(b), str(cid), str(method_b))
                    if ka not in by_budget_case_method or kb not in by_budget_case_method:
                        continue
                    diffs.append(float(by_budget_case_method[ka]) - float(by_budget_case_method[kb]))
                if len(diffs) < 5:
                    continue
                stats = _bootstrap_ci_mean(
                    diffs,
                    n_boot=int(args.bootstrap_n),
                    alpha=float(alpha),
                    seed=int(args.seed) + int(b) * 10007,
                )
                if best is None or float(stats["ci_low"]) > float(best["ci_low"]):
                    best = dict(stats)
                    best_budget = int(b)
                if float(stats["ci_low"]) >= float(delta_ge):
                    passed = True
                    break

            if not passed:
                if best is None:
                    errors.append(
                        f"require-improvement failed: no budgets had enough paired cases for methods '{method_a}' vs '{method_b}'"
                    )
                else:
                    errors.append(
                        "require-improvement failed: "
                        f"best_budget={best_budget} mean={best['mean']:.6f} ci_low={best['ci_low']:.6f} "
                        f"(need ci_low >= {float(delta_ge):.6f}) metric={metric_key} A={method_a} B={method_b}"
                    )
        else:
            errors.append("require-improvement failed: summary.metrics_jsonl_path missing or not found")

    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] grounding benchmark validated")


if __name__ == "__main__":
    main()
