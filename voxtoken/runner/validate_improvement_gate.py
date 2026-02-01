from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        return [dict(row) for row in r]


def _find_row(rows: List[Dict[str, Any]], *, method: str, budget_B: int) -> Dict[str, Any] | None:
    method = str(method).strip()
    for row in rows:
        if str(row.get("method", "")).strip() != method:
            continue
        try:
            b = int(float(row.get("budget_B", 0) or 0))
        except Exception:
            continue
        if int(b) != int(budget_B):
            continue
        return row
    return None


def _f(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate improvement gate on paper_export table1_main_ci.csv.")
    parser.add_argument("--table", required=True, help="Path to table1_main_ci.csv")
    parser.add_argument("--budget", type=int, required=True, help="Budget_B to evaluate (e.g., 32)")
    parser.add_argument("--lhs", default="learned", help="Method A (default: learned)")
    parser.add_argument("--rhs", default="random", help="Method B (default: random)")
    parser.add_argument("--metric", default="ground_mean_iou", help="Metric base name (default: ground_mean_iou)")
    parser.add_argument("--delta-ge", type=float, required=True, help="Require lhs_mean - rhs_mean >= delta_ge")
    parser.add_argument(
        "--require-ci-nonoverlap",
        action="store_true",
        help="Also require lhs_ci_low >= rhs_ci_high (non-overlapping 95% CI in table).",
    )
    parser.add_argument(
        "--also-pass-if-ci-nonoverlap",
        action="store_true",
        help="Also pass if lhs_ci_low >= rhs_ci_high (non-overlapping 95% CI in table).",
    )
    args = parser.parse_args()

    table = Path(str(args.table)).expanduser()
    if not table.exists():
        print(f"[ERR] table not found: {table}", file=sys.stderr)
        sys.exit(2)

    rows = _read_csv(table)
    if not rows:
        print(f"[ERR] table is empty: {table}", file=sys.stderr)
        sys.exit(1)

    budget_B = int(args.budget)
    lhs = _find_row(rows, method=str(args.lhs), budget_B=int(budget_B))
    rhs = _find_row(rows, method=str(args.rhs), budget_B=int(budget_B))
    if lhs is None:
        print(f"[ERR] missing method={args.lhs!r} budget_B={budget_B} in {table}", file=sys.stderr)
        sys.exit(1)
    if rhs is None:
        print(f"[ERR] missing method={args.rhs!r} budget_B={budget_B} in {table}", file=sys.stderr)
        sys.exit(1)

    metric = str(args.metric).strip()
    mean_k = f"{metric}_mean"
    lo_k = f"{metric}_ci_low"
    hi_k = f"{metric}_ci_high"

    lhs_mean = _f(lhs, mean_k)
    rhs_mean = _f(rhs, mean_k)
    lhs_lo = _f(lhs, lo_k)
    rhs_hi = _f(rhs, hi_k)

    delta = float(lhs_mean) - float(rhs_mean)
    thr = float(args.delta_ge)
    ci_nonoverlap = bool(float(lhs_lo) >= float(rhs_hi))

    passed_by_delta = bool(delta >= thr)
    passed_by_ci = bool(ci_nonoverlap)
    passed = bool(passed_by_delta) or (bool(args.also_pass_if_ci_nonoverlap) and bool(passed_by_ci))
    if bool(args.require_ci_nonoverlap):
        passed = bool(passed) and bool(passed_by_ci)
    if not bool(passed):
        reasons: list[str] = []
        if not bool(passed_by_delta) and not (bool(args.also_pass_if_ci_nonoverlap) and bool(passed_by_ci)):
            reasons.append(f"delta {delta:.6f} < {thr:.6f}")
        if bool(args.require_ci_nonoverlap) and not bool(passed_by_ci):
            reasons.append(f"ci_nonoverlap false (lhs_ci_low {lhs_lo:.6f} < rhs_ci_high {rhs_hi:.6f})")
        if not reasons:
            reasons.append("unspecified failure (check flags)")
        print(
            "[ERR] improvement gate failed: "
            f"budget_B={budget_B} metric={metric} lhs={args.lhs} rhs={args.rhs} "
            f"delta={delta:.6f} (thr {thr:.6f}), ci_nonoverlap={ci_nonoverlap}, "
            f"lhs_ci_low={lhs_lo:.6f}, rhs_ci_high={rhs_hi:.6f} :: " + "; ".join(reasons),
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "[OK] improvement gate passed: "
        f"budget_B={budget_B} metric={metric} lhs={args.lhs} rhs={args.rhs} "
        f"delta={delta:.6f} (>= {thr:.6f}) ci_nonoverlap={ci_nonoverlap}"
    )


if __name__ == "__main__":
    main()
