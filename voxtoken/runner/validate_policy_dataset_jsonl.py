from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _parse_require_budgets(s: str) -> List[int]:
    raw = str(s or "").strip()
    if not raw:
        return []
    out: List[int] = []
    for part in raw.split(","):
        part = str(part).strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            continue
    out = sorted({int(x) for x in out if int(x) > 0})
    return out


def validate_policy_dataset(
    rows: List[Dict[str, Any]],
    *,
    require_rows_ge: int | None,
    require_cases_ge: int | None,
    require_label: bool,
    require_budgets: List[int],
    require_step_idx: bool,
    require_exactly_one_positive_per_step: bool,
) -> List[str]:
    errors: List[str] = []
    if require_rows_ge is not None and len(rows) < int(require_rows_ge):
        errors.append(f"n_rows({len(rows)}) is not >= {int(require_rows_ge)}")

    cases = {str(r.get('case_id', '')).strip() for r in rows if str(r.get('case_id', '')).strip()}
    if require_cases_ge is not None and len(cases) < int(require_cases_ge):
        errors.append(f"n_cases({len(cases)}) is not >= {int(require_cases_ge)}")

    required_keys = ["case_id", "token_id", "recon_error", "evidence_entropy", "citation_pressure", "history_splits", "reward"]
    for i, r in enumerate(rows[: min(50, len(rows))]):
        for k in required_keys:
            if k not in r:
                errors.append(f"row[{i}] missing key: {k}")
        if require_label:
            if "label" not in r:
                errors.append(f"row[{i}] missing key: label")
            else:
                try:
                    v = int(r.get("label", 0))
                    if v not in {0, 1}:
                        errors.append(f"row[{i}] label must be 0/1 (got {r.get('label')!r})")
                except Exception:
                    errors.append(f"row[{i}] label must be int-like 0/1 (got {r.get('label')!r})")
        if require_step_idx and "step_idx" not in r:
            errors.append(f"row[{i}] missing key: step_idx")

    if require_budgets:
        seen = set()
        for r in rows:
            try:
                b = int(r.get("budget_B", 0))
            except Exception:
                b = 0
            if b > 0:
                seen.add(int(b))
        missing = [int(b) for b in require_budgets if int(b) not in seen]
        if missing:
            errors.append(f"dataset missing required budgets: {missing} (seen={sorted(seen)})")

    if require_step_idx or require_exactly_one_positive_per_step:
        for i, r in enumerate(rows):
            if "step_idx" not in r:
                if require_step_idx:
                    errors.append(f"row[{i}] missing key: step_idx")
                continue
            try:
                s = int(r.get("step_idx", 0))
            except Exception:
                errors.append(f"row[{i}] step_idx must be int-like (got {r.get('step_idx')!r})")
                continue
            if int(s) < 0:
                errors.append(f"row[{i}] step_idx must be >=0 (got {s})")

    if require_exactly_one_positive_per_step:
        by_group: Dict[Tuple[str, int, int], Dict[str, int]] = {}
        for r in rows:
            cid = str(r.get("case_id", "")).strip()
            if not cid:
                continue
            try:
                b = int(r.get("budget_B", 0) or 0)
                s = int(r.get("step_idx", 0) or 0)
            except Exception:
                continue
            if int(b) <= 0 or int(s) < 0:
                continue
            gk = (cid, int(b), int(s))
            st = by_group.get(gk)
            if st is None:
                st = {"n": 0, "pos": 0}
                by_group[gk] = st
            st["n"] += 1
            try:
                y = int(r.get("label", 0) or 0)
            except Exception:
                y = 0
            if int(y) == 1:
                st["pos"] += 1

        if not by_group:
            errors.append("no (case_id,budget_B,step_idx) groups found; cannot validate per-step label constraints")
        else:
            bad: List[str] = []
            for (cid, b, s), st in by_group.items():
                if int(st.get("n", 0)) < 2:
                    bad.append(f"{cid}|B{b}|step{s}: n={st.get('n')} (<2)")
                    continue
                if int(st.get("pos", 0)) != 1:
                    bad.append(f"{cid}|B{b}|step{s}: pos={st.get('pos')} (!=1)")
                    continue
                if len(bad) >= 10:
                    break
            if bad:
                errors.append("per-step label constraint failed for some groups (showing up to 10): " + "; ".join(bad))

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate policy dataset.jsonl invariants.")
    parser.add_argument("--in", dest="dataset_jsonl", required=True, help="Path to dataset.jsonl")
    parser.add_argument("--require-rows-ge", type=int, default=None)
    parser.add_argument("--require-cases-ge", type=int, default=None)
    parser.add_argument("--require-label", action="store_true", help="Require oracle-imitation label field (0/1)")
    parser.add_argument("--require-budgets", default="", help="Comma-separated list of required budget_B values (e.g., '16,32')")
    parser.add_argument("--require-step-idx", action="store_true", help="Require step_idx field (for listwise training)")
    parser.add_argument(
        "--require-exactly-one-positive-per-step",
        action="store_true",
        help="Require exactly one label==1 per (case_id,budget_B,step_idx) group (and group size>=2)",
    )
    args = parser.parse_args()

    p = Path(args.dataset_jsonl)
    if not p.exists():
        print(f"[ERR] missing dataset: {p}", file=sys.stderr)
        sys.exit(2)

    rows = _load_jsonl(p)
    req_budgets = _parse_require_budgets(str(args.require_budgets))
    errors = validate_policy_dataset(
        rows,
        require_rows_ge=args.require_rows_ge,
        require_cases_ge=args.require_cases_ge,
        require_label=bool(args.require_label),
        require_budgets=list(req_budgets),
        require_step_idx=bool(args.require_step_idx),
        require_exactly_one_positive_per_step=bool(args.require_exactly_one_positive_per_step),
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] policy dataset validated")


if __name__ == "__main__":
    main()
