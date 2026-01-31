from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError("run.json must be a JSON object")
    return obj


def _sentences(report: str) -> List[str]:
    return [s.strip() for s in str(report).splitlines() if s.strip()]


def _as_int_pair(span: Any) -> Tuple[int, int] | None:
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            return int(span[0]), int(span[1])
        except Exception:
            return None
    return None


def _sorted_ints(values: Any) -> Tuple[int, ...]:
    if not isinstance(values, list):
        return tuple()
    out: List[int] = []
    for v in values:
        try:
            out.append(int(v))
        except Exception:
            continue
    return tuple(sorted(out))


def _sorted_strs(values: Any) -> Tuple[str, ...]:
    if not isinstance(values, list):
        return tuple()
    out: List[str] = []
    for v in values:
        s = str(v).strip()
        if s:
            out.append(s)
    return tuple(sorted(out))


def _canonical_issues(run: Dict[str, Any]) -> List[Tuple[str, int, int, str, Tuple[int, ...], Tuple[str, ...]]]:
    issues = run.get("issues", [])
    out: List[Tuple[str, int, int, str, Tuple[int, ...], Tuple[str, ...]]] = []
    if not isinstance(issues, list):
        return out
    for it in issues:
        if not isinstance(it, dict):
            continue
        tp = str(it.get("type", "")).strip()
        sp = _as_int_pair(it.get("span", None)) or (0, 0)
        reason = str(it.get("reason", "")).strip()
        rel_tokens = _sorted_ints(it.get("related_tokens", []))
        rel_eids = _sorted_strs(it.get("related_eids", []))
        out.append((tp, int(sp[0]), int(sp[1]), reason, rel_tokens, rel_eids))
    out.sort()
    return out


def _validate_issue_localization(run: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    n_sent = len(_sentences(str(run.get("report", ""))))
    issues = run.get("issues", [])
    if not isinstance(issues, list):
        errors.append("run.issues must be a list")
        return errors

    for i, it in enumerate(issues):
        if not isinstance(it, dict):
            errors.append(f"issue[{i}] must be an object")
            continue

        tp = str(it.get("type", "")).strip()
        if not tp:
            errors.append(f"issue[{i}].type is empty")
        reason = str(it.get("reason", "")).strip()
        if not reason:
            errors.append(f"issue[{i}].reason is empty")

        sp = _as_int_pair(it.get("span", None))
        if sp is None:
            errors.append(f"issue[{i}].span must be a pair of ints")
            sp = (0, 0)
        s0, s1 = int(sp[0]), int(sp[1])

        rel_tokens = list(_sorted_ints(it.get("related_tokens", [])))
        rel_eids = list(_sorted_strs(it.get("related_eids", [])))

        span_in_range = False
        if n_sent > 0:
            span_in_range = 0 <= int(s0) < int(n_sent) and 0 <= int(s1) < int(n_sent)
            if not span_in_range:
                errors.append(f"issue[{i}].span {sp} out of sentence range n={n_sent}")

        # At least one localization signal must be present: span points to a sentence OR related tokens/evidence.
        if not span_in_range and not rel_tokens and not rel_eids:
            errors.append(f"issue[{i}] not localized (span not in range and no related_tokens/related_eids): type={tp}")

    return errors


def validate_verifier_stability(
    run_a: Dict[str, Any],
    run_b: Dict[str, Any],
    *,
    score_tol: float = 1e-9,
) -> List[str]:
    errors: List[str] = []

    try:
        sa = float(run_a.get("verifier_score", 0.0))
        sb = float(run_b.get("verifier_score", 0.0))
    except Exception:
        errors.append("verifier_score must be parseable as float in both runs")
    else:
        if not (math.isfinite(sa) and math.isfinite(sb)):
            errors.append("verifier_score must be finite")
        elif abs(sa - sb) > float(score_tol):
            errors.append(f"verifier_score differs: {sa} vs {sb} (tol={float(score_tol)})")

    errors.extend(_validate_issue_localization(run_a))
    errors.extend(_validate_issue_localization(run_b))

    ia = _canonical_issues(run_a)
    ib = _canonical_issues(run_b)
    if ia != ib:
        errors.append(f"issues differ: len(a)={len(ia)} len(b)={len(ib)}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate verifier stability across two runs.")
    parser.add_argument("--a", required=True, help="Path to first run.json")
    parser.add_argument("--b", required=True, help="Path to second run.json")
    parser.add_argument("--score-tol", type=float, default=1e-9)
    args = parser.parse_args()

    run_a = _load_json(Path(args.a))
    run_b = _load_json(Path(args.b))
    errors = validate_verifier_stability(run_a, run_b, score_tol=float(args.score_tol))
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)

    print("[OK] verifier stability validated")


if __name__ == "__main__":
    main()
