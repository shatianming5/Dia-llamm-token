from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


def _group_key(row: Dict[str, Any]) -> Tuple[str, int, int]:
    case_id = str(row.get("case_id", "")).strip()
    try:
        budget_B = int(row.get("budget_B", 0) or 0)
    except Exception:
        budget_B = 0
    try:
        step_idx = int(row.get("step_idx", row.get("step", 0)) or 0)
    except Exception:
        step_idx = 0
    return case_id, int(budget_B), int(step_idx)


def _compute_mu_std(rows: List[Dict[str, Any]], *, eps: float) -> Tuple[float, float]:
    """
    Compute global μ/σ from a token partition using law of total variance.

    Requires keys:
      - mean_intensity
      - recon_error (token variance proxy)
      - box_volume_mm3 (weights)
    """
    w_sum = 0.0
    w_mean = 0.0
    w_e2 = 0.0
    for r in rows:
        w = float(r.get("box_volume_mm3", 0.0) or 0.0)
        if not (w > 0.0):
            w = 1.0
        mu_i = float(r.get("mean_intensity", 0.0) or 0.0)
        var_i = float(r.get("recon_error", 0.0) or 0.0)
        w_sum += float(w)
        w_mean += float(w) * float(mu_i)
        w_e2 += float(w) * (float(var_i) + float(mu_i) * float(mu_i))
    if not (w_sum > 0.0):
        return 0.0, 1.0
    mu = float(w_mean) / float(w_sum)
    e2 = float(w_e2) / float(w_sum)
    var = max(0.0, float(e2) - float(mu) * float(mu))
    std = float(math.sqrt(float(var) + float(eps)))
    return float(mu), float(std)


def augment_dataset_intensity_z(
    *,
    in_jsonl: Path,
    out_jsonl: Path,
    eps: float,
    fail_if_not_grouped: bool,
) -> Dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_rows = 0
    n_groups = 0
    n_rows_with_z = 0
    n_bad_group_order = 0

    cur_key: Tuple[str, int, int] | None = None
    cur_rows: List[Dict[str, Any]] = []
    seen_keys: set[Tuple[str, int, int]] = set()

    def flush_group(rows: List[Dict[str, Any]], key: Tuple[str, int, int]) -> Iterable[str]:
        nonlocal n_groups, n_rows_with_z
        n_groups += 1
        mu, std = _compute_mu_std(rows, eps=float(eps))
        for r in rows:
            m = float(r.get("mean_intensity", 0.0) or 0.0)
            mx = float(r.get("max_intensity", 0.0) or 0.0)
            r["mean_intensity_z"] = float(m - float(mu)) / float(std)
            r["max_intensity_z"] = float(mx - float(mu)) / float(std)
            n_rows_with_z += 1
            yield json.dumps(r, ensure_ascii=False)

    lines_out: List[str] = []
    for r in _iter_jsonl(in_jsonl):
        n_rows += 1
        k = _group_key(r)
        if cur_key is None:
            cur_key = k
        if k != cur_key:
            # Best-effort check that groups are contiguous (streaming-safe).
            if fail_if_not_grouped:
                if cur_key in seen_keys:
                    n_bad_group_order += 1
                    raise RuntimeError(f"non-contiguous group detected: {cur_key} (input likely shuffled)")
            seen_keys.add(cur_key)
            lines_out.extend([s for s in flush_group(cur_rows, cur_key)])
            cur_rows = []
            cur_key = k
        cur_rows.append(r)

    if cur_key is not None and cur_rows:
        if fail_if_not_grouped and cur_key in seen_keys:
            n_bad_group_order += 1
            raise RuntimeError(f"non-contiguous group detected: {cur_key} (input likely shuffled)")
        lines_out.extend([s for s in flush_group(cur_rows, cur_key)])

    out_jsonl.write_text("".join(s + "\n" for s in lines_out), encoding="utf-8")
    return {
        "in_jsonl": str(in_jsonl),
        "out_jsonl": str(out_jsonl),
        "eps": float(eps),
        "n_rows": int(n_rows),
        "n_groups": int(n_groups),
        "n_rows_with_z": int(n_rows_with_z),
        "fail_if_not_grouped": bool(fail_if_not_grouped),
        "n_bad_group_order": int(n_bad_group_order),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment a policy dataset.jsonl with per-group mean/max intensity z-scores (streaming; expects grouped input)."
    )
    parser.add_argument("--in", dest="in_jsonl", required=True, help="Input dataset.jsonl")
    parser.add_argument("--out", dest="out_jsonl", required=True, help="Output dataset.jsonl (augmented)")
    parser.add_argument("--eps", type=float, default=1.0e-6, help="Epsilon added to variance (default: 1e-6)")
    parser.add_argument(
        "--fail-if-not-grouped",
        action="store_true",
        help="Fail if (case_id,budget_B,step_idx) groups are not contiguous in the input.",
    )
    args = parser.parse_args()

    in_jsonl = Path(str(args.in_jsonl)).expanduser()
    out_jsonl = Path(str(args.out_jsonl)).expanduser()
    if not in_jsonl.exists():
        print(f"[ERR] missing input: {in_jsonl}", file=sys.stderr)
        sys.exit(2)

    summary = augment_dataset_intensity_z(
        in_jsonl=in_jsonl,
        out_jsonl=out_jsonl,
        eps=float(args.eps),
        fail_if_not_grouped=bool(args.fail_if_not_grouped),
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()

