from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _get_metric(payload: Dict[str, Any], key: str) -> float:
    if "." not in key:
        return float(payload.get(key, 0.0))
    cur: Any = payload
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return 0.0
        cur = cur[part]
    return float(cur)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate learned policy improves over heuristic on a metric.")
    parser.add_argument("--heuristic", required=True, help="Path to heuristic metrics.json")
    parser.add_argument("--learned", required=True, help="Path to learned metrics.json")
    parser.add_argument("--metric", default="ground_mean_iou", help="Metric key (supports dot paths)")
    parser.add_argument("--require-delta-ge", type=float, default=0.0, help="Require learned - heuristic >= delta")
    args = parser.parse_args()

    m_h = _load_json(Path(args.heuristic))
    m_l = _load_json(Path(args.learned))

    key = str(args.metric)
    v_h = float(_get_metric(m_h, key))
    v_l = float(_get_metric(m_l, key))
    delta = float(v_l) - float(v_h)
    thr = float(args.require_delta_ge)

    errors: list[str] = []
    if not (v_l > v_h):
        errors.append(f"learned({v_l}) is not > heuristic({v_h}) for metric '{key}'")
    if delta < thr:
        errors.append(f"delta({delta}) is not >= {thr} for metric '{key}'")

    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] policy improvement validated")


if __name__ == "__main__":
    main()

