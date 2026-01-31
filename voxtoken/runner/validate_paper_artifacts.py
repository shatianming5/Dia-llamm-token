from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List


TABLE1_FIELDS = ["method", "budget_B", "n", "ground_mean_iou_mean", "ground_hit@0.1_mean", "tokens_used_mean", "latency_ms.total_mean"]
TABLE1_CI_FIELDS = [
    "method",
    "budget_B",
    "n",
    "ground_mean_iou_mean",
    "ground_mean_iou_ci_low",
    "ground_mean_iou_ci_high",
    "ground_hit@0.1_mean",
    "ground_hit@0.1_ci_low",
    "ground_hit@0.1_ci_high",
    "tokens_used_mean",
    "latency_ms.total_mean",
]
TABLE2_FIELDS = [
    "budget_B",
    "delta_iou_learned_vs_heuristic",
    "delta_hit01_learned_vs_heuristic",
    "delta_tokens_used_learned_vs_heuristic",
    "delta_latency_ms_total_learned_vs_heuristic",
]


def _read_csv_header(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r, [])
    return [str(x).strip() for x in header]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate paper export artifacts (tables/figures).")
    parser.add_argument("--dir", required=True, help="Output directory (artifacts/paper_e0910)")
    args = parser.parse_args()

    root = Path(args.dir)
    if not root.exists():
        print(f"[ERR] missing dir: {root}", file=sys.stderr)
        sys.exit(2)

    required_files = [
        root / "table1_main.csv",
        root / "table1_main_ci.csv",
        root / "table2_ablation.csv",
        root / "fig2_pareto_tokens.png",
        root / "fig2_pareto_tokens_ci.png",
        root / "fig2_pareto_latency.png",
    ]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        for m in missing:
            print(f"[ERR] missing required file: {m}")
        sys.exit(1)

    h1 = _read_csv_header(root / "table1_main.csv")
    if h1 != TABLE1_FIELDS:
        print(f"[ERR] table1_main.csv header mismatch: {h1} != {TABLE1_FIELDS}")
        sys.exit(1)

    h1ci = _read_csv_header(root / "table1_main_ci.csv")
    if h1ci != TABLE1_CI_FIELDS:
        print(f"[ERR] table1_main_ci.csv header mismatch: {h1ci} != {TABLE1_CI_FIELDS}")
        sys.exit(1)

    h2 = _read_csv_header(root / "table2_ablation.csv")
    if h2 != TABLE2_FIELDS:
        print(f"[ERR] table2_ablation.csv header mismatch: {h2} != {TABLE2_FIELDS}")
        sys.exit(1)

    fig3_dir = root / "fig3_examples"
    svgs = list(fig3_dir.glob("*.svg")) if fig3_dir.exists() else []
    if not svgs:
        print("[ERR] missing fig3_examples/*.svg")
        sys.exit(1)

    # PNG signature check.
    sig = b"\x89PNG\r\n\x1a\n"
    for p in [root / "fig2_pareto_tokens.png", root / "fig2_pareto_tokens_ci.png", root / "fig2_pareto_latency.png"]:
        data = p.read_bytes()[:8]
        if data != sig:
            print(f"[ERR] invalid PNG signature: {p}")
            sys.exit(1)

    print("[OK] paper artifacts validated")


if __name__ == "__main__":
    main()
