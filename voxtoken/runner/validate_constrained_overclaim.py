from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sentences(report: str) -> List[str]:
    return [s.strip() for s in str(report).splitlines() if s.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate constrained decoding blocks overclaim.")
    parser.add_argument("--out", required=True, help="Output directory (will contain run.json).")
    args = parser.parse_args()

    out_dir = Path(str(args.out))
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "voxtoken.runner.infer_refine",
        "--out",
        str(out_dir),
        "--budget",
        "16",
        "--config",
        "voxtoken/configs/inference_overclaim_constrained.yaml",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print("[ERR] infer_refine failed", file=sys.stderr)
        if proc.stdout:
            print(proc.stdout, file=sys.stderr)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        sys.exit(2)

    run_path = out_dir / "run.json"
    if not run_path.exists():
        print(f"[ERR] missing run.json: {run_path}", file=sys.stderr)
        sys.exit(2)

    run = _load_json(run_path)
    report = str(run.get("report", ""))
    facts = (run.get("plan") or {}).get("facts", []) if isinstance(run.get("plan"), dict) else []
    n_facts = int(len(facts)) if isinstance(facts, list) else 0
    sents = _sentences(report)

    if n_facts <= 0:
        print("[ERR] expected non-empty plan.facts", file=sys.stderr)
        sys.exit(1)

    if len(sents) != n_facts:
        print(f"[ERR] expected report sentence count == len(plan.facts) ({n_facts}), got {len(sents)}", file=sys.stderr)
        sys.exit(1)

    if "hallucination" in report.lower():
        print("[ERR] report still contains overclaim finding_type 'hallucination'", file=sys.stderr)
        sys.exit(1)

    issues = run.get("issues", [])
    if not isinstance(issues, list):
        issues = []
    for it in issues:
        if isinstance(it, dict) and str(it.get("type", "")).strip() == "overclaim":
            print("[ERR] found overclaim issue despite constrained decoding", file=sys.stderr)
            sys.exit(1)

    print("[OK] constrained decoding blocked overclaim")


if __name__ == "__main__":
    main()

