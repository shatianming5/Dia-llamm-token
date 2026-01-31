from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate refine stop rule: ΔV/Δ|T| < tau.")
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
        "voxtoken/configs/inference_tau_stop.yaml",
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
    trace = run.get("trace", [])
    if not isinstance(trace, list):
        print("[ERR] run.trace must be a list", file=sys.stderr)
        sys.exit(1)

    max_rounds = 3
    if len(trace) != 1:
        print(f"[ERR] expected trace length 1 due to tau stop (max_rounds={max_rounds}), got {len(trace)}", file=sys.stderr)
        sys.exit(1)

    step0 = trace[0]
    if not isinstance(step0, dict):
        print("[ERR] trace[0] must be an object", file=sys.stderr)
        sys.exit(1)

    split_ids = step0.get("split_token_ids", [])
    added_ids = step0.get("added_token_ids", [])
    if not (isinstance(split_ids, list) and split_ids):
        print("[ERR] expected non-empty trace[0].split_token_ids", file=sys.stderr)
        sys.exit(1)
    if not (isinstance(added_ids, list) and added_ids):
        print("[ERR] expected non-empty trace[0].added_token_ids", file=sys.stderr)
        sys.exit(1)

    print("[OK] tau stop validated")


if __name__ == "__main__":
    main()

