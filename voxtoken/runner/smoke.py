from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from ..schemas import ReportPlan


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_smoke(out_dir: str) -> Dict[str, Any]:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    run = {
        "report": "",
        "citations": [],
        "plan": asdict(ReportPlan(facts=[], impression=[])),
        "trace": [],
        "issues": [],
    }

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
    }

    (out_path / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"out_dir": str(out_path), "run_path": str(out_path / "run.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="VoxToken++ repo smoke test (artifacts/schema only).")
    parser.add_argument("--out", required=True, help="Output directory (e.g., artifacts/smoke)")
    args = parser.parse_args()

    result = run_smoke(args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

