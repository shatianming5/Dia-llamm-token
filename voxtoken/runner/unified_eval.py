from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _compute_unsupported_rate(report: str, citations: List[Dict[str, Any]]) -> float:
    sentences = [s.strip() for s in report.splitlines() if s.strip()]
    if not sentences:
        return 0.0

    cited_by_sent = {int(c.get("sent_id", -1)): c.get("cited_token_ids", []) for c in citations}
    unsupported = 0
    for sent_id in range(len(sentences)):
        token_ids = cited_by_sent.get(sent_id, [])
        if not token_ids:
            unsupported += 1
    return unsupported / float(len(sentences))


def unified_eval(run_json: str, out_dir: str, *, case_id: str = "case-0000", budget_B: int = 0) -> Dict[str, Any]:
    run_path = Path(run_json)
    run = _load_json(run_path)

    report = str(run.get("report", ""))
    citations = list(run.get("citations", []))
    unsupported_rate = _compute_unsupported_rate(report, citations)

    metrics: Dict[str, Any] = {
        "case_id": case_id,
        "budget_B": budget_B,
        "tokens_used": 0,
        "latency_ms": {"total": 0.0},
        "slot_f1": 0.0,
        "unsupported_rate": float(unsupported_rate),
        "verifier_score": 0.0,
    }

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_path / "metrics.jsonl").write_text(json.dumps(metrics, ensure_ascii=False) + "\n", encoding="utf-8")

    return {"out_dir": str(out_path), "metrics_path": str(out_path / "metrics.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified evaluation (schema placeholder).")
    parser.add_argument("--in", dest="run_json", required=True, help="Path to run.json")
    parser.add_argument("--out", required=True, help="Output directory (e.g., artifacts/eval)")
    parser.add_argument("--case-id", default="case-0000")
    parser.add_argument("--budget", type=int, default=0)
    args = parser.parse_args()

    result = unified_eval(args.run_json, args.out, case_id=args.case_id, budget_B=args.budget)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

