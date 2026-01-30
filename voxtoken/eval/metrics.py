from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def compute_metrics(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute core metrics: correctness, grounding, unsupported, efficiency."""
    metrics_jsonl = cfg.get("metrics_jsonl") or cfg.get("metrics_path") or cfg.get("path")
    if not metrics_jsonl:
        return {"error": "missing metrics_jsonl"}

    path = Path(str(metrics_jsonl))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return {"n": 0}

    def mean(values: list[float]) -> float:
        return float(sum(values) / float(len(values))) if values else 0.0

    verifier_scores = [float(r.get("verifier_score", 0.0)) for r in rows]
    unsupported_rates = [float(r.get("unsupported_rate", 0.0)) for r in rows]
    tokens_used = [float(r.get("tokens_used", 0.0)) for r in rows]
    slot_f1 = [float(r.get("slot_f1", 0.0)) for r in rows]
    latency_total = [float((r.get("latency_ms") or {}).get("total", 0.0)) for r in rows]

    return {
        "n": int(len(rows)),
        "mean": {
            "verifier_score": mean(verifier_scores),
            "unsupported_rate": mean(unsupported_rates),
            "tokens_used": mean(tokens_used),
            "slot_f1": mean(slot_f1),
            "latency_ms.total": mean(latency_total),
        },
    }
