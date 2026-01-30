from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def run_counterfactuals(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Causal tests: shuffle citations, swap tokens, etc."""
    run_json = cfg.get("run_json") or cfg.get("path")
    if not run_json:
        return {"error": "missing run_json"}

    run = json.loads(Path(str(run_json)).read_text(encoding="utf-8"))
    report = str(run.get("report", ""))
    citations = list(run.get("citations", []))

    sentences = [s.strip() for s in report.splitlines() if s.strip()]

    def unsupported_rate(cites: list[dict[str, Any]]) -> float:
        if not sentences:
            return 0.0
        cited_by_sent = {int(c.get("sent_id", -1)): c.get("cited_token_ids", []) for c in cites}
        unsupported = 0
        for sent_id in range(len(sentences)):
            ids = cited_by_sent.get(sent_id, [])
            if not ids:
                unsupported += 1
        return unsupported / float(len(sentences))

    base = float(unsupported_rate(citations))

    # Citation swap: rotate citations by 1 sentence.
    swapped = []
    for c in citations:
        swapped.append(dict(c))
    for i in range(len(swapped)):
        swapped[i]["sent_id"] = int((int(swapped[i].get("sent_id", 0)) + 1) % max(1, len(sentences)))
    swap_rate = float(unsupported_rate(swapped))

    # Remove all citations.
    removed = [dict(c, cited_token_ids=[]) for c in citations]
    removed_rate = float(unsupported_rate(removed))

    return {
        "n_sentences": int(len(sentences)),
        "unsupported_rate": {
            "base": base,
            "citation_swap": swap_rate,
            "remove_citations": removed_rate,
        },
    }
