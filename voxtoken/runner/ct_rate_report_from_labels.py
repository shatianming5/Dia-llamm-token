from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from ..schemas import Citation, FactSlot, ReportPlan
from ..verify.verifier import Verifier


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest_row(manifest_jsonl: Path, case_id: str) -> Dict[str, Any]:
    for line in manifest_jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if not isinstance(obj, dict):
            continue
        if str(obj.get("case_id", "")).strip() == str(case_id).strip():
            return obj
    raise KeyError(f"case_id not found in manifest: {case_id}")


def _label_to_finding_type(label: str) -> str:
    """
    Convert a CT-RATE label string (often contains spaces/punctuation) into a stable
    single-token finding_type so repo-skeleton verifier rules do not flag overclaim.

    Note: ct_rate_label_eval normalizes '_' back into spaces, so this is compatible
    with gold labels.
    """

    s = str(label or "").strip()
    if not s:
        return ""
    s = re.sub(r"\\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def ct_rate_report_from_labels(
    *,
    manifest_jsonl: str,
    out_dir: str,
    case_id: str,
    max_labels: int | None = None,
) -> Dict[str, Any]:
    manifest_path = Path(manifest_jsonl)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    row = _load_manifest_row(manifest_path, case_id)
    labels_raw = row.get("labels_pos", [])
    if not isinstance(labels_raw, list):
        labels_raw = []

    labels: List[str] = []
    for x in labels_raw:
        ft = _label_to_finding_type(str(x))
        if ft:
            labels.append(ft)
    if max_labels is not None:
        labels = labels[: max(0, int(max_labels))]

    if not labels:
        raise ValueError(f"no usable labels_pos for case_id={case_id}")

    facts: List[FactSlot] = []
    citations: List[Citation] = []
    lines: List[str] = []
    for i, ft in enumerate(labels):
        facts.append(
            FactSlot(
                finding_type=str(ft),
                side="U",
                location="U",
                size_bin="U",
                certainty="U",
                supported_token_ids=[int(i + 1)],
            )
        )
        lines.append(f"{ft} (side=U, location=U, size=U, certainty=U).")
        citations.append(Citation(sent_id=int(i), cited_token_ids=[int(i + 1)]))

    plan = ReportPlan(facts=facts, impression=[])
    report = "\n".join(lines).strip() + "\n"

    verifier = Verifier(cfg={"weights": {}})
    score, issues = verifier.verify(report, citations, plan)

    run: Dict[str, Any] = {
        "case_id": str(case_id),
        "budget_B": int(len(labels)),
        "tokens_used": int(len(labels)),
        "verifier_score": float(score),
        "report": report,
        "citations": [asdict(c) for c in citations],
        "plan": asdict(plan),
        "trace": [],
        "issues": [asdict(x) for x in issues],
        "meta": {
            "input": {
                "case_id": str(case_id),
                "manifest_jsonl": str(manifest_path),
                "report_path": row.get("report_path"),
                "volume_path": row.get("volume_path"),
                "labels_pos_raw": labels_raw,
                "labels_pos_used": labels,
            },
        },
    }

    summary = {
        "timestamp_utc": _utc_now_iso(),
        "python": sys.version,
        "platform": platform.platform(),
        "manifest_jsonl": str(manifest_path),
        "case_id": str(case_id),
        "out_dir": str(out_path),
    }

    (out_path / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (out_path / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"out_dir": str(out_path), "run_path": str(out_path / "run.json")}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a CT-RATE-style report directly from manifest labels_pos (repo skeleton baseline)."
    )
    parser.add_argument("--manifest", dest="manifest_jsonl", required=True, help="Path to manifest.jsonl (must contain labels_pos)")
    parser.add_argument("--out", required=True, help="Output directory (writes run.json + summary.json)")
    parser.add_argument("--case-id", required=True, help="Case ID to generate a report for")
    parser.add_argument("--max-labels", type=int, default=None, help="Optional cap on number of labels used")
    args = parser.parse_args()

    result = ct_rate_report_from_labels(
        manifest_jsonl=str(args.manifest_jsonl),
        out_dir=str(args.out),
        case_id=str(args.case_id).strip(),
        max_labels=(int(args.max_labels) if args.max_labels is not None else None),
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

