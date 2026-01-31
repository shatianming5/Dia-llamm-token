from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sentences(report: str) -> List[str]:
    return [s.strip() for s in str(report).splitlines() if s.strip()]


def validate_run(
    run: Dict[str, Any],
    *,
    require_trace: bool = False,
    require_split: bool = False,
    require_overclaim: bool = False,
    require_tokenizer_codes: bool = False,
    require_finding_types_ge: int | None = None,
    require_meta_input: bool = False,
    require_meta_input_case_id: str | None = None,
    require_meta_input_volume_loader: str | None = None,
    require_meta_input_volume_path_exists: bool = False,
    require_meta_input_report_path_exists: bool = False,
    run_json_path: Path | None = None,
    require_final_report_txt: bool = False,
    require_evidence_graph_json: bool = False,
    require_trace_jsonl: bool = False,
) -> List[str]:
    errors: List[str] = []

    for key in ["report", "citations", "plan", "trace", "issues"]:
        if key not in run:
            errors.append(f"missing required key: {key}")

    report = str(run.get("report", ""))
    if not _sentences(report):
        errors.append("report has no non-empty sentences")

    # Basic budget sanity (if present).
    if "budget_B" in run and "tokens_used" in run:
        try:
            budget_B = int(run.get("budget_B", 0))
            tokens_used = int(run.get("tokens_used", 0))
            if budget_B > 0 and tokens_used > budget_B:
                errors.append(f"tokens_used({tokens_used}) > budget_B({budget_B})")
        except Exception:
            errors.append("budget_B/tokens_used are not parseable as ints")

    # Citation completeness: every sentence should have non-empty cited_token_ids.
    sentences = _sentences(report)
    citations = run.get("citations", [])
    cited_by_sent: Dict[int, List[int]] = {}
    if isinstance(citations, list):
        for c in citations:
            if not isinstance(c, dict):
                continue
            sid = int(c.get("sent_id", -1))
            tids = c.get("cited_token_ids", [])
            if isinstance(tids, list):
                cited_by_sent[sid] = [int(x) for x in tids if isinstance(x, (int, float, str))]
    missing = [i for i in range(len(sentences)) if not cited_by_sent.get(int(i), [])]
    if missing:
        errors.append(f"missing citations for sentence ids: {missing}")

    trace = run.get("trace", [])
    if require_trace:
        if not isinstance(trace, list) or len(trace) == 0:
            errors.append("trace is required but empty")

    if require_split:
        ok = False
        if isinstance(trace, list):
            for step in trace:
                if not isinstance(step, dict):
                    continue
                split_ids = step.get("split_token_ids", [])
                if isinstance(split_ids, list) and len(split_ids) > 0:
                    ok = True
                    break
        if not ok:
            errors.append("split trace is required but no trace step has non-empty split_token_ids")

    if require_overclaim:
        issues = run.get("issues", [])
        ok = False
        if isinstance(issues, list):
            for it in issues:
                if isinstance(it, dict) and str(it.get("type", "")).strip() == "overclaim":
                    ok = True
                    break
        if not ok:
            errors.append("overclaim issue is required but not found in run.issues")

    if require_tokenizer_codes:
        meta = run.get("meta", {})
        tok_meta = meta.get("tokenizer", {}) if isinstance(meta, dict) else {}
        if not (isinstance(tok_meta, dict) and bool(tok_meta.get("codes_enabled", False))):
            errors.append("tokenizer codes are required but meta.tokenizer.codes_enabled is not true")

        tokens = run.get("tokens", [])
        has_code = False
        if isinstance(tokens, list):
            for t in tokens:
                if isinstance(t, dict) and t.get("code", None) is not None:
                    has_code = True
                    break
        if not has_code:
            errors.append("tokenizer codes are required but no token in run.tokens has non-null code")

    if require_finding_types_ge is not None:
        plan = run.get("plan", {})
        facts = plan.get("facts", []) if isinstance(plan, dict) else []
        types = set()
        if isinstance(facts, list):
            for f in facts:
                if isinstance(f, dict):
                    ft = str(f.get("finding_type", "")).strip()
                    if ft:
                        types.add(ft)
        if len(types) < int(require_finding_types_ge):
            errors.append(f"plan finding_type count {len(types)} is < required {int(require_finding_types_ge)}")

    if require_meta_input or require_meta_input_case_id is not None or require_meta_input_volume_loader is not None:
        meta = run.get("meta", {})
        if not isinstance(meta, dict):
            errors.append("meta.input is required but run.meta is not an object")
        else:
            inp = meta.get("input", {})
            if not isinstance(inp, dict):
                errors.append("meta.input is required but missing or not an object")
            else:
                if require_meta_input_case_id is not None:
                    want = str(require_meta_input_case_id).strip()
                    got = str(inp.get("case_id", "")).strip()
                    if want and got != want:
                        errors.append(f"meta.input.case_id '{got}' does not match required '{want}'")

                if require_meta_input_volume_loader is not None:
                    want = str(require_meta_input_volume_loader).strip().lower()
                    got = str(inp.get("volume_loader", "")).strip().lower()
                    if want and got != want:
                        errors.append(f"meta.input.volume_loader '{got}' does not match required '{want}'")

                if require_meta_input_volume_path_exists:
                    vp = str(inp.get("volume_path", "")).strip()
                    if not vp:
                        errors.append("meta.input.volume_path is required but empty")
                    else:
                        if not Path(vp).exists():
                            errors.append(f"meta.input.volume_path not found: {vp}")

                if require_meta_input_report_path_exists:
                    rp = str(inp.get("report_path", "")).strip()
                    if not rp:
                        errors.append("meta.input.report_path is required but empty")
                    else:
                        if not Path(rp).exists():
                            errors.append(f"meta.input.report_path not found: {rp}")

    if require_final_report_txt or require_evidence_graph_json or require_trace_jsonl:
        if run_json_path is None:
            errors.append("sidecar validation requires run_json_path")
        else:
            out_dir = run_json_path.parent

            if require_final_report_txt:
                p = out_dir / "final_report.txt"
                if not p.exists():
                    errors.append(f"missing final_report.txt: {p}")
                else:
                    text = p.read_text(encoding="utf-8")
                    if not _sentences(text):
                        errors.append("final_report.txt has no non-empty sentences")

            if require_evidence_graph_json:
                p = out_dir / "evidence_graph.json"
                if not p.exists():
                    errors.append(f"missing evidence_graph.json: {p}")
                else:
                    try:
                        obj = json.loads(p.read_text(encoding="utf-8"))
                    except Exception:
                        errors.append("evidence_graph.json is not valid JSON")
                    else:
                        if not isinstance(obj, dict):
                            errors.append("evidence_graph.json must be a JSON object")
                        else:
                            nodes = obj.get("evidence_nodes", None)
                            if not isinstance(nodes, list):
                                errors.append("evidence_graph.json.evidence_nodes must be a list")

                            # Build a token_id -> omega_box_mm index (prefer explicit token_omega_index).
                            token_index: Dict[int, List[float]] = {}
                            idx = obj.get("token_omega_index", None)
                            if isinstance(idx, dict):
                                for k, box in idx.items():
                                    try:
                                        tid = int(k)
                                    except Exception:
                                        continue
                                    if not (isinstance(box, (list, tuple)) and len(box) == 6):
                                        continue
                                    try:
                                        token_index[int(tid)] = [float(x) for x in box]
                                    except Exception:
                                        continue

                            toks = obj.get("tokens", None)
                            if isinstance(toks, list):
                                for t in toks:
                                    if not isinstance(t, dict):
                                        continue
                                    tid = t.get("token_id", None)
                                    box = t.get("omega_box_mm", None)
                                    if tid is None:
                                        continue
                                    if not (isinstance(box, (list, tuple)) and len(box) == 6):
                                        continue
                                    try:
                                        token_index.setdefault(int(tid), [float(x) for x in box])
                                    except Exception:
                                        continue

                            if not token_index:
                                errors.append("evidence_graph.json must include tokens or token_omega_index with omega_box_mm")

                            if isinstance(nodes, list) and token_index:
                                missing_tids: List[int] = []
                                for n in nodes:
                                    if not isinstance(n, dict):
                                        continue
                                    tids = n.get("supported_token_ids", [])
                                    if not isinstance(tids, list):
                                        continue
                                    for tid_raw in tids:
                                        try:
                                            tid = int(tid_raw)
                                        except Exception:
                                            continue
                                        if tid not in token_index:
                                            missing_tids.append(int(tid))
                                if missing_tids:
                                    missing_sorted = sorted(set(int(x) for x in missing_tids))
                                    errors.append(
                                        "evidence_graph.json has supported_token_ids not present in token index: "
                                        + str(missing_sorted[:20])
                                    )

            if require_trace_jsonl:
                p = out_dir / "trace.jsonl"
                if not p.exists():
                    errors.append(f"missing trace.jsonl: {p}")
                else:
                    # Empty trace is allowed, but if there are lines they must be valid JSON.
                    for i, line in enumerate(p.read_text(encoding="utf-8").splitlines()):
                        if not line.strip():
                            continue
                        try:
                            json.loads(line)
                        except Exception:
                            errors.append(f"trace.jsonl line[{i}] is not valid JSON")
                            break

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate run.json invariants for the repo skeleton.")
    parser.add_argument("--in", dest="run_json", required=True, help="Path to run.json")
    parser.add_argument("--require-trace", action="store_true")
    parser.add_argument("--require-split", action="store_true")
    parser.add_argument("--require-overclaim", action="store_true")
    parser.add_argument("--require-tokenizer-codes", action="store_true")
    parser.add_argument("--require-finding-types-ge", type=int, default=None)
    parser.add_argument("--require-meta-input", action="store_true")
    parser.add_argument("--require-meta-input-case-id", default=None)
    parser.add_argument("--require-meta-input-volume-loader", default=None)
    parser.add_argument("--require-meta-input-volume-path-exists", action="store_true")
    parser.add_argument("--require-meta-input-report-path-exists", action="store_true")
    parser.add_argument("--require-final-report-txt", action="store_true")
    parser.add_argument("--require-evidence-graph-json", action="store_true")
    parser.add_argument("--require-trace-jsonl", action="store_true")
    args = parser.parse_args()

    run_json_path = Path(args.run_json)
    run = _load_json(run_json_path)
    errors = validate_run(
        run,
        require_trace=bool(args.require_trace),
        require_split=bool(args.require_split),
        require_overclaim=bool(args.require_overclaim),
        require_tokenizer_codes=bool(args.require_tokenizer_codes),
        require_finding_types_ge=args.require_finding_types_ge,
        require_meta_input=bool(args.require_meta_input),
        require_meta_input_case_id=args.require_meta_input_case_id,
        require_meta_input_volume_loader=args.require_meta_input_volume_loader,
        require_meta_input_volume_path_exists=bool(args.require_meta_input_volume_path_exists),
        require_meta_input_report_path_exists=bool(args.require_meta_input_report_path_exists),
        run_json_path=run_json_path,
        require_final_report_txt=bool(args.require_final_report_txt),
        require_evidence_graph_json=bool(args.require_evidence_graph_json),
        require_trace_jsonl=bool(args.require_trace_jsonl),
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] run.json validated")


if __name__ == "__main__":
    main()
