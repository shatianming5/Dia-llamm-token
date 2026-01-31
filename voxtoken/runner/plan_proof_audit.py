from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    evidence: List[str]


@dataclass(frozen=True)
class ExperimentRow:
    exp_id: str
    goal_claim: str
    smoke_checked: bool
    full_checked: bool
    artifacts: str


@dataclass(frozen=True)
class ResultHit:
    path: Path
    stage: str
    status: str
    exit_code: int


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_exp_id(raw: str) -> str:
    s = str(raw or "").strip().upper()
    if not s:
        return s
    if re.fullmatch(r"E\d{4}", s):
        return s
    m = re.fullmatch(r"EXP[-_]?(\d{4})", s)
    if m:
        return f"E{m.group(1)}"
    m = re.fullmatch(r"(\d{4})", s)
    if m:
        return f"E{m.group(1)}"
    return s


def _escape_md_cell(text: str) -> str:
    s = str(text or "")
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.replace("|", "\\|")
    return s.strip()


def _parse_plan_claims(plan_path: Path) -> List[Claim]:
    claim_re = re.compile(r"^\s*-\s*\[[ xX]\]\s*(C\d{4})\s*:\s*(.*)\s*$")
    evid_re = re.compile(r"\bE\d{4}\b")

    claims: List[Claim] = []
    cur_id: str | None = None
    cur_text: str = ""
    cur_evidence: List[str] = []

    for line in plan_path.read_text(encoding="utf-8").splitlines():
        m = claim_re.match(line)
        if m:
            if cur_id is not None:
                claims.append(Claim(claim_id=cur_id, text=cur_text, evidence=sorted(set(cur_evidence))))
            cur_id = str(m.group(1))
            cur_text = str(m.group(2)).strip()
            cur_evidence = []
            continue

        if cur_id is None:
            continue
        if "Evidence" in line:
            for eid in evid_re.findall(line):
                cur_evidence.append(_normalize_exp_id(eid))

    if cur_id is not None:
        claims.append(Claim(claim_id=cur_id, text=cur_text, evidence=sorted(set(cur_evidence))))

    return claims


def _parse_experiment_table(ledger_path: Path) -> Dict[str, ExperimentRow]:
    lines = ledger_path.read_text(encoding="utf-8").splitlines()
    header_cells: List[str] = []
    rows: Dict[str, ExperimentRow] = {}

    in_table = False
    for line in lines:
        if not in_table:
            if line.strip().startswith("| ID |"):
                header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
                in_table = True
            continue

        if line.strip().startswith("## "):
            break
        if not line.strip() or not line.strip().startswith("|"):
            continue
        if line.strip().startswith("|---"):
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != len(header_cells):
            raise ValueError(f"Malformed experiment table row (bad cell count): {line}")

        row = {header_cells[i]: cells[i] for i in range(len(header_cells))}
        exp_id = _normalize_exp_id(row.get("ID", ""))
        if not re.fullmatch(r"E\d{4}", exp_id):
            continue

        smoke_raw = str(row.get("Smoke", "")).strip().lower()
        full_raw = str(row.get("Full", "")).strip().lower()

        rows[exp_id] = ExperimentRow(
            exp_id=exp_id,
            goal_claim=str(row.get("Goal/Claim", "")).strip(),
            smoke_checked=("[x]" in smoke_raw),
            full_checked=("[x]" in full_raw),
            artifacts=str(row.get("Artifacts/Results", "")).strip(),
        )

    return rows


def _load_results(results_dir: Path) -> Dict[str, List[ResultHit]]:
    by_exp: Dict[str, List[ResultHit]] = {}
    if not results_dir.exists():
        return by_exp

    for p in sorted(results_dir.glob("E*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        exp_id = _normalize_exp_id(str(payload.get("id", "")).strip())
        if not re.fullmatch(r"E\d{4}", exp_id):
            continue

        stage = str(payload.get("stage", "")).strip()
        status = str(payload.get("status", "")).strip()
        exit_code = int(payload.get("exit_code", 0) or 0)
        by_exp.setdefault(exp_id, []).append(ResultHit(path=p, stage=stage, status=status, exit_code=exit_code))

    return by_exp


def _find_passed_hits(hits: Iterable[ResultHit], *, kind: str) -> List[ResultHit]:
    kind = str(kind).strip().lower()
    out: List[ResultHit] = []
    for h in hits:
        if str(h.status).strip().lower() != "passed":
            continue
        stage = str(h.stage).strip().lower()
        if kind and kind in stage:
            out.append(h)
    return out


def _claim_status(
    claim: Claim,
    *,
    ledger: Dict[str, ExperimentRow],
    results: Dict[str, List[ResultHit]],
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if not claim.evidence:
        return False, ["missing Evidence: E####"]

    for eid in claim.evidence:
        if eid not in ledger:
            reasons.append(f"{eid}: not found in docs/experiment.md")
            continue
        row = ledger[eid]
        if not row.smoke_checked:
            reasons.append(f"{eid}: Smoke checkbox not [x]")
        if not row.full_checked:
            reasons.append(f"{eid}: Full checkbox not [x]")

        hits = results.get(eid, [])
        smoke_hits = _find_passed_hits(hits, kind="smoke")
        full_hits = _find_passed_hits(hits, kind="full")
        if not smoke_hits:
            reasons.append(f"{eid}: missing passed smoke result in .rd_queue/results")
        if not full_hits:
            reasons.append(f"{eid}: missing passed full result in .rd_queue/results")

    return (len(reasons) == 0), reasons


def _write_audit_md(
    out_path: Path,
    *,
    plan_path: Path,
    ledger_path: Path,
    results_dir: Path,
    claims: List[Claim],
    ledger: Dict[str, ExperimentRow],
    results: Dict[str, List[ResultHit]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Tuple[str, str, str, str, str, str]] = []
    unproved: List[Tuple[Claim, List[str]]] = []

    for claim in claims:
        ok, reasons = _claim_status(claim, ledger=ledger, results=results)
        if not ok:
            unproved.append((claim, reasons))

        evid = ", ".join(claim.evidence) if claim.evidence else "-"

        smoke_ok = all((ledger.get(e) and ledger[e].smoke_checked) for e in claim.evidence) if claim.evidence else False
        full_ok = all((ledger.get(e) and ledger[e].full_checked) for e in claim.evidence) if claim.evidence else False
        res_smoke_ok = all(bool(_find_passed_hits(results.get(e, []), kind="smoke")) for e in claim.evidence) if claim.evidence else False
        res_full_ok = all(bool(_find_passed_hits(results.get(e, []), kind="full")) for e in claim.evidence) if claim.evidence else False

        rows.append(
            (
                claim.claim_id,
                claim.text,
                evid,
                f"{'Y' if smoke_ok else 'N'}/{'Y' if full_ok else 'N'}",
                f"{'Y' if res_smoke_ok else 'N'}/{'Y' if res_full_ok else 'N'}",
                "PROVED" if ok else "NOT PROVED",
            )
        )

    total = len(claims)
    proved = total - len(unproved)

    lines: List[str] = []
    lines.append("# Proof Audit\n")
    lines.append(f"Generated at (UTC): `{_utc_now_iso()}`\n")

    lines.append("\n## Inputs\n")
    lines.append(f"- plan: `{plan_path}`\n")
    lines.append(f"- ledger: `{ledger_path}`\n")
    lines.append(f"- results_dir: `{results_dir}`\n")

    lines.append("\n## Summary\n")
    lines.append(f"- total_claims: **{total}**\n")
    lines.append(f"- proved: **{proved}**\n")
    lines.append(f"- not_proved: **{len(unproved)}**\n")

    lines.append("\n## Claims\n")
    lines.append("| Claim | Text | Evidence | Ledger smoke/full | Results smoke/full | Status |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for cid, text, evid, lsf, rsf, status in rows:
        lines.append(
            f"| {_escape_md_cell(cid)} | {_escape_md_cell(text)} | {_escape_md_cell(evid)} |"
            f" {_escape_md_cell(lsf)} | {_escape_md_cell(rsf)} | {_escape_md_cell(status)} |\n"
        )

    lines.append("\n## Not Proved\n\n")
    if not unproved:
        lines.append("- (empty)\n")
    else:
        for claim, reasons in unproved:
            lines.append(f"- **{_escape_md_cell(claim.claim_id)}**: {_escape_md_cell(claim.text)}\n")
            if claim.evidence:
                lines.append(f"  - Evidence: {', '.join(_escape_md_cell(e) for e in claim.evidence)}\n")
            for r in reasons:
                lines.append(f"  - Gap: {_escape_md_cell(r)}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Proof-audit docs/plan.md claims against docs/experiment.md + .rd_queue/results."
    )
    parser.add_argument("--plan", default="docs/plan.md", help="Path to docs/plan.md")
    parser.add_argument("--ledger", default="docs/experiment.md", help="Path to docs/experiment.md")
    parser.add_argument("--results-dir", default=".rd_queue/results", help="Directory containing rdq result JSONs")
    parser.add_argument("--out", default="docs/proof_audit.md", help="Output markdown path")
    args = parser.parse_args()

    plan_path = Path(args.plan)
    ledger_path = Path(args.ledger)
    results_dir = Path(args.results_dir)
    out_path = Path(args.out)

    if not plan_path.exists():
        print(f"[ERR] plan not found: {plan_path}")
        sys.exit(1)
    if not ledger_path.exists():
        print(f"[ERR] ledger not found: {ledger_path}")
        sys.exit(1)

    claims = _parse_plan_claims(plan_path)
    ledger = _parse_experiment_table(ledger_path)
    results = _load_results(results_dir)

    _write_audit_md(
        out_path,
        plan_path=plan_path,
        ledger_path=ledger_path,
        results_dir=results_dir,
        claims=claims,
        ledger=ledger,
        results=results,
    )
    print(f"[OK] wrote {out_path}")

    # Exit non-zero if any claim is not proved.
    n_unproved = 0
    for c in claims:
        ok, _ = _claim_status(c, ledger=ledger, results=results)
        if not ok:
            n_unproved += 1
    if n_unproved:
        print(f"[ERR] not proved: {n_unproved} claim(s)")
        sys.exit(2)


if __name__ == "__main__":
    main()
