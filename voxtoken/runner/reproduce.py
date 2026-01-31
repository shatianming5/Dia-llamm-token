from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


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


def _strip_markdown_code(text: str) -> str:
    s = str(text or "").strip()
    if not s or s == "-":
        return ""
    if s.startswith("`") and s.endswith("`") and len(s) >= 2:
        s = s[1:-1]
    return s.strip()


def _parse_experiment_table(ledger_path: Path) -> Dict[str, Dict[str, str]]:
    """
    Parse the runnable experiment table in docs/experiment.md and return:
      { "E####": {header -> cell, ...}, ... }
    """
    lines = ledger_path.read_text(encoding="utf-8").splitlines()

    header_cells: List[str] = []
    rows: Dict[str, Dict[str, str]] = {}

    in_table = False
    for line in lines:
        if not in_table:
            if line.strip().startswith("| ID |"):
                header_cells = [c.strip() for c in line.strip().strip("|").split("|")]
                in_table = True
            continue

        # table ends at the next markdown section
        if line.strip().startswith("## "):
            break
        if not line.strip():
            continue
        if line.strip().startswith("|---"):
            continue
        if not line.strip().startswith("|"):
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < len(header_cells):
            continue
        row = {header_cells[i]: cells[i] for i in range(len(header_cells))}
        exp_id = _normalize_exp_id(row.get("ID", ""))
        if re.fullmatch(r"E\d{4}", exp_id):
            rows[exp_id] = row

    return rows


_STACK_ENV = "VOXTOKEN_REPRODUCE_STACK"
_MAX_NESTED_REPRODUCE_DEPTH = 20


def _load_reproduce_stack() -> List[str]:
    raw = str(os.environ.get(_STACK_ENV, "")).strip()
    if not raw:
        return []
    try:
        payload = json.loads(raw)
        if isinstance(payload, list):
            out: List[str] = []
            for x in payload:
                s = _normalize_exp_id(str(x))
                if re.fullmatch(r"E\d{4}", s):
                    out.append(s)
            return out
    except Exception:
        pass

    # Fallback for non-JSON stacks: comma-separated list.
    parts = [_normalize_exp_id(p.strip()) for p in raw.split(",") if p.strip()]
    return [p for p in parts if re.fullmatch(r"E\d{4}", p)]


def reproduce(exp_id: str, *, ledger: str, dry_run: bool) -> Dict[str, Any]:
    started_at = _utc_now_iso()
    ledger_path = Path(ledger)
    if not ledger_path.exists():
        raise FileNotFoundError(f"Ledger not found: {ledger_path}")

    exp_id_norm = _normalize_exp_id(exp_id)
    table = _parse_experiment_table(ledger_path)
    if exp_id_norm not in table:
        known = ", ".join(sorted(table.keys()))
        raise KeyError(f"Experiment '{exp_id}' not found in ledger. Known: {known}")

    row = table[exp_id_norm]
    if "1GPU script" not in row:
        raise KeyError("Ledger table missing required column: 1GPU script")

    cmd = _strip_markdown_code(row.get("1GPU script", ""))
    if not cmd:
        raise ValueError(f"Experiment {exp_id_norm} has empty 1GPU script.")

    stack = _load_reproduce_stack()
    if exp_id_norm in stack:
        cycle = " -> ".join([*stack, exp_id_norm])
        raise RuntimeError(f"Refusing to reproduce {exp_id_norm}: cycle detected ({cycle}).")
    if len(stack) >= _MAX_NESTED_REPRODUCE_DEPTH:
        raise RuntimeError(
            f"Refusing to reproduce {exp_id_norm}: nested reproduce depth exceeded "
            f"({_MAX_NESTED_REPRODUCE_DEPTH})."
        )
    next_stack = [*stack, exp_id_norm]

    print(f"[reproduce] exp_id={exp_id_norm}")
    print(f"[reproduce] ledger={ledger_path}")
    print(f"[reproduce] cmd={cmd}")
    if next_stack:
        print(f"[reproduce] stack_depth={len(next_stack)} stack={next_stack}")

    exit_code = 0
    if not dry_run:
        env = dict(os.environ)
        env[_STACK_ENV] = json.dumps(next_stack, ensure_ascii=False)
        proc = subprocess.run(["bash", "-lc", cmd], check=False, env=env)
        exit_code = int(proc.returncode or 0)

    ended_at = _utc_now_iso()
    return {
        "exp_id": exp_id_norm,
        "ledger": str(ledger_path),
        "command": cmd,
        "dry_run": bool(dry_run),
        "exit_code": int(exit_code),
        "started_at": started_at,
        "ended_at": ended_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce an experiment by E-ID, by executing its `1GPU script` in docs/experiment.md."
    )
    parser.add_argument("--exp", required=True, help="Experiment ID (E#### or legacy EXP-####)")
    parser.add_argument("--ledger", default="docs/experiment.md", help="Path to experiment ledger markdown")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved command but do not execute it")
    args = parser.parse_args()

    try:
        result = reproduce(args.exp, ledger=args.ledger, dry_run=bool(args.dry_run))
    except Exception as exc:  # noqa: BLE001
        print(f"[reproduce] ERROR: {exc}", file=sys.stderr)
        raise
    finally:
        # Always print a machine-readable footer for audit even when the command fails.
        # (If an exception is raised above, this footer may be absent.)
        pass

    print(json.dumps(result, ensure_ascii=False))
    sys.exit(int(result.get("exit_code", 1)))


if __name__ == "__main__":
    main()
