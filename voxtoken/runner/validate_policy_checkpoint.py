from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_WEIGHTS = {
    "recon_error": 1.0,
    "evidence_entropy": 1.0,
    "citation_pressure": 1.0,
    "history_splits": -1.0,
}


def validate_policy_checkpoint(payload: Dict[str, Any], *, require_nondefault_weights: bool) -> List[str]:
    errors: List[str] = []

    if str(payload.get("stage", "")).strip() != "P":
        errors.append("checkpoint.stage must be 'P'")

    if not str(payload.get("run_id", "")).strip():
        errors.append("checkpoint.run_id missing")

    weights = payload.get("weights", None)
    if not isinstance(weights, dict):
        errors.append("checkpoint.weights missing or not a dict")
        return errors

    required_keys = ["recon_error", "evidence_entropy", "citation_pressure", "history_splits"]
    for k in required_keys:
        if k not in weights:
            errors.append(f"checkpoint.weights missing key: {k}")

    fit_meta = payload.get("fit_meta", None)
    if not isinstance(fit_meta, dict) or str(fit_meta.get("fit", "")).strip() != "dataset":
        errors.append("checkpoint.fit_meta.fit must be 'dataset'")

    if require_nondefault_weights and isinstance(weights, dict):
        nondefault = False
        for k, v0 in DEFAULT_WEIGHTS.items():
            try:
                v = float(weights.get(k, v0))
            except Exception:
                continue
            if abs(float(v) - float(v0)) > 1e-6:
                nondefault = True
                break
        if not nondefault:
            errors.append("checkpoint.weights are all default; require non-default weights")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a policy checkpoint JSON file.")
    parser.add_argument("--in", dest="checkpoint_json", required=True, help="Path to checkpoint.json")
    parser.add_argument("--require-nondefault-weights", action="store_true")
    args = parser.parse_args()

    p = Path(args.checkpoint_json)
    if not p.exists():
        print(f"[ERR] missing checkpoint: {p}", file=sys.stderr)
        sys.exit(2)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        print("[ERR] checkpoint JSON must be an object", file=sys.stderr)
        sys.exit(2)

    errors = validate_policy_checkpoint(payload, require_nondefault_weights=bool(args.require_nondefault_weights))
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)
    print("[OK] policy checkpoint validated")


if __name__ == "__main__":
    main()

