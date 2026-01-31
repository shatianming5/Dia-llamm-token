from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict


def _load_metrics(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".jsonl":
        last = None
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            last = line
        if last is None:
            raise ValueError("metrics.jsonl is empty")
        obj = json.loads(last)
    else:
        obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError("metrics must be a JSON object")
    return obj


def validate_tokenizer_train_metrics(
    metrics: Dict[str, Any],
    *,
    require_perplexity_ge: float | None = None,
    require_codebook_used_ge: int | None = None,
) -> list[str]:
    errors: list[str] = []

    perplexity = metrics.get("perplexity", metrics.get("codebook_perplexity", None))
    if perplexity is None and isinstance(metrics.get("codebook", None), dict):
        perplexity = metrics["codebook"].get("perplexity", None)

    if require_perplexity_ge is not None:
        try:
            p = float(perplexity)
        except Exception:
            errors.append("perplexity is required but missing or not a number")
        else:
            if not math.isfinite(p):
                errors.append("perplexity must be finite")
            elif p < float(require_perplexity_ge):
                errors.append(f"perplexity({p}) < required({float(require_perplexity_ge)})")

    if require_codebook_used_ge is not None:
        used = metrics.get("codebook_used", None)
        if used is None and isinstance(metrics.get("codebook", None), dict):
            used = metrics["codebook"].get("used", None)
        try:
            u = int(used)
        except Exception:
            errors.append("codebook_used is required but missing or not an int")
        else:
            if u < int(require_codebook_used_ge):
                errors.append(f"codebook_used({u}) < required({int(require_codebook_used_ge)})")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tokenizer training metrics (perplexity/codebook usage).")
    parser.add_argument("--in", dest="metrics_path", required=True, help="Path to metrics.json or metrics.jsonl")
    parser.add_argument("--require-perplexity-ge", type=float, default=None)
    parser.add_argument("--require-codebook-used-ge", type=int, default=None)
    args = parser.parse_args()

    path = Path(args.metrics_path)
    metrics = _load_metrics(path)
    errors = validate_tokenizer_train_metrics(
        metrics,
        require_perplexity_ge=args.require_perplexity_ge,
        require_codebook_used_ge=args.require_codebook_used_ge,
    )
    if errors:
        for e in errors:
            print(f"[ERR] {e}")
        sys.exit(1)

    p = metrics.get("perplexity", None)
    used = metrics.get("codebook_used", None)
    print(f"[OK] tokenizer metrics validated (perplexity={p}, codebook_used={used})")


if __name__ == "__main__":
    main()

