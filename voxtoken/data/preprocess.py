from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .splits import assign_split as assign_split_deterministic


def preprocess(cfg: Dict[str, Any]) -> None:
    """Preprocess volumes/reports (resample, windowing, caching, splits)."""
    in_manifest = Path(str(cfg.get("in_manifest", "data/raw/manifest.jsonl")))
    out_dir = Path(str(cfg.get("out_dir", "data/processed")))
    out_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 0))
    split_cfg = dict(cfg.get("split", {"train": 0.8, "val": 0.1, "test": 0.1}))
    train_p = float(split_cfg.get("train", 0.8))
    val_p = float(split_cfg.get("val", 0.1))

    lines_in = [line for line in in_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]

    out_lines = []
    for line in lines_in:
        row = json.loads(line)
        case_id = str(row.get("case_id", ""))
        row["split"] = assign_split_deterministic(case_id, seed=int(seed), train_p=float(train_p), val_p=float(val_p))
        out_lines.append(json.dumps(row, ensure_ascii=False))

    out_manifest = Path(str(cfg.get("out_manifest", out_dir / "manifest.jsonl")))
    out_manifest.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "in_manifest": str(in_manifest),
        "out_manifest": str(out_manifest),
        "n": int(len(out_lines)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset preprocess (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    import yaml  # pyyaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dict(cfg or {})
    preprocess(cfg)

    out_dir = str(cfg.get("out_dir", "data/processed"))
    out_manifest = str(cfg.get("out_manifest", str(Path(out_dir) / "manifest.jsonl")))
    print(json.dumps({"out_dir": out_dir, "out_manifest": out_manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
