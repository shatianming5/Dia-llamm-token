from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def train_policy(cfg: Dict[str, Any]) -> None:
    """Stage P: offline contextual bandit / policy training."""
    out_root = Path(str(cfg.get("out_dir", "outputs/train_policy")))
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps({"stage": "P", "run_id": run_id}, ensure_ascii=False), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"loss": 0.0, "step": 0}) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage P: train policy (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    import yaml  # pyyaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dict(cfg or {})
    train_policy(cfg)
    print(json.dumps({"status": "ok", "out_dir": str(cfg.get("out_dir", "outputs/train_policy"))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
