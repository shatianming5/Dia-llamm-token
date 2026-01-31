from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def train_evidence_head(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Stage E: evidence head training (skeleton: stdlib-only code->finding mapping)."""
    cfg = dict(cfg or {})
    train_cfg = cfg.get("train", {})
    if "out_dir" not in cfg and isinstance(train_cfg, dict) and train_cfg.get("save_dir"):
        cfg["out_dir"] = train_cfg.get("save_dir")

    out_root = Path(str(cfg.get("out_dir", "outputs/train_evidence")))
    out_root.mkdir(parents=True, exist_ok=True)

    run_id = str(cfg.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_ckpt = cfg.get("tokenizer_checkpoint_path") or cfg.get("tokenizer_checkpoint")
    k = 4
    if tokenizer_ckpt:
        p = Path(str(tokenizer_ckpt))
        if p.exists():
            payload = _load_json(p)
            cb = payload.get("codebook", {}) if isinstance(payload, dict) else {}
            if isinstance(cb, dict) and "k" in cb:
                try:
                    k = int(cb.get("k", k))
                except Exception:
                    k = int(k)

    mapping_cfg = dict(cfg.get("mapping", {}))
    label_low = str(mapping_cfg.get("label_low", "normal"))
    label_high = str(mapping_cfg.get("label_high", "nodule"))
    threshold = int(mapping_cfg.get("threshold", max(1, int(k) // 2)))

    code_to_finding: Dict[str, str] = {}
    for code in range(max(1, int(k))):
        code_to_finding[str(int(code))] = label_low if int(code) < int(threshold) else label_high

    checkpoint = {
        "stage": "E",
        "run_id": run_id,
        "tokenizer_checkpoint_path": str(tokenizer_ckpt) if tokenizer_ckpt else None,
        "code_to_finding": code_to_finding,
    }

    (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"mapping_size": int(len(code_to_finding)), "step": 0}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return {"status": "ok", "out_dir": str(out_root), "run_dir": str(run_dir), "checkpoint_path": str(run_dir / "checkpoint.json")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage E: train evidence head (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    import yaml  # pyyaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dict(cfg or {})
    result = train_evidence_head(cfg)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
