from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def ingest(cfg: Dict[str, Any]) -> None:
    """Download/ingest datasets into a normalized on-disk layout."""
    out_dir = Path(str(cfg.get("out_dir", "data/raw")))
    out_dir.mkdir(parents=True, exist_ok=True)

    num_cases = int(cfg.get("num_cases", 1))
    manifest_path = Path(str(cfg.get("manifest_path", out_dir / "manifest.jsonl")))

    lines = []
    for i in range(num_cases):
        case_id = f"case-{i:04d}"
        vol_path = out_dir / f"{case_id}.vol.json"
        rpt_path = out_dir / f"{case_id}.txt"

        vol = {"shape_cdhw": [1, 8, 8, 8], "fill": 0.0}
        vol_path.write_text(json.dumps(vol, ensure_ascii=False, indent=2), encoding="utf-8")
        rpt_path.write_text("No findings.\n", encoding="utf-8")

        lines.append(
            json.dumps(
                {
                    "case_id": case_id,
                    "volume_path": str(vol_path),
                    "report_path": str(rpt_path),
                },
                ensure_ascii=False,
            )
        )

    manifest_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset ingest (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    import yaml  # pyyaml

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    cfg = dict(cfg or {})
    ingest(cfg)

    out_dir = str(cfg.get("out_dir", "data/raw"))
    manifest_path = str(cfg.get("manifest_path", str(Path(out_dir) / "manifest.jsonl")))
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": out_dir,
        "manifest_path": manifest_path,
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
