from __future__ import annotations

import argparse
from typing import Any, Dict


def ingest(cfg: Dict[str, Any]) -> None:
    """Download/ingest datasets into a normalized on-disk layout."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset ingest (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    print(f"[placeholder] ingest is not implemented yet. config={args.config}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
