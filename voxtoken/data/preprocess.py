from __future__ import annotations

import argparse
from typing import Any, Dict


def preprocess(cfg: Dict[str, Any]) -> None:
    """Preprocess volumes/reports (resample, windowing, caching, splits)."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset preprocess (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    print(f"[placeholder] preprocess is not implemented yet. config={args.config}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
