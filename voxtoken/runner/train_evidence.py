from __future__ import annotations

import argparse
from typing import Any, Dict


def train_evidence_head(cfg: Dict[str, Any]) -> None:
    """Stage E: evidence head training (token -> structured evidence)."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage E: train evidence head (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    print(f"[placeholder] train_evidence_head is not implemented yet. config={args.config}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
