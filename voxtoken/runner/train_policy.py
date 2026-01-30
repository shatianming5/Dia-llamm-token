from __future__ import annotations

import argparse
from typing import Any, Dict


def train_policy(cfg: Dict[str, Any]) -> None:
    """Stage P: offline contextual bandit / policy training."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage P: train policy (placeholder).")
    parser.add_argument("--config", required=True, help="Path to a YAML config file")
    args = parser.parse_args()

    print(f"[placeholder] train_policy is not implemented yet. config={args.config}")
    raise SystemExit(2)


if __name__ == "__main__":
    main()
