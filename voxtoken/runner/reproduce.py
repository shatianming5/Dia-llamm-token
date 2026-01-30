from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce an experiment by EXP-ID (placeholder).")
    parser.add_argument("--exp", required=True, help="Experiment ID, e.g. EXP-0100")
    args = parser.parse_args()

    msg = (
        f"[placeholder] EXP-ID: {args.exp}\n"
        "Reproduce is not implemented yet.\n"
        "For now, follow the commands and artifacts contract described in docs/experiment.md."
    )
    print(msg)


if __name__ == "__main__":
    main()

