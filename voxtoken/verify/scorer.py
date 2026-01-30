from __future__ import annotations

from typing import List

from ..schemas import Issue


def score_from_issues(
    slot_f1: float,
    issues: List[Issue],
    *,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 1.0,
) -> float:
    """Implements: V = SlotF1 - a#missing - b#inconsistency - c#overclaim - d#unsupported."""
    n_missing = 0
    n_incon = 0
    n_over = 0
    n_unsup = 0
    for issue in issues:
        if issue.type == "missing_slot":
            n_missing += 1
        elif issue.type == "inconsistency":
            n_incon += 1
        elif issue.type == "overclaim":
            n_over += 1
        elif issue.type == "unsupported":
            n_unsup += 1

    return float(slot_f1) - alpha * float(n_missing) - beta * float(n_incon) - gamma * float(n_over) - delta * float(n_unsup)
