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
    raise NotImplementedError

