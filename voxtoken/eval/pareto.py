from __future__ import annotations

from typing import Any, Dict, List, Tuple


def pareto_front(points: List[Tuple[float, float]]) -> List[int]:
    """
    Returns indices of non-dominated points for (correctness, cost) style plots.

    NOTE: Placeholder; define dominance convention in implementation.
    """
    front: List[int] = []
    for i, (corr_i, cost_i) in enumerate(points):
        dominated = False
        for j, (corr_j, cost_j) in enumerate(points):
            if i == j:
                continue
            better_or_equal = float(corr_j) >= float(corr_i) and float(cost_j) <= float(cost_i)
            strictly_better = float(corr_j) > float(corr_i) or float(cost_j) < float(cost_i)
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front
