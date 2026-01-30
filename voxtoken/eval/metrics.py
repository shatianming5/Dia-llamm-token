from __future__ import annotations

from typing import Any, Dict


def compute_metrics(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Compute core metrics: correctness, grounding, unsupported, efficiency."""
    raise NotImplementedError

