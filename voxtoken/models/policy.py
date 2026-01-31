from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..schemas import TokenFeatures
from ..torch_compat import Module


class SplitPolicy(Module):
    """Learned split policy interface (contextual bandit / offline RL)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "heuristic"))

        weights_cfg = dict(cfg.get("weights", {}))
        self.weights: Dict[str, float] = {
            "recon_error": float(weights_cfg.get("recon_error", 1.0)),
            "evidence_entropy": float(weights_cfg.get("evidence_entropy", 1.0)),
            "citation_pressure": float(weights_cfg.get("citation_pressure", 1.0)),
            # Use additive weights; the default matches the historical score:
            # recon + entropy + citation - history_splits  <=> history_splits weight = -1.0
            "history_splits": float(weights_cfg.get("history_splits", -1.0)),
        }

        ckpt_path = cfg.get("checkpoint_path") or cfg.get("checkpoint")
        if ckpt_path:
            p = Path(str(ckpt_path))
            if p.exists():
                payload = json.loads(p.read_text(encoding="utf-8"))
                w = payload.get("weights", {})
                if isinstance(w, dict):
                    for k in list(self.weights.keys()):
                        if k in w:
                            self.weights[k] = float(w[k])

    def score(self, feats: List[TokenFeatures]) -> List[Tuple[int, float]]:
        """Returns (token_id, score)."""
        w_recon = float(self.weights.get("recon_error", 1.0))
        w_ent = float(self.weights.get("evidence_entropy", 1.0))
        w_cit = float(self.weights.get("citation_pressure", 1.0))
        w_hist = float(self.weights.get("history_splits", -1.0))

        scored: List[Tuple[int, float]] = []
        for f in feats:
            s = (
                w_recon * float(f.recon_error)
                + w_ent * float(f.evidence_entropy)
                + w_cit * float(f.citation_pressure)
                + w_hist * float(f.history_splits)
            )
            scored.append((int(f.token_id), s))
        scored.sort(key=lambda x: (-float(x[1]), int(x[0])))
        return scored

    def export(self) -> Dict[str, Any]:
        return {"mode": self.mode, "weights": dict(self.weights)}


__all__ = [
    "SplitPolicy",
]
