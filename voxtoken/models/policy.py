from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..schemas import TokenFeatures
from ..torch_compat import Module, torch


class SplitPolicy(Module):
    """Learned split policy interface (contextual bandit / offline RL)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.mode = str(cfg.get("mode", "heuristic"))
        self.use_torch = False
        self.torch_model = None
        self.torch_input_dim = 4
        self.torch_feature_norm = None

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

                model_path = payload.get("model_path", None)
                model_cfg = payload.get("model", {}) if isinstance(payload, dict) else {}
                feat_norm = payload.get("feature_norm", None) if isinstance(payload, dict) else None
                if isinstance(feat_norm, dict) and str(feat_norm.get("type", "")).strip().lower() == "zscore":
                    mean = feat_norm.get("mean", None)
                    std = feat_norm.get("std", None)
                    if isinstance(mean, list) and isinstance(std, list) and len(mean) == len(std) and len(mean) > 0:
                        try:
                            mean_f = [float(x) for x in mean]
                            std_f = [float(x) if float(x) > 0.0 else 1.0 for x in std]
                            self.torch_feature_norm = {"type": "zscore", "mean": mean_f, "std": std_f}
                        except Exception:
                            self.torch_feature_norm = None
                if isinstance(model_path, str) and model_path.strip() and torch is not None:
                    mp = Path(model_path.strip())
                    if not mp.is_absolute():
                        mp = p.parent / mp
                    if mp.exists():
                        try:
                            from .policy_mlp import PolicyMLP
                        except Exception:
                            PolicyMLP = None  # type: ignore[assignment]

                        if PolicyMLP is not None:
                            try:
                                input_dim = int((model_cfg or {}).get("input_dim", 4))
                            except Exception:
                                input_dim = 4
                            raw_hidden = (model_cfg or {}).get("hidden_dims", [64, 64])
                            if not isinstance(raw_hidden, list):
                                raw_hidden = [64, 64]
                            hidden_dims = [int(x) for x in raw_hidden if int(x) > 0] or [64, 64]
                            activation = str((model_cfg or {}).get("activation", "relu"))

                            m = PolicyMLP(input_dim=int(input_dim), hidden_dims=list(hidden_dims), activation=str(activation))
                            state = torch.load(str(mp), map_location="cpu")
                            if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
                                state = state["state_dict"]
                            if isinstance(state, dict):
                                m.load_state_dict(state)
                                m.eval()
                                self.torch_model = m
                                self.torch_input_dim = int(input_dim)
                                self.use_torch = True

    def score(self, feats: List[TokenFeatures]) -> List[Tuple[int, float]]:
        """Returns (token_id, score)."""
        if self.use_torch and torch is not None and self.torch_model is not None and feats:
            dim = int(getattr(self, "torch_input_dim", 4) or 4)
            # Keep this in sync with train_policy_torch._extract_feature_vector().
            # We support additive growth in feature dims for new experiments.
            dim = max(4, min(32, int(dim)))
            fn = getattr(self, "torch_feature_norm", None)
            mean = None
            std = None
            if isinstance(fn, dict) and str(fn.get("type", "")).strip().lower() == "zscore":
                m = fn.get("mean", None)
                s = fn.get("std", None)
                if isinstance(m, list) and isinstance(s, list) and len(m) == len(s) and len(m) == int(dim):
                    mean = [float(x) for x in m]
                    std = [float(x) if float(x) > 0.0 else 1.0 for x in s]

            def vec(f: TokenFeatures) -> List[float]:
                base = [
                    float(f.recon_error),
                    float(f.evidence_entropy),
                    float(f.citation_pressure),
                    float(f.history_splits),
                ]
                if dim >= 5:
                    base.append(float(getattr(f, "center_x_mm", 0.0)))
                if dim >= 6:
                    base.append(float(getattr(f, "center_y_mm", 0.0)))
                if dim >= 7:
                    base.append(float(getattr(f, "center_z_mm", 0.0)))
                if dim >= 8:
                    base.append(float(getattr(f, "mean_intensity", 0.0)))
                if dim >= 9:
                    base.append(float(getattr(f, "max_intensity", 0.0)))
                if dim >= 10:
                    base.append(float(getattr(f, "level", 0.0)))
                if dim >= 11:
                    base.append(float(getattr(f, "box_dx_mm", 0.0)))
                if dim >= 12:
                    base.append(float(getattr(f, "box_dy_mm", 0.0)))
                if dim >= 13:
                    base.append(float(getattr(f, "box_dz_mm", 0.0)))
                if dim >= 14:
                    base.append(float(getattr(f, "box_volume_mm3", 0.0)))
                if dim >= 15:
                    base.append(float(getattr(f, "step_idx", 0.0)))
                if dim >= 16:
                    base.append(float(getattr(f, "budget_B", 0.0)))
                if dim >= 17:
                    base.append(float(getattr(f, "mean_intensity_z", 0.0)))
                if dim >= 18:
                    base.append(float(getattr(f, "max_intensity_z", 0.0)))
                v = base[:dim] + [0.0 for _ in range(max(0, dim - len(base)))]
                if mean is not None and std is not None:
                    v = [(float(v[i]) - float(mean[i])) / float(std[i]) for i in range(int(dim))]
                return v

            x = torch.tensor([vec(f) for f in feats], dtype=torch.float32)
            with torch.no_grad():
                y = self.torch_model(x)
            scores = [float(v) for v in y.detach().cpu().view(-1).tolist()]
            scored = [(int(f.token_id), float(s)) for f, s in zip(feats, scores)]
            scored.sort(key=lambda x: (-float(x[1]), int(x[0])))
            return scored

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
