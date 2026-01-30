from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..torch_compat import Module, Tensor, torch


@dataclass
class QuantizeResult:
    """Quantization outputs for VQ/RVQ-style tokenizers."""

    codes: Tensor
    embeddings: Tensor
    aux: Dict[str, Any]


class ResidualVectorQuantizer(Module):
    """Residual VQ (RVQ) interface placeholder."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg

    def quantize(self, feats: Tensor) -> QuantizeResult:
        codebook_size = int(self.cfg.get("codebook_size", 256))

        if torch is not None and hasattr(feats, "mean"):
            # Assume torch.Tensor-like
            emb = feats
            row_mean = emb.mean(dim=-1) if hasattr(emb, "dim") and int(emb.dim()) >= 2 else emb
            scaled = row_mean.abs() * float(codebook_size - 1)
            codes = scaled.long() % int(codebook_size)
            return QuantizeResult(codes=codes, embeddings=emb, aux={"codebook_size": codebook_size, "mode": "torch"})

        seq = list(feats)  # type: ignore[arg-type]
        codes_list = []
        for row in seq:
            row_seq = list(row) if hasattr(row, "__iter__") else [row]
            denom = float(len(row_seq)) if row_seq else 1.0
            mean = sum(float(x) for x in row_seq) / denom
            code = int(abs(mean) * float(codebook_size - 1)) % int(codebook_size)
            codes_list.append(code)

        return QuantizeResult(codes=codes_list, embeddings=feats, aux={"codebook_size": codebook_size, "mode": "python"})
