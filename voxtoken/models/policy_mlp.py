from __future__ import annotations

from typing import List

from ..torch_compat import torch

if torch is None:
    PolicyMLP = None  # type: ignore[assignment]
else:
    import torch.nn as nn  # type: ignore

    def _activation(name: str) -> nn.Module:
        s = str(name or "").strip().lower()
        if s == "tanh":
            return nn.Tanh()
        if s == "gelu":
            return nn.GELU()
        if s == "sigmoid":
            return nn.Sigmoid()
        return nn.ReLU()

    class PolicyMLP(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: List[int], activation: str = "relu") -> None:
            super().__init__()
            dims = [int(input_dim), *[int(x) for x in (hidden_dims or [])], 1]

            layers: List[nn.Module] = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
                layers.append(_activation(str(activation)))
            layers.append(nn.Linear(int(dims[-2]), int(dims[-1])))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # type: ignore[override]
            y = self.net(x)
            return y.view(-1)

