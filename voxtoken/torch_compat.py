from __future__ import annotations

import importlib.util
from typing import Any, Callable, TypeVar

TCallable = TypeVar("TCallable", bound=Callable[..., Any])

_torch_spec = importlib.util.find_spec("torch")

if _torch_spec is None:
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[assignment]

    class Module:  # noqa: D101 - torch.nn.Module compatibility shim
        pass

    def no_grad() -> Callable[[TCallable], TCallable]:
        def decorator(fn: TCallable) -> TCallable:
            return fn

        return decorator
else:
    import torch  # type: ignore

    from torch import Tensor  # type: ignore
    from torch.nn import Module  # type: ignore

    no_grad = torch.no_grad

__all__ = [
    "Module",
    "Tensor",
    "no_grad",
    "torch",
]
