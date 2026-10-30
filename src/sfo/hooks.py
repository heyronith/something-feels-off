"""Residual-stream hook utilities (context-manager based)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import torch


def _resolve_layers(model: Any) -> torch.nn.ModuleList:
    """Locate decoder layers without asserting production layer count."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise AttributeError("Could not locate decoder layers on model")


def _apply_vector(
    hidden: torch.Tensor,
    vector: torch.Tensor,
    positions: Union[str, Sequence[int]],
) -> torch.Tensor:
    vec = vector.to(device=hidden.device, dtype=hidden.dtype)
    if positions == "all":
        return hidden + vec.view(1, 1, -1)
    modified = hidden.clone()
    idx = list(positions)
    modified[:, idx, :] = modified[:, idx, :] + vec.view(1, 1, -1)
    return modified


@contextmanager
def add_to_residual(
    model: Any,
    layer_idx: int,
    vector: torch.Tensor,
    positions: Union[str, Sequence[int]] = "all",
) -> Iterator[None]:
    """
    Add `vector` (shape [d_model], bf16) to layer `layer_idx` residual output.

    Handles tuple decoder-layer outputs: modifies output[0], returns the tuple.
    Removes the hook on exit, including on exceptions.
    """
    layers = _resolve_layers(model)
    layer = layers[layer_idx]

    def hook(_module: Any, _input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            modified = _apply_vector(hidden, vector, positions)
            return (modified,) + output[1:]
        return _apply_vector(output, vector, positions)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_residual(
    model: Any,
    layer_idxs: Sequence[int],
) -> Iterator[Dict[int, torch.Tensor]]:
    """
    Capture each listed layer's output hidden states (detached, CPU float32).
    """
    layers = _resolve_layers(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module: Any, _input: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach().to(dtype=torch.float32, device="cpu")

        return hook

    for idx in layer_idxs:
        handles.append(layers[idx].register_forward_hook(make_hook(int(idx))))
    try:
        yield captured
    finally:
        for h in handles:
            h.remove()
