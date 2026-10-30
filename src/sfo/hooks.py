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


@contextmanager
def add_to_residual_batched(
    model: Any,
    layer_idx: int,
    vectors: torch.Tensor,
    start_positions: torch.Tensor,
) -> Iterator[None]:
    """
    Batched residual injection from per-row start positions (padded coordinates).

    vectors: [B, d_model]
    start_positions: [B] int — absolute padded index; add at positions ≥ start,
    including every generated token across KV-cached decode steps.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be [B, d_model], got {tuple(vectors.shape)}")
    if start_positions.ndim != 1 or start_positions.shape[0] != vectors.shape[0]:
        raise ValueError("start_positions must be [B] matching vectors batch")

    layers = _resolve_layers(model)
    layer = layers[layer_idx]
    state = {"abs_pos": 0}

    def hook(_module: Any, _input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None

        B, T, _D = hidden.shape
        device = hidden.device
        vecs = vectors.to(device=device, dtype=hidden.dtype)
        starts = start_positions.to(device=device, dtype=torch.long)

        abs_idx = torch.arange(
            state["abs_pos"], state["abs_pos"] + T, device=device, dtype=torch.long
        )  # [T]
        # mask[b, t] True iff absolute position >= start_positions[b]
        mask = abs_idx.unsqueeze(0) >= starts.unsqueeze(1)  # [B, T]
        modified = hidden + vecs.unsqueeze(1) * mask.unsqueeze(-1).to(hidden.dtype)
        state["abs_pos"] += T

        if rest is None:
            return modified
        return (modified,) + rest

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()
