"""Mean-ablation blockers (B1) for residual-stream directions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Sequence, Union

import torch

from sfo.hooks import _resolve_layers


@contextmanager
def mean_ablate_batched(
    model: Any,
    layer_idxs: Sequence[int],
    directions: Dict[int, torch.Tensor],
    mus: Dict[int, float],
    start_positions: torch.Tensor,
) -> Iterator[None]:
    """
    Directional mean-ablation at listed layers for positions ≥ start (padded coords).

    h ← h − (h·d̂ − μ) d̂   where d̂ is unit-normalized.

    Tracks absolute positions across KV-cached generate steps (same as
    add_to_residual_batched). Removes all hooks in finally.
    Composes with steering hooks at other layers.
    """
    if len(layer_idxs) == 0:
        yield
        return

    layers = _resolve_layers(model)
    starts = start_positions
    state = {"abs_pos": 0}
    handles = []

    # Pre-place directions on a dict; move to device lazily in hook
    dir_cpu = {int(l): directions[int(l)].detach().float().cpu() for l in layer_idxs}
    mu_map = {int(l): float(mus[int(l)]) for l in layer_idxs}

    def make_hook(layer_idx: int):
        def hook(_module: Any, _input: Any, output: Any) -> Any:
            if isinstance(output, tuple):
                hidden = output[0]
                rest = output[1:]
            else:
                hidden = output
                rest = None

            B, T, _D = hidden.shape
            device = hidden.device
            dtype = hidden.dtype
            d = dir_cpu[layer_idx].to(device=device, dtype=torch.float32)
            # If direction is ~0, identity
            if float(d.norm().item()) < 1e-12:
                state_note = T  # still advance? shared state — only first hook should advance
                # Don't advance here; outer shared counter advanced once per forward via first layer
                if rest is None:
                    return hidden
                return (hidden,) + rest

            mu = mu_map[layer_idx]
            st = starts.to(device=device, dtype=torch.long)
            abs_idx = torch.arange(
                state["abs_pos"], state["abs_pos"] + T, device=device, dtype=torch.long
            )
            mask = (abs_idx.unsqueeze(0) >= st.unsqueeze(1)).to(dtype=torch.float32)  # [B,T]

            h_f = hidden.float()
            proj = (h_f * d.view(1, 1, -1)).sum(dim=-1)  # [B,T]
            delta = (proj - mu) * mask  # [B,T]
            modified = h_f - delta.unsqueeze(-1) * d.view(1, 1, -1)
            modified = modified.to(dtype=dtype)

            if rest is None:
                return modified
            return (modified,) + rest

        return hook

    # Advance abs_pos only after the last hooked layer in this forward so all
    # layers see the same absolute positions for the current chunk.
    last_layer = int(sorted(layer_idxs)[-1])

    def make_hook_advancing(layer_idx: int, advance: bool):
        base = make_hook(layer_idx)

        def hook(module, inp, output):
            out = base(module, inp, output)
            if advance:
                if isinstance(output, tuple):
                    T = output[0].shape[1]
                else:
                    T = output.shape[1]
                state["abs_pos"] += T
            return out

        return hook

    for li in layer_idxs:
        advance = int(li) == last_layer
        handles.append(
            layers[int(li)].register_forward_hook(make_hook_advancing(int(li), advance))
        )

    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def mean_ablate_from_l0(
    model: Any,
    l0: int,
    directions: Dict[int, torch.Tensor],
    mus: Dict[int, float],
    start_positions: torch.Tensor,
    max_layer: int = 61,
) -> Iterator[None]:
    """Apply mean-ablation at every layer l with l0 ≤ l ≤ max_layer that has a direction."""
    layer_idxs = [l for l in range(int(l0), int(max_layer) + 1) if l in directions]
    with mean_ablate_batched(model, layer_idxs, directions, mus, start_positions):
        yield
