"""Mean-ablation blockers (B1) for residual-stream directions."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Sequence

import torch

from sfo.hooks import _resolve_layers


def _split_output(output: Any):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _join_output(hidden: torch.Tensor, rest):
    if rest is None:
        return hidden
    return (hidden,) + rest


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

    Tracks absolute positions across KV-cached generate steps.
    Removes all hooks in finally. Composes with steering at other layers.
    """
    layer_idxs = [int(l) for l in layer_idxs]
    if not layer_idxs:
        yield
        return

    layers = _resolve_layers(model)
    state = {"abs_pos": 0}
    handles: List[Any] = []

    dir_cpu = {
        int(l): directions[int(l)].detach().float().reshape(-1).cpu() for l in layer_idxs
    }
    mu_map = {int(l): float(mus[int(l)]) for l in layer_idxs}
    starts_cpu = start_positions.detach().long().cpu()

    last_layer = max(layer_idxs)

    def make_hook(layer_idx: int):
        def hook(_module: Any, _input: Any, output: Any) -> Any:
            hidden, rest = _split_output(output)
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise TypeError(
                    f"Expected layer output [B,T,D], got {type(hidden)} "
                    f"shape={getattr(hidden, 'shape', None)}"
                )

            B, T, D = hidden.shape
            d = dir_cpu[layer_idx]
            if d.numel() != D:
                raise ValueError(f"direction dim {d.numel()} != hidden {D}")

            if float(d.norm().item()) < 1e-12:
                out = output
            else:
                device = hidden.device
                dtype = hidden.dtype
                d_dev = d.to(device=device, dtype=torch.float32)
                st = starts_cpu.to(device=device)
                abs_idx = torch.arange(
                    state["abs_pos"],
                    state["abs_pos"] + T,
                    device=device,
                    dtype=torch.long,
                )
                mask = abs_idx.unsqueeze(0) >= st.unsqueeze(1)  # [B,T]
                h_f = hidden.float()
                proj = (h_f * d_dev.view(1, 1, -1)).sum(dim=-1)
                delta = torch.where(
                    mask, proj - mu_map[layer_idx], torch.zeros_like(proj)
                )
                modified = h_f - delta.unsqueeze(-1) * d_dev.view(1, 1, -1)
                out = _join_output(modified.to(dtype=dtype), rest)

            if layer_idx == last_layer:
                state["abs_pos"] += T
            return out

        return hook

    for li in layer_idxs:
        handles.append(layers[li].register_forward_hook(make_hook(li)))

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
    layer_idxs = [l for l in range(int(l0), int(max_layer) + 1) if int(l) in directions]
    with mean_ablate_batched(model, layer_idxs, directions, mus, start_positions):
        yield
