"""Paired-twin steering and B1′ shift-removal blocker."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from sfo.hooks import _resolve_layers


def _split_output(output: Any):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def _join_output(hidden: torch.Tensor, rest):
    if rest is None:
        return hidden
    return (hidden,) + rest


def _hidden_size(model: Any) -> int:
    cfg = model.config
    if hasattr(cfg, "text_config"):
        return int(cfg.text_config.hidden_size)
    return int(cfg.hidden_size)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-12)


def duplicate_batch(
    input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
):
    """Stack steered then twin copies: [B,T] → [2B,T]."""
    ids = torch.cat([input_ids, input_ids], dim=0)
    attn = None
    if attention_mask is not None:
        attn = torch.cat([attention_mask, attention_mask], dim=0)
    return ids, attn


def steered_twin_starts(start_positions: torch.Tensor) -> torch.Tensor:
    return torch.cat([start_positions, start_positions], dim=0)


def steer_vectors_paired(vecs: torch.Tensor) -> torch.Tensor:
    """vecs [B,D] for steered rows → [2B,D] with zeros for twins."""
    return torch.cat([vecs, torch.zeros_like(vecs)], dim=0)


@contextmanager
def add_steering_paired(
    model: Any,
    layer_idx: int,
    vectors_2b: torch.Tensor,
    start_positions_2b: torch.Tensor,
) -> Iterator[None]:
    """Add vectors_2b at layer for positions ≥ start (padded abs coords)."""
    if vectors_2b.ndim != 2:
        raise ValueError(f"vectors must be [2B, D], got {tuple(vectors_2b.shape)}")
    layers = _resolve_layers(model)
    layer = layers[layer_idx]
    state = {"abs_pos": 0}

    def hook(_module: Any, _input: Any, output: Any) -> Any:
        hidden, rest = _split_output(output)
        _B, T, _D = hidden.shape
        device = hidden.device
        vecs = vectors_2b.to(device=device, dtype=hidden.dtype)
        starts = start_positions_2b.to(device=device, dtype=torch.long)
        abs_idx = torch.arange(
            state["abs_pos"], state["abs_pos"] + T, device=device, dtype=torch.long
        )
        mask = abs_idx.unsqueeze(0) >= starts.unsqueeze(1)
        modified = hidden + vecs.unsqueeze(1) * mask.unsqueeze(-1).to(hidden.dtype)
        state["abs_pos"] += T
        return _join_output(modified, rest)

    handle = layer.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def b1prime_shift_remove(
    model: Any,
    layer_idxs: Sequence[int],
    directions: Dict[int, torch.Tensor],
    start_positions_2b: torch.Tensor,
    n_steered: int,
) -> Iterator[None]:
    """
    B1′ on steered rows only:
      h_i ← h_i − ((h_i − h_{i+B}) · d̂_l) d̂_l
    Batch layout: [steered (n_steered) | twin (n_steered)].
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
    starts_cpu = start_positions_2b.detach().long().cpu()
    last_layer = max(layer_idxs)
    B_half = int(n_steered)

    def make_hook(layer_idx: int):
        def hook(_module: Any, _input: Any, output: Any) -> Any:
            hidden, rest = _split_output(output)
            if not torch.is_tensor(hidden) or hidden.ndim != 3:
                raise TypeError(
                    f"Expected [B,T,D], got {type(hidden)} "
                    f"shape={getattr(hidden, 'shape', None)}"
                )
            Bt, T, D = hidden.shape
            if Bt != 2 * B_half:
                raise ValueError(f"B1′ expects batch {2 * B_half}, got {Bt}")
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
                mask = abs_idx.unsqueeze(0) >= st.unsqueeze(1)
                h_f = hidden.float()
                h_s, h_t = h_f[:B_half], h_f[B_half:]
                proj = ((h_s - h_t) * d_dev.view(1, 1, -1)).sum(dim=-1)
                corr = torch.where(mask[:B_half], proj, torch.zeros_like(proj))
                h_s2 = h_s - corr.unsqueeze(-1) * d_dev.view(1, 1, -1)
                modified = torch.cat([h_s2, h_t], dim=0).to(dtype=dtype)
                out = _join_output(modified, rest)

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
def b1prime_from_l0(
    model: Any,
    l0: int,
    directions: Dict[int, torch.Tensor],
    start_positions_2b: torch.Tensor,
    n_steered: int,
    max_layer: int = 61,
) -> Iterator[None]:
    layer_idxs = [l for l in range(int(l0), int(max_layer) + 1) if int(l) in directions]
    with b1prime_shift_remove(
        model, layer_idxs, directions, start_positions_2b, n_steered
    ):
        yield


def _top_k_top_p_filter(
    logits: torch.Tensor, top_k: int, top_p: float
) -> torch.Tensor:
    filtered = logits.clone()
    if top_k > 0:
        k = min(top_k, filtered.size(-1))
        vals, _ = torch.topk(filtered, k, dim=-1)
        cutoff = vals[:, -1].unsqueeze(-1)
        filtered = torch.where(
            filtered < cutoff, torch.full_like(filtered, float("-inf")), filtered
        )
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(filtered, descending=True, dim=-1)
        probs = F.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
        filtered = torch.full_like(filtered, float("-inf"))
        filtered.scatter_(1, sorted_idx, sorted_logits)
    return filtered


@torch.no_grad()
def paired_generate(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_positions: torch.Tensor,
    steer_vecs: Optional[torch.Tensor],
    steer_layer: int,
    blocker_dirs: Optional[Dict[int, torch.Tensor]],
    l0: Optional[int],
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 64,
    top_p: float = 0.95,
    seed: int = 0,
    do_sample: bool = True,
    eos_ids: Sequence[int] = (1, 106),
    max_layer: int = 61,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Manual paired sampling with KV cache; twins teacher-forced on steered tokens.

    Inputs are steered-half only ([B,...]). Returns (steered_ids, twin_ids) lists.
    """
    device = input_ids.device
    B = input_ids.shape[0]
    if steer_vecs is None:
        steer_vecs = torch.zeros(
            B, _hidden_size(model), device=device, dtype=torch.bfloat16
        )

    ids_2b, attn_2b = duplicate_batch(input_ids, attention_mask)
    starts_2b = steered_twin_starts(start_positions).to(device)
    vecs_2b = steer_vectors_paired(steer_vecs.to(device=device, dtype=torch.bfloat16))

    gen = torch.Generator()
    gen.manual_seed(int(seed))

    eos_set = {int(x) for x in eos_ids}
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    steer_cm = add_steering_paired(model, steer_layer, vecs_2b, starts_2b)
    blk_cm = (
        b1prime_from_l0(model, int(l0), blocker_dirs, starts_2b, B, max_layer=max_layer)
        if blocker_dirs is not None and l0 is not None
        else nullcontext()
    )

    steered_seqs = [input_ids[i].detach().cpu().clone() for i in range(B)]
    twin_seqs = [input_ids[i].detach().cpu().clone() for i in range(B)]

    with steer_cm, blk_cm:
        out = model(input_ids=ids_2b, attention_mask=attn_2b, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]

        for _step in range(max_new_tokens):
            logits_s = logits[:B].float()
            if do_sample:
                scaled = logits_s / float(temperature)
                filtered = _top_k_top_p_filter(scaled, top_k=top_k, top_p=top_p)
                probs = F.softmax(filtered, dim=-1)
                next_tok = (
                    torch.multinomial(probs.cpu(), num_samples=1, generator=gen)
                    .squeeze(-1)
                    .to(device)
                )
            else:
                next_tok = torch.argmax(logits_s, dim=-1)

            for i in range(B):
                if finished[i]:
                    next_tok[i] = int(steered_seqs[i][-1].item())

            for i in range(B):
                if not finished[i]:
                    t = next_tok[i : i + 1].detach().cpu()
                    steered_seqs[i] = torch.cat([steered_seqs[i], t])
                    twin_seqs[i] = torch.cat([twin_seqs[i], t])
                    if int(next_tok[i].item()) in eos_set:
                        finished[i] = True

            if bool(finished.all()):
                break

            next_2b = torch.cat([next_tok, next_tok], dim=0).unsqueeze(-1)
            attn_2b = torch.cat(
                [
                    attn_2b,
                    torch.ones((2 * B, 1), device=device, dtype=attn_2b.dtype),
                ],
                dim=1,
            )
            out = model(
                input_ids=next_2b,
                attention_mask=attn_2b,
                past_key_values=past,
                use_cache=True,
            )
            past = out.past_key_values
            logits = out.logits[:, -1, :]

    return steered_seqs, twin_seqs


@torch.no_grad()
def paired_forward_logits(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_positions: torch.Tensor,
    steer_vecs: Optional[torch.Tensor],
    steer_layer: int,
    blocker_dirs: Optional[Dict[int, torch.Tensor]],
    l0: Optional[int],
    max_layer: int = 61,
) -> torch.Tensor:
    """Single forward on paired batch. Returns logits [2B, T, V]."""
    device = input_ids.device
    B = input_ids.shape[0]
    if steer_vecs is None:
        steer_vecs = torch.zeros(
            B, _hidden_size(model), device=device, dtype=torch.bfloat16
        )

    ids_2b, attn_2b = duplicate_batch(input_ids, attention_mask)
    starts_2b = steered_twin_starts(start_positions).to(device)
    vecs_2b = steer_vectors_paired(steer_vecs.to(device=device, dtype=torch.bfloat16))

    steer_cm = add_steering_paired(model, steer_layer, vecs_2b, starts_2b)
    blk_cm = (
        b1prime_from_l0(model, int(l0), blocker_dirs, starts_2b, B, max_layer=max_layer)
        if blocker_dirs is not None and l0 is not None
        else nullcontext()
    )
    with steer_cm, blk_cm:
        out = model(input_ids=ids_2b, attention_mask=attn_2b)
    return out.logits


@torch.no_grad()
def paired_capture_shifts(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    start_positions: torch.Tensor,
    steer_vecs: torch.Tensor,
    steer_layer: int,
    capture_layers: Sequence[int],
) -> Dict[int, torch.Tensor]:
    """
    Mean over positions ≥ start of (h_steered − h_twin) per layer.
    Returns {layer: [B, D] float32 CPU}.
    """
    device = input_ids.device
    B = input_ids.shape[0]
    ids_2b, attn_2b = duplicate_batch(input_ids, attention_mask)
    starts_2b = steered_twin_starts(start_positions).to(device)
    vecs_2b = steer_vectors_paired(steer_vecs.to(device=device, dtype=torch.bfloat16))

    layers = _resolve_layers(model)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def hook(_module: Any, _input: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[idx] = hidden.detach()

        return hook

    for idx in capture_layers:
        handles.append(layers[int(idx)].register_forward_hook(make_hook(int(idx))))
    try:
        with add_steering_paired(model, steer_layer, vecs_2b, starts_2b):
            model(input_ids=ids_2b, attention_mask=attn_2b)
    finally:
        for h in handles:
            h.remove()

    starts = start_positions.to(device)
    out: Dict[int, torch.Tensor] = {}
    for li, h in captured.items():
        h_s, h_t = h[:B], h[B:]
        T = h_s.shape[1]
        abs_idx = torch.arange(T, device=device)
        mask = abs_idx.unsqueeze(0) >= starts.unsqueeze(1)
        delta = (h_s - h_t).float()
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        out[int(li)] = ((delta * mask_f).sum(dim=1) / denom).cpu()
    return out
