"""Feature-level clamp / sham interventions at post_feedforward_layernorm (WO-004).

Capture JumpReLU activations from ``pre_feedforward_layernorm`` output (fp32 columns),
then add Σ_f (target − a_f)·w_dec[f] to ``post_feedforward_layernorm`` output (fp32→bf16).
clampMLP replaces that output with another row's tensor.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor


def jumprelu(pre: Tensor, threshold: Tensor) -> Tensor:
    return torch.where(pre > threshold, torch.relu(pre), torch.zeros_like(pre))


@dataclass
class FeatureBundle:
    """Encode columns + decode rows for a small feature set at one layer."""

    layer: int
    feature_ids: List[int]
    w_enc: Tensor          # [d, F] fp32
    b_enc: Tensor          # [F] fp32
    threshold: Tensor      # [F] fp32
    w_dec: Tensor          # [F, d] fp32
    w_dec_norm: Tensor     # [F] fp32


def load_feature_bundle(
    folder: Union[str, Path],
    feature_ids: Sequence[int],
    device: Union[str, torch.device],
) -> FeatureBundle:
    """Load only the needed w_enc columns and w_dec rows via safetensors slices."""
    from safetensors import safe_open

    folder = Path(folder)
    ids = [int(f) for f in feature_ids]
    if not ids:
        d = 5376
        z = torch.zeros(d, 0, device=device, dtype=torch.float32)
        return FeatureBundle(
            layer=-1, feature_ids=[], w_enc=z, b_enc=z[0], threshold=z[0],
            w_dec=z.T.contiguous(), w_dec_norm=z[0],
        )
    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
        # w_enc [d, width], w_dec [width, d]
        cols, rows, b, thr, norms = [], [], [], [], []
        for fid in ids:
            we = f.get_slice("w_enc")[:, int(fid)].float()
            wd = f.get_slice("w_dec")[int(fid), :].float()
            cols.append(we)
            rows.append(wd)
            b.append(f.get_slice("b_enc")[int(fid)].float())
            thr.append(f.get_slice("threshold")[int(fid)].float())
            norms.append(wd.norm())
        w_enc = torch.stack(cols, dim=1).to(device)       # [d, F]
        w_dec = torch.stack(rows, dim=0).to(device)       # [F, d]
        b_enc = torch.stack(b).reshape(-1).to(device)
        threshold = torch.stack(thr).reshape(-1).to(device)
        w_dec_norm = torch.stack(norms).reshape(-1).to(device)
    # layer from folder name
    import re
    m = re.search(r"layer_(\d+)", folder.name)
    layer = int(m.group(1)) if m else -1
    return FeatureBundle(
        layer=layer, feature_ids=ids, w_enc=w_enc, b_enc=b_enc,
        threshold=threshold, w_dec=w_dec, w_dec_norm=w_dec_norm,
    )


def encode_features_fp32(x: Tensor, bundle: FeatureBundle) -> Tensor:
    """x [..., d] -> acts [..., F] fp32 JumpReLU."""
    if not bundle.feature_ids:
        return x.new_zeros(*x.shape[:-1], 0)
    xf = x.float()
    pre = xf @ bundle.w_enc + bundle.b_enc
    return jumprelu(pre, bundle.threshold)


def delta_from_targets(
    acts: Tensor,
    targets: Tensor,
    w_dec: Tensor,
) -> Tensor:
    """Σ_f (target − a)·w_dec[f] → [..., d] fp32.

    acts, targets: [..., F]; w_dec: [F, d].
    """
    if acts.shape[-1] == 0:
        return acts.new_zeros(*acts.shape[:-1], w_dec.shape[-1] if w_dec.numel() else 0)
    coeff = (targets - acts).float()                    # [..., F]
    return coeff @ w_dec.float()                        # [..., d]


def select_m_dla(
    a0: Dict[int, float],
    dla_raw: np.ndarray,
    gate_ids: Sequence[int],
    *,
    k_max: int = 50,
) -> Tuple[List[int], float, float]:
    """Greedy No-writer match on Σ a0·dla_raw vs gates. Returns (ids, K_frac, gate_total)."""
    gates = set(int(g) for g in gate_ids)
    gate_total = float(sum(a0.get(g, 0.0) * float(dla_raw[g]) for g in gates if a0.get(g, 0.0) > 0))
    if gate_total <= 0:
        return [], 0.0, gate_total
    cands = [
        f for f, a in a0.items()
        if f not in gates and a > 0 and float(dla_raw[f]) > 0
    ]
    cands.sort(key=lambda f: (-a0[f] * float(dla_raw[f]), f))
    chosen: List[int] = []
    acc = 0.0
    for f in cands:
        if len(chosen) >= k_max:
            break
        chosen.append(int(f))
        acc += a0[f] * float(dla_raw[f])
        if acc >= gate_total:
            break
    frac = acc / gate_total if gate_total > 0 else 0.0
    return chosen, frac, gate_total


def select_m_norm(
    a0: Dict[int, float],
    w_norm: np.ndarray,
    dla_normed: np.ndarray,
    gate_ids: Sequence[int],
    exclude: Sequence[int],
    *,
    p50_abs_dla: float,
    rng: np.random.Generator,
    k_max: int = 50,
) -> Tuple[List[int], float, float]:
    """Greedy match on Σ a0·‖w_dec‖ among low-|dla| non-M_dla features."""
    gates = set(int(g) for g in gate_ids)
    excl = set(int(f) for f in exclude) | gates
    gate_total = float(sum(a0.get(g, 0.0) * float(w_norm[g]) for g in gates if a0.get(g, 0.0) > 0))
    if gate_total <= 0:
        return [], 0.0, gate_total
    cands = [
        f for f, a in a0.items()
        if f not in excl and a > 0 and abs(float(dla_normed[f])) < p50_abs_dla
    ]
    order = np.array(cands, dtype=np.int64)
    if order.size:
        order = rng.permutation(order)
    chosen: List[int] = []
    acc = 0.0
    for f in order.tolist():
        if len(chosen) >= k_max:
            break
        chosen.append(int(f))
        acc += a0[int(f)] * float(w_norm[int(f)])
        if acc >= gate_total:
            break
    frac = acc / gate_total if gate_total > 0 else 0.0
    return chosen, frac, gate_total


@dataclass
class ClampPlan:
    """Per-batch intervention plan.

    ``target_row[b]`` is the batch index supplying target activations / MLP output
    for row b (−1 = no intervention of that kind).
    ``feature_mask[b]`` True → apply feature delta at layers in ``layers``.
    ``mlp_mask[b]`` True → replace post_ff_ln output with target_row's captured output.
    ``last_only[b]`` True → feature delta only at the last position.
    ``feature_id_sets[b]`` optional per-row subset of ``FeatureBundle.feature_ids`` to
    include in the clamp delta (``None`` = use all features in the bundle for that row).
    ``row_layers[b]`` optional per-row layer tuple for feature clamp (``None`` = ``layers``).
    """

    target_row: Tensor          # [B] long
    feature_mask: Tensor        # [B] bool
    mlp_mask: Tensor            # [B] bool
    last_only: Tensor           # [B] bool
    layers: Tuple[int, ...] = (45, 50)
    feature_id_sets: Optional[List[Optional[List[int]]]] = None
    row_layers: Optional[List[Optional[Tuple[int, ...]]]] = None


@contextmanager
def apply_feature_clamp(
    model: Any,
    bundles: Dict[int, FeatureBundle],
    plan: ClampPlan,
    *,
    capture_acts_out: Optional[Dict[int, Tensor]] = None,
    capture_post_out: Optional[Dict[int, Tensor]] = None,
) -> Iterator[Dict[str, Any]]:
    """Install pre/post FF LN hooks for feature clamp / sham / clampMLP.

    Yields a mutable state dict. On exit, removes hooks.
    If ``capture_acts_out`` / ``capture_post_out`` are provided, fills them with
    last-forward tensors keyed by layer (detached clones).
    """
    from sfo.hooks import _resolve_layers

    layers_mod = _resolve_layers(model)
    state: Dict[str, Any] = {"acts": {}, "post": {}, "x_pre": {}}
    handles = []

    # --- pre_ff_ln: store x and compute acts
    def make_pre_hook(layer_idx: int):
        def hook(_mod: Any, _args: Any, output: Tensor) -> None:
            x = output
            state["x_pre"][layer_idx] = x
            bun = bundles.get(layer_idx)
            if bun is None or not bun.feature_ids:
                state["acts"][layer_idx] = x.new_zeros(*x.shape[:2], 0)
            else:
                state["acts"][layer_idx] = encode_features_fp32(x, bun)
            if capture_acts_out is not None:
                capture_acts_out[layer_idx] = state["acts"][layer_idx].detach().clone()

        return hook

    # --- post_ff_ln: add delta or replace
    def make_post_hook(layer_idx: int):
        def hook(_mod: Any, _args: Any, output: Tensor) -> Tensor:
            B, T, _D = output.shape
            # Stash unmodified post for clampMLP target rows (before any delta).
            raw_key = f"raw_post_{layer_idx}"
            if raw_key not in state:
                state[raw_key] = output.detach().clone()
            out = output
            bun = bundles.get(layer_idx)
            acts = state["acts"].get(layer_idx)

            # Feature clamp / sham (skipped for mlp-replace rows)
            do_feat = (
                bun is not None and bun.feature_ids and acts is not None
                and plan.feature_mask.any()
                and layer_idx in plan.layers
            )
            if do_feat:
                tr = plan.target_row
                tgt = acts[tr.clamp(min=0)]
                coeff = (tgt - acts).float()  # [B, T, F]
                active_rows = torch.ones(B, device=output.device, dtype=torch.float32)
                if plan.row_layers is not None:
                    for b in range(B):
                        rl = plan.row_layers[b] if b < len(plan.row_layers) else None
                        allowed = plan.layers if rl is None else rl
                        if layer_idx not in allowed:
                            active_rows[b] = 0.0
                if plan.feature_id_sets is not None and bun.feature_ids:
                    F = len(bun.feature_ids)
                    fid_to_col = {int(f): j for j, f in enumerate(bun.feature_ids)}
                    feat_mask = torch.ones(B, F, device=output.device, dtype=torch.float32)
                    for b in range(B):
                        fids = plan.feature_id_sets[b] if b < len(plan.feature_id_sets) else None
                        if fids is not None:
                            feat_mask[b] = 0.0
                            for f in fids:
                                j = fid_to_col.get(int(f))
                                if j is not None:
                                    feat_mask[b, j] = 1.0
                    coeff = coeff * feat_mask.view(B, 1, F)
                mask = plan.feature_mask.to(device=output.device) & (~plan.mlp_mask.to(device=output.device))
                last_only = plan.last_only.to(device=output.device)
                coeff = coeff * mask.view(B, 1, 1).float() * active_rows.view(B, 1, 1)
                if last_only.any():
                    pos_scale = torch.ones(B, T, 1, device=output.device, dtype=torch.float32)
                    for b in range(B):
                        if last_only[b] and mask[b]:
                            pos_scale[b, :-1] = 0.0
                    coeff = coeff * pos_scale
                delta = coeff @ bun.w_dec.float()
                out = out + delta.to(dtype=out.dtype)

            # clampMLP: replace with unmodified post of target row
            if plan.mlp_mask.any() and layer_idx in plan.layers:
                tr = plan.target_row
                raw = state[raw_key]
                replaced = out.clone()
                mm = plan.mlp_mask.to(device=output.device)
                for b in range(B):
                    if mm[b]:
                        src = int(tr[b].item())
                        if src >= 0:
                            replaced[b] = raw[src].to(dtype=out.dtype)
                out = replaced

            state["post"][layer_idx] = out
            if capture_post_out is not None:
                capture_post_out[layer_idx] = out.detach().clone()
            return out

        return hook

    try:
        for li in sorted(set(plan.layers) | set(bundles.keys())):
            layer = layers_mod[li]
            handles.append(layer.pre_feedforward_layernorm.register_forward_hook(make_pre_hook(li)))
            handles.append(layer.post_feedforward_layernorm.register_forward_hook(make_post_hook(li)))
        yield state
    finally:
        for h in handles:
            h.remove()


def identity_delta_check(
    post_clamp: Tensor,
    post_steer: Tensor,
    acts_base: Tensor,
    acts_steer: Tensor,
    w_dec: Tensor,
    rtol: float = 1e-2,
) -> Tuple[bool, float]:
    """‖(post_clamp − post_steer) − Σ(a_base − a_steer)·w_dec‖ / ‖expected‖ ≤ rtol."""
    expected = delta_from_targets(acts_steer, acts_base, w_dec)
    got = (post_clamp.float() - post_steer.float())
    # align shapes [d] or [T,d] — use last position if needed
    if got.ndim == 2 and expected.ndim == 1:
        got = got[-1]
    if got.ndim == 2 and expected.ndim == 2:
        pass
    diff = got - expected
    denom = expected.norm().clamp_min(1e-12)
    rel = float(diff.norm() / denom)
    return rel <= rtol, rel
