"""GemmaScope-2 jump-ReLU transcoder loading and encoding (encode-side focus).

Independent reimplementation. A transcoder maps the MLP input (here the output of
``pre_feedforward_layernorm``) to sparse features:

    pre  = x @ w_enc + b_enc
    acts = (pre > threshold) * relu(pre)          # JumpReLU

Safetensors keys: w_enc [d_in, width], b_enc [width], threshold [width],
w_dec [width, d_out], b_dec [d_out], affine_skip_connection [d_in, d_out].
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

EXPECTED_WIDTH = 262144
EXPECTED_D_IN = 5376


@dataclass
class TranscoderConfig:
    layer: int
    d_in: int
    width: int
    affine_connection: bool
    hf_hook_point_in: str
    raw: dict = field(default_factory=dict)


def _first(raw: dict, keys: Sequence[str], default: Any = None) -> Any:
    for k in keys:
        if k in raw and raw[k] is not None:
            return raw[k]
    return default


def load_transcoder_config(folder: Path) -> TranscoderConfig:
    """Read ``config.json`` in ``folder`` and validate it.

    Asserts width == 262144, affine_connection is True, folder name ends with
    ``_affine`` and ``hf_hook_point_in == model.layers.{layer}.pre_feedforward_layernorm.output``.
    """
    folder = Path(folder)
    with open(folder / "config.json") as f:
        raw = json.load(f)

    # GemmaScope config.json has no layer key — parse from folder name.
    m = re.search(r"layer_(\d+)", folder.name)
    assert m, f"cannot determine layer from folder name {folder.name!r}"
    layer = int(m.group(1))

    if "width" not in raw:
        raise RuntimeError(f"config.json missing 'width'; keys={sorted(raw)}")
    if "affine_connection" not in raw:
        raise RuntimeError(f"config.json missing 'affine_connection'; keys={sorted(raw)}")
    if "hf_hook_point_in" not in raw:
        raise RuntimeError(f"config.json missing 'hf_hook_point_in'; keys={sorted(raw)}")

    width = int(raw["width"])
    affine = bool(raw["affine_connection"])
    hook_in = raw["hf_hook_point_in"]
    d_in = EXPECTED_D_IN

    assert width == EXPECTED_WIDTH, f"width={width}, expected {EXPECTED_WIDTH}"
    assert affine is True, "affine_connection must be True"
    assert folder.name.endswith("_affine"), f"folder {folder.name!r} must end with _affine"
    expected_hook = f"model.layers.{layer}.pre_feedforward_layernorm.output"
    assert hook_in == expected_hook, f"hf_hook_point_in={hook_in!r}, expected {expected_hook!r}"

    return TranscoderConfig(
        layer=layer, d_in=d_in, width=width, affine_connection=affine,
        hf_hook_point_in=hook_in, raw=raw,
    )


@dataclass
class TranscoderWeights:
    """Encode weights (bf16, full) plus fp32 copies of selected gate columns."""

    layer: int
    w_enc: Tensor            # [d_in, width]
    b_enc: Tensor            # [width]
    threshold: Tensor        # [width]
    gate_ids: List[int]
    w_enc_gates: Tensor      # [d_in, n_gates] fp32
    b_enc_gates: Tensor      # [n_gates] fp32
    threshold_gates: Tensor  # [n_gates] fp32
    w_dec: Optional[Tensor] = None
    b_dec: Optional[Tensor] = None
    affine_skip: Optional[Tensor] = None


def load_encode_weights(
    folder: Path,
    gate_ids: Sequence[int],
    device: Union[str, torch.device],
    load_decode: bool = False,
) -> TranscoderWeights:
    """Load encode weights from ``folder/params.safetensors`` via safetensors.safe_open.

    Full w_enc/b_enc/threshold are cast to bf16 on ``device``; gate columns are
    sliced from the stored (pre-cast) tensors and kept in fp32.
    """
    from safetensors import safe_open

    folder = Path(folder)
    cfg = load_transcoder_config(folder)
    gate_ids = [int(g) for g in gate_ids]
    idx = torch.tensor(gate_ids, dtype=torch.long)

    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        w_enc = f.get_tensor("w_enc")
        b_enc = f.get_tensor("b_enc")
        thr = f.get_tensor("threshold")
        assert w_enc.shape == (cfg.d_in, cfg.width), f"w_enc shape {tuple(w_enc.shape)}"

        w_g = w_enc.index_select(1, idx).float().to(device)
        b_g = b_enc.index_select(0, idx).float().to(device)
        t_g = thr.index_select(0, idx).float().to(device)

        out = TranscoderWeights(
            layer=cfg.layer,
            w_enc=w_enc.to(device=device, dtype=torch.bfloat16),
            b_enc=b_enc.to(device=device, dtype=torch.bfloat16),
            threshold=thr.to(device=device, dtype=torch.bfloat16),
            gate_ids=gate_ids,
            w_enc_gates=w_g,
            b_enc_gates=b_g,
            threshold_gates=t_g,
        )
        del w_enc
        if load_decode:
            out.w_dec = f.get_tensor("w_dec").to(device)
            out.b_dec = f.get_tensor("b_dec").to(device)
            if "affine_skip_connection" in keys:
                out.affine_skip = f.get_tensor("affine_skip_connection").to(device)
    return out


def _jumprelu(pre: Tensor, threshold: Tensor) -> Tensor:
    return torch.where(pre > threshold, torch.relu(pre), torch.zeros_like(pre))


@torch.no_grad()
def encode_full(x: Tensor, w: TranscoderWeights) -> Tensor:
    """x [..., d] -> acts [..., width] in w_enc dtype (JumpReLU)."""
    x = x.to(device=w.w_enc.device, dtype=w.w_enc.dtype)
    pre = x @ w.w_enc + w.b_enc
    return _jumprelu(pre, w.threshold.to(pre.dtype))


@torch.no_grad()
def encode_gates_fp32(x: Tensor, w: TranscoderWeights) -> Tuple[Tensor, Tensor]:
    """Return (pre_g, act_g), each [..., n_gates] fp32, using fp32 gate columns."""
    xf = x.to(device=w.w_enc_gates.device, dtype=torch.float32)
    pre_g = xf @ w.w_enc_gates + w.b_enc_gates
    act_g = _jumprelu(pre_g, w.threshold_gates)
    return pre_g, act_g


def sparse_from_acts(acts: Tensor) -> Tuple[Tensor, Tensor]:
    """1-D acts -> (indices int32, values float16) of nonzero features."""
    assert acts.dim() == 1, "acts must be 1-D"
    idx = torch.nonzero(acts, as_tuple=False).squeeze(-1)
    vals = acts[idx].to(torch.float16)
    return idx.to(torch.int32), vals


def dense_from_sparse(indices: Tensor, values: Tensor, width: int, dtype=torch.float32) -> Tensor:
    out = torch.zeros(width, dtype=dtype)
    out[indices.long()] = values.to(dtype)
    return out


def save_sparse_dict(path: Union[str, Path], d: Dict[str, Tuple[Tensor, Tensor]]) -> None:
    """Save {name: (indices, values)} to an .npz file."""
    arrs: Dict[str, np.ndarray] = {}
    for name, (idx, vals) in d.items():
        arrs[f"{name}__idx"] = idx.detach().cpu().numpy().astype(np.int32)
        arrs[f"{name}__val"] = vals.detach().cpu().numpy().astype(np.float16)
    path = str(path)
    with open(path, "wb") as f:
        np.savez_compressed(f, **arrs)


def load_sparse_dict(path: Union[str, Path]) -> Dict[str, Tuple[Tensor, Tensor]]:
    out: Dict[str, Tuple[Tensor, Tensor]] = {}
    with np.load(str(path)) as z:
        names = sorted({k[: -len("__idx")] for k in z.files if k.endswith("__idx")})
        for n in names:
            out[n] = (
                torch.from_numpy(z[f"{n}__idx"].copy()),
                torch.from_numpy(z[f"{n}__val"].copy()),
            )
    return out
