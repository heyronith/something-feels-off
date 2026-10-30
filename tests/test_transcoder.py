"""CPU tests for sfo.transcoder with tiny random tensors (no files)."""

from __future__ import annotations

import json

import pytest
import torch

from sfo.transcoder import (
    TranscoderWeights,
    dense_from_sparse,
    encode_full,
    encode_gates_fp32,
    load_sparse_dict,
    load_transcoder_config,
    save_sparse_dict,
    sparse_from_acts,
)

D, W = 16, 64


def _make_weights(dtype=torch.float32, gate_ids=(3, 17, 40, 63), seed=0) -> TranscoderWeights:
    g = torch.Generator().manual_seed(seed)
    w_enc = torch.randn(D, W, generator=g)
    b_enc = torch.randn(W, generator=g) * 0.1
    thr = torch.rand(W, generator=g) * 0.5
    idx = torch.tensor(gate_ids)
    return TranscoderWeights(
        layer=0,
        w_enc=w_enc.to(dtype),
        b_enc=b_enc.to(dtype),
        threshold=thr.to(dtype),
        gate_ids=list(gate_ids),
        w_enc_gates=w_enc[:, idx].float(),
        b_enc_gates=b_enc[idx].float(),
        threshold_gates=thr[idx].float(),
    )


def test_encode_matches_manual_jumprelu():
    w = _make_weights()
    x = torch.randn(2, 5, D)
    acts = encode_full(x, w)
    assert acts.shape == (2, 5, W)
    for b in range(2):
        for t in range(5):
            for f in range(W):
                pre = float(x[b, t] @ w.w_enc[:, f] + w.b_enc[f])
                exp = pre if pre > float(w.threshold[f]) else 0.0
                assert acts[b, t, f].item() == pytest.approx(exp, abs=1e-5)
    # jump semantics: nonzero entries are strictly above threshold and positive
    pre = x @ w.w_enc + w.b_enc
    assert ((acts != 0) == ((pre > w.threshold) & (pre > 0))).all()


def test_encode_bf16_runs_and_close_to_fp32():
    w32 = _make_weights()
    wbf = _make_weights(dtype=torch.bfloat16)
    x = torch.randn(3, D)
    a = encode_full(x, wbf)
    assert a.dtype == torch.bfloat16
    ref = encode_full(x, w32)
    big = ref.abs() > 1.0  # avoid threshold-boundary flips
    assert torch.allclose(a.float()[big], ref[big], rtol=0.05, atol=0.1)


def test_gates_fp32_match_full_encode():
    w = _make_weights()
    x = torch.randn(4, 3, D)
    full = encode_full(x, w)
    pre_g, act_g = encode_gates_fp32(x, w)
    assert pre_g.dtype == torch.float32 and act_g.dtype == torch.float32
    assert act_g.shape == (4, 3, len(w.gate_ids))
    ids = torch.tensor(w.gate_ids)
    assert torch.allclose(act_g, full[..., ids], atol=1e-5, rtol=1e-5)
    pre_full = x @ w.w_enc + w.b_enc
    assert torch.allclose(pre_g, pre_full[..., ids], atol=1e-5, rtol=1e-5)


def test_sparse_roundtrip(tmp_path):
    w = _make_weights()
    x = torch.randn(3, D)
    acts = encode_full(x, w)
    d = {f"tok{i}": sparse_from_acts(acts[i]) for i in range(3)}
    for idx, vals in d.values():
        assert idx.dtype == torch.int32 and vals.dtype == torch.float16
    p = tmp_path / "sp.npz"
    save_sparse_dict(p, d)
    back = load_sparse_dict(p)
    assert set(back) == set(d)
    for k in d:
        assert torch.equal(back[k][0], d[k][0])
        assert torch.equal(back[k][1], d[k][1])
        dense = dense_from_sparse(*back[k], width=W)
        assert torch.allclose(dense, acts[int(k[3:])].to(torch.float16).float())


def test_sparse_empty(tmp_path):
    idx, vals = sparse_from_acts(torch.zeros(W))
    assert idx.numel() == 0 and vals.numel() == 0
    save_sparse_dict(tmp_path / "e.npz", {"a": (idx, vals)})
    b = load_sparse_dict(tmp_path / "e.npz")
    assert b["a"][0].numel() == 0


def _write_cfg(root, layer=45, **over):
    folder = root / f"layer_{layer}_width_262k_l0_big_affine"
    folder.mkdir(parents=True, exist_ok=True)
    cfg = {
        "layer": layer, "width": 262144, "d_in": 5376, "affine_connection": True,
        "hf_hook_point_in": f"model.layers.{layer}.pre_feedforward_layernorm.output",
    }
    cfg.update(over)
    (folder / "config.json").write_text(json.dumps(cfg))
    return folder


def test_config_validation(tmp_path):
    c = load_transcoder_config(_write_cfg(tmp_path))
    assert c.layer == 45 and c.width == 262144 and c.d_in == 5376
    with pytest.raises(AssertionError):
        load_transcoder_config(_write_cfg(tmp_path / "a", width=16384))
    with pytest.raises(AssertionError):
        load_transcoder_config(_write_cfg(tmp_path / "b", hf_hook_point_in="model.layers.45.post_feedforward_layernorm.output"))
    with pytest.raises(AssertionError):
        load_transcoder_config(_write_cfg(tmp_path / "c", affine_connection=False))
