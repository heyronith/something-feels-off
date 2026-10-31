"""CPU tests for sfo.feature_clamp (tiny random weights)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sfo.feature_clamp import (
    delta_from_targets,
    encode_features_fp32,
    jumprelu,
    select_m_dla,
    select_m_norm,
    FeatureBundle,
)


def _bundle(d=8, f_ids=(1, 3, 5), seed=0) -> FeatureBundle:
    g = torch.Generator().manual_seed(seed)
    F = len(f_ids)
    w_enc = torch.randn(d, F, generator=g)
    w_dec = torch.randn(F, d, generator=g)
    b = torch.randn(F, generator=g) * 0.1
    thr = torch.zeros(F)
    return FeatureBundle(
        layer=0, feature_ids=list(f_ids), w_enc=w_enc, b_enc=b, threshold=thr,
        w_dec=w_dec, w_dec_norm=w_dec.norm(dim=1),
    )


def test_delta_formula():
    bun = _bundle()
    x = torch.randn(2, 4, 8)
    acts = encode_features_fp32(x, bun)
    # target = acts → delta 0
    d0 = delta_from_targets(acts, acts, bun.w_dec)
    assert torch.allclose(d0, torch.zeros_like(d0), atol=1e-6)
    tgt = acts.clone()
    tgt[0] = acts[1]
    d = delta_from_targets(acts, tgt, bun.w_dec)
    # row0 delta = (acts1 - acts0) @ w_dec
    expect = (acts[1] - acts[0]) @ bun.w_dec
    assert torch.allclose(d[0], expect, atol=1e-5)


def test_identity_when_target_equals_own():
    bun = _bundle()
    x = torch.randn(3, 5, 8)
    acts = encode_features_fp32(x, bun)
    d = delta_from_targets(acts, acts, bun.w_dec)
    assert float(d.norm()) == pytest.approx(0.0, abs=1e-6)


def test_cross_row_target_lookup():
    bun = _bundle()
    x = torch.randn(2, 3, 8)
    acts = encode_features_fp32(x, bun)
    # row1 targets row0's acts
    tr = torch.tensor([0, 0])
    tgt = acts[tr]
    d = delta_from_targets(acts, tgt, bun.w_dec)
    assert torch.allclose(d[0], torch.zeros_like(d[0]), atol=1e-6)
    expect = (acts[0] - acts[1]) @ bun.w_dec
    assert torch.allclose(d[1], expect, atol=1e-5)


def test_select_m_dla_greedy():
    a0 = {0: 10.0, 1: 5.0, 2: 4.0, 3: 3.0, 9: 8.0}  # 9 = gate
    dla = np.zeros(20)
    dla[9] = 2.0   # gate total = 16
    dla[0] = 1.0   # 10
    dla[1] = 1.0   # 5 → cum 15
    dla[2] = 1.0   # 4 → cum 19 ≥ 16
    dla[3] = 1.0
    ids, frac, total = select_m_dla(a0, dla, gate_ids=[9], k_max=50)
    assert total == pytest.approx(16.0)
    assert ids == [0, 1, 2]
    assert frac >= 1.0 - 1e-9


def test_select_m_dla_inactive_gate():
    a0 = {0: 1.0}
    dla = np.ones(5)
    ids, frac, total = select_m_dla(a0, dla, gate_ids=[4], k_max=50)
    assert ids == [] and total == 0.0


def test_select_m_norm():
    a0 = {0: 2.0, 1: 2.0, 2: 2.0, 9: 3.0}
    w_norm = np.ones(20)
    dla_n = np.zeros(20)
    dla_n[0] = 0.01
    dla_n[1] = 0.01
    dla_n[2] = 0.01
    dla_n[9] = 5.0
    rng = np.random.default_rng(0)
    ids, frac, total = select_m_norm(
        a0, w_norm, dla_n, gate_ids=[9], exclude=[], p50_abs_dla=1.0, rng=rng, k_max=50,
    )
    assert total == pytest.approx(3.0)
    assert len(ids) >= 1
    assert 9 not in ids
