"""CPU unit tests for sfo.damping (WO-003a formulas)."""

from __future__ import annotations

import numpy as np
import pytest

from sfo.damping import (
    affine_s,
    matched_window,
    midrank_percentile,
    n_at_signed_frac,
    proportional_k,
    ratio_of_sums,
)


def test_proportional_k_exact():
    a0 = np.array([2.0, 4.0, 6.0])
    k_true = 0.5
    da = -k_true * a0
    k, r2 = proportional_k(a0, da)
    assert k == pytest.approx(0.5)
    assert r2 == pytest.approx(1.0)


def test_affine_s_exact():
    b = np.array([1.0, 1.0, 1.0])
    a0 = np.array([3.0, 5.0, 7.0])
    s_true = 0.25
    a1 = b + s_true * (a0 - b)
    s, r2 = affine_s(a0, a1, b)
    assert s == pytest.approx(0.25)
    assert r2 == pytest.approx(1.0)


def test_midrank_percentile_most_suppressed():
    # value most negative → near 0
    p = midrank_percentile(-5.0, [-1.0, 0.0, 1.0, 2.0])
    assert p == pytest.approx(0.0)
    # most positive → 100
    p2 = midrank_percentile(5.0, [-1.0, 0.0, 1.0, 2.0])
    assert p2 == pytest.approx(100.0)
    # tie midrank
    p3 = midrank_percentile(0.0, [-1.0, 0.0, 0.0, 1.0])
    assert 0.0 < p3 < 100.0


def test_matched_window_at_edge_rg1():
    # ranks_desc: feature ids by descending activation; gate at position 0 (rank 1)
    ids = list(range(100, 150))  # 50 features
    gate_pos = 0
    m, above, below = matched_window(ids, gate_pos, n=20, exclude={100})
    # gate id is 100; excluded. Neighbors are only below.
    assert len(m) == 20
    assert above == 0
    assert below == 20
    assert 100 not in m
    assert m == list(range(101, 121))


def test_matched_window_symmetric_and_exclude():
    ids = list(range(0, 41))
    gate_pos = 20  # id 20
    m, above, below = matched_window(ids, gate_pos, n=20, exclude={20, 19, 21})
    assert len(m) == 20
    assert 20 not in m and 19 not in m and 21 not in m
    assert above + below == 20


def test_ratio_of_sums():
    assert ratio_of_sums([1.0, 2.0], [3.0, 1.0]) == pytest.approx(0.75)
    assert np.isnan(ratio_of_sums([1.0], [0.0]))


def test_n50_not_reached():
    # contributions never accumulate to 50% of total
    c = [0.1, 0.1, 0.1]
    total = 10.0
    assert n_at_signed_frac(c, total, 0.5) is None
    # reaches
    c2 = [3.0, 2.0, 1.0]
    assert n_at_signed_frac(c2, 6.0, 0.5) == 1  # 3/6 >= 0.5
    assert n_at_signed_frac(c2, 6.0, 0.8) == 2  # 5/6 >= 0.8
