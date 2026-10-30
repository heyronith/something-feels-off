"""Damping / rank-matched / DLA helpers for WO-003a (CPU, no forwards)."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np


def proportional_k(a0: np.ndarray, da: np.ndarray) -> Tuple[float, float]:
    """k = -Σ a0·Δa / Σ a0² over baseline-active features; R² of Δa ≈ -k a0."""
    a0 = np.asarray(a0, dtype=np.float64)
    da = np.asarray(da, dtype=np.float64)
    ss = float(np.dot(a0, a0))
    if ss <= 0:
        return float("nan"), float("nan")
    k = float(-np.dot(a0, da) / ss)
    pred = -k * a0
    return k, r2_actual_pred(da, pred)


def affine_s(a0: np.ndarray, a1: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """s = Σ (a0−b)(a1−b) / Σ (a0−b)²; R² of (a1−b) ≈ s (a0−b)."""
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x = a0 - b
    y = a1 - b
    ss = float(np.dot(x, x))
    if ss <= 0:
        return float("nan"), float("nan")
    s = float(np.dot(x, y) / ss)
    return s, r2_actual_pred(y, s * x)


def r2_actual_pred(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    ss_res = float(np.sum((actual - pred) ** 2))
    if ss_tot <= 0:
        return float("nan") if ss_res > 0 else 1.0
    return 1.0 - ss_res / ss_tot


def slope_actual_on_pred(actual: np.ndarray, pred: np.ndarray) -> float:
    """OLS slope of actual ~ pred (no intercept): Σ pred·actual / Σ pred²."""
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    ss = float(np.dot(pred, pred))
    if ss <= 0:
        return float("nan")
    return float(np.dot(pred, actual) / ss)


def midrank_percentile(value: float, others: Sequence[float]) -> float:
    """Midrank percentile of ``value`` in ``others`` on a 0–100 scale.

    0 = most suppressed (most negative): fraction of others strictly greater
    than value, with ties getting midrank credit. Empty others → NaN.
    """
    arr = np.asarray(list(others), dtype=np.float64)
    if arr.size == 0 or not np.isfinite(value):
        return float("nan")
    # Rank ascending: smaller = more suppressed. Midrank among all including value.
    allv = np.concatenate([arr, np.asarray([value], dtype=np.float64)])
    # Average rank of ties (0-based), then map so most negative → 0.
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv), dtype=np.float64)
    i = 0
    n = len(allv)
    while i < n:
        j = i
        while j + 1 < n and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        mid = 0.5 * (i + j)
        for k in range(i, j + 1):
            ranks[order[k]] = mid
        i = j + 1
    # value is the last element
    r = ranks[-1]
    if n == 1:
        return 50.0
    return float(100.0 * r / (n - 1))


def matched_window(
    ranks_desc: Sequence[int],
    gate_pos: int,
    n: int = 20,
    exclude: Optional[Iterable[int]] = None,
) -> Tuple[List[int], int, int]:
    """Collect ``n`` non-excluded feature ids nearest in rank to the gate.

    ``ranks_desc`` is feature ids sorted by baseline activation descending
    (index 0 = rank 1). ``gate_pos`` is the 0-based index of the gate in that list.
    Returns (matched_ids, n_above, n_below) where above = higher activation
    (lower rank index) and below = lower activation.
    """
    excl = set(int(x) for x in (exclude or ()))
    N = len(ranks_desc)
    assert 0 <= gate_pos < N
    out: List[int] = []
    n_above = n_below = 0
    offset = 1
    while len(out) < n and (gate_pos - offset >= 0 or gate_pos + offset < N):
        for side, pos in (("above", gate_pos - offset), ("below", gate_pos + offset)):
            if len(out) >= n:
                break
            if pos < 0 or pos >= N:
                continue
            fid = int(ranks_desc[pos])
            if fid in excl:
                continue
            out.append(fid)
            if side == "above":
                n_above += 1
            else:
                n_below += 1
        offset += 1
    return out, n_above, n_below


def ratio_of_sums(numer: Sequence[float], denom: Sequence[float]) -> float:
    """Σ numer / Σ denom; NaN if denom sum is 0."""
    n = float(np.nansum(np.asarray(numer, dtype=np.float64)))
    d = float(np.nansum(np.asarray(denom, dtype=np.float64)))
    if d == 0.0:
        return float("nan")
    return n / d


def n_at_signed_frac(
    contribs: Sequence[float],
    total: float,
    frac: float,
) -> Optional[int]:
    """Smallest n after sorting |c| desc such that cumulative signed sum / total ≥ frac.

    Returns None ("not reached") if never met or total == 0.
    """
    c = np.asarray(contribs, dtype=np.float64)
    if c.size == 0 or not np.isfinite(total) or total == 0.0:
        return None
    order = np.argsort(-np.abs(c), kind="mergesort")
    cum = 0.0
    thr = frac * total
    # Compare signed cumulative against frac * total (same sign as total).
    for i, j in enumerate(order, start=1):
        cum += float(c[j])
        if total > 0 and cum >= thr:
            return i
        if total < 0 and cum <= thr:
            return i
    return None


def rho_from_acts(a0: float, a1: float) -> float:
    if not (np.isfinite(a0) and a0 > 0 and np.isfinite(a1)):
        return float("nan")
    return float(a1 / a0 - 1.0)


def rho_prime(a0: float, a1: float, b: float) -> float:
    if not (np.isfinite(a0) and np.isfinite(a1) and np.isfinite(b)):
        return float("nan")
    d0 = a0 - b
    if d0 == 0.0:
        return float("nan")
    return float((a1 - b) / d0 - 1.0)


def zscore(value: float, sample: Sequence[float]) -> float:
    arr = np.asarray(list(sample), dtype=np.float64)
    if arr.size < 2 or not np.isfinite(value):
        return float("nan")
    sd = float(arr.std(ddof=1))
    if sd == 0.0:
        return float("nan")
    return float((value - arr.mean()) / sd)
