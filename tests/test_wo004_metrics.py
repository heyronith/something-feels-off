"""Regression: WO-004 metrics must key LD by prompt_idx (WO-004a)."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from wo_004_metrics import (  # noqa: E402
    build_ld_maps,
    matrix_for,
    nanmean,
    row_kind,
    vec_key,
)


def _synthetic_rows() -> List[dict]:
    """2 prompts × 2 concept vectors; LD differs by prompt (and by vector)."""
    rows: List[dict] = []
    # prompt 0 base=10; Dust steer=7 (dLD=-3); Sat steer=4 (dLD=-6)
    # prompt 1 base=20; Dust steer=18 (dLD=-2); Sat steer=11 (dLD=-9)
    for p, base in ((0, 10.0), (1, 20.0)):
        rows.append(
            {
                "set": "S1",
                "prompt_idx": p,
                "row_kind": "base",
                "condition": "base",
                "ld": base,
            }
        )
    for p, dust, sat in ((0, 7.0, 4.0), (1, 18.0, 11.0)):
        rows.append(
            {
                "set": "S1",
                "prompt_idx": p,
                "row_kind": "concept",
                "condition": "steer",
                "concept": "Dust",
                "ld": dust,
            }
        )
        rows.append(
            {
                "set": "S1",
                "prompt_idx": p,
                "row_kind": "concept",
                "condition": "steer",
                "concept": "Satellites",
                "ld": sat,
            }
        )
    return rows


def _plain_groupby_mean_dld(rows: List[dict], s: str, k: str, cond: str) -> float:
    base: Dict[Tuple[str, int], float] = {}
    for r in rows:
        if row_kind(r) == "baseline" or r.get("condition") == "base":
            base[(r["set"], int(r["prompt_idx"]))] = float(r["ld"])
    dlds = []
    for r in rows:
        if r["set"] != s or row_kind(r) != k or r.get("condition") != cond:
            continue
        bl = base[(s, int(r["prompt_idx"]))]
        dlds.append(float(r["ld"]) - bl)
    return float(np.mean(dlds))


def _buggy_build_ld_maps(rows: List[dict]):
    """Pre-WO-004a bug: cell[set][class][cond][vec_key] overwrites across prompts."""
    base_ld: Dict[Tuple[str, int], float] = {}
    cell: Dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for r in rows:
        s, p = r["set"], int(r["prompt_idx"])
        cond = r.get("condition", "")
        rk = row_kind(r)
        if rk == "baseline" or cond == "base":
            base_ld[(s, p)] = float(r["ld"])
            continue
        if rk not in ("concept", "random"):
            continue
        vk = vec_key(r)
        if vk is None:
            continue
        cell[s][rk][cond][vk] = float(r["ld"])
    return base_ld, cell


def _buggy_matrix_for(base_ld, cell, s, k, cond, prompts, vectors):
    P, V = len(prompts), len(vectors)
    dld = np.full((P, V), np.nan)
    ld = np.full((P, V), np.nan)
    pi = {p: i for i, p in enumerate(prompts)}
    vi = {v: j for j, v in enumerate(vectors)}
    for p in prompts:
        bl = base_ld.get((s, p), float("nan"))
        for v, j in vi.items():
            lv = cell.get(s, {}).get(k, {}).get(cond, {}).get(v)  # no prompt_idx
            if lv is None:
                continue
            i = pi[p]
            ld[i, j] = lv
            if np.isfinite(bl):
                dld[i, j] = lv - bl
    return dld, ld, (ld < 0).astype(float)


def test_buggy_keying_disagrees_with_groupby():
    rows = _synthetic_rows()
    truth = _plain_groupby_mean_dld(rows, "S1", "concept", "steer")
    # truth = mean([-3,-6,-2,-9]) = -5.0
    assert truth == pytest.approx(-5.0)
    base_ld, cell = _buggy_build_ld_maps(rows)
    dld, _, _ = _buggy_matrix_for(
        base_ld, cell, "S1", "concept", "steer", [0, 1], ["Dust", "Satellites"]
    )
    buggy_mean = nanmean(dld)
    # Last prompt overwrites: Dust=18, Sat=11 → dLD for p0: 8,-9? wait
    # base p0=10 → 18-10=8, 11-10=1; p1: 18-20=-2, 11-20=-9; mean=(8+1-2-9)/4=-0.5
    assert buggy_mean == pytest.approx(-0.5)
    assert buggy_mean != pytest.approx(truth)


def test_fixed_matrix_for_matches_groupby():
    rows = _synthetic_rows()
    truth = _plain_groupby_mean_dld(rows, "S1", "concept", "steer")
    base_ld, cell = build_ld_maps(rows)
    dld, _, _ = matrix_for(
        base_ld, cell, "S1", "concept", "steer", [0, 1], ["Dust", "Satellites"]
    )
    assert nanmean(dld) == pytest.approx(truth)
    assert nanmean(dld) == pytest.approx(-5.0)
    # Per-prompt values differ
    assert dld[0, 0] == pytest.approx(-3.0)
    assert dld[1, 0] == pytest.approx(-2.0)
