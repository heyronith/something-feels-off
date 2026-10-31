"""WO-004 metrics.json + figures (numbers only; no interpretation).

Reads ``<results_dir>/raw/rows.jsonl``, ``controls.jsonl``, ``decomposition.jsonl``,
and ``<results_dir>/run_summary.json`` written by ``scripts/wo_004_run.py``.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]

SET_ORDER = ["S1", "S2v", "S3", "S4", "S8"]
LOPO_SETS = ["S1", "S3", "S8"]
FIG_SETS = ["S1", "S3", "S8", "S4"]
CLASSES = ["concept", "random"]
CONCEPT_CONDS = [
    "steer",
    "clampG",
    "clampG_last",
    "clampG45",
    "clampMdla",
    "clampMnorm",
    "clampMLP",
    "shamG",
]
RANDOM_CONDS = ["steer", "clampG", "clampMLP"]
NECESSITY_X = ["G", "G_last", "G45", "Mdla", "Mnorm", "MLP"]
COND_FOR_X = {
    "G": "clampG",
    "G_last": "clampG_last",
    "G45": "clampG45",
    "Mdla": "clampMdla",
    "Mnorm": "clampMnorm",
    "MLP": "clampMLP",
}
N_BOOT = 2000
BOOT_SEED = 0
CI95 = (2.5, 97.5)
DECOMP_SETS = ["S1", "S3", "S4", "S8"]
TAP_LAYERS = [45, 50]


def clean(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.generic):
        return clean(x.item())
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    return x


def read_jsonl(path: Path) -> List[dict]:
    out: List[dict] = []
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fl(x: Any) -> float:
    return float("nan") if x is None else float(x)


def nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = np.isfinite(a)
    return float(a[m].mean()) if m.any() else float("nan")


def pct(a: np.ndarray, q: Sequence[float]) -> List[float]:
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return [float("nan")] * len(q)
    return [float(v) for v in np.percentile(a, list(q))]


def ci_block(point: float, boots: Optional[np.ndarray]) -> Dict[str, Any]:
    if boots is None:
        return {"m": point, "lo95": None, "hi95": None}
    lo95, hi95 = pct(boots, CI95)
    return {"m": point, "lo95": lo95, "hi95": hi95}


def row_kind(r: dict) -> str:
    k = r.get("row_kind") or r.get("class") or ""
    if k in ("base", "baseline"):
        return "baseline"
    return str(k)


def vec_key(r: dict) -> Optional[str]:
    if r.get("concept") is not None:
        return str(r["concept"])
    if r.get("seed") is not None:
        return f"s{int(r['seed'])}"
    return None


def gate_acts(r: dict, layer: int) -> Optional[List[float]]:
    for key in (f"gate_act_L{layer}", f"gate_act", f"L{layer}"):
        v = r.get(key)
        if v is None:
            continue
        if isinstance(v, dict):
            v = v.get(str(layer), v.get(f"L{layer}"))
        if isinstance(v, list):
            return [fl(x) for x in v]
    return None


def _counts(rng: np.random.Generator, n_boot: int, n: int) -> np.ndarray:
    idx = rng.integers(0, n, size=(n_boot, n))
    return (idx[:, :, None] == np.arange(n)[None, None, :]).sum(axis=1).astype(np.float64)


class Boot:
    """Two-way bootstrap: prompts and vectors resampled independently."""

    def __init__(self, n_boot: int, seed: int, set_p: Dict[str, int], class_v: Dict[str, int]):
        self.n_boot = n_boot
        self.Cp: Dict[str, np.ndarray] = {}
        self.Cv: Dict[str, np.ndarray] = {}
        for si, s in enumerate(SET_ORDER):
            if s not in set_p:
                continue
            rng = np.random.default_rng([seed, si, 1])
            self.Cp[s] = _counts(rng, n_boot, set_p[s])
        for ci, k in enumerate(CLASSES):
            if k in class_v:
                rng = np.random.default_rng([seed, 100 + ci, 2])
                self.Cv[k] = _counts(rng, n_boot, class_v[k])

    def means(self, M: np.ndarray, s: str, k: str) -> np.ndarray:
        fin = np.isfinite(M)
        M0 = np.where(fin, M, 0.0)
        W = fin.astype(np.float64)
        Cp, Cv = self.Cp[s], self.Cv[k]
        num = np.einsum("bp,pv,bv->b", Cp, M0, Cv)
        den = np.einsum("bp,pv,bv->b", Cp, W, Cv)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def prompt_means(self, r: np.ndarray, s: str) -> np.ndarray:
        fin = np.isfinite(r)
        Cp = self.Cp[s]
        num = np.einsum("bp,p->b", Cp, np.where(fin, r, 0.0))
        den = np.einsum("bp,p->b", Cp, fin.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)


# cell[set][class][cond][prompt_idx][vec_key] -> ld
CellMap = Dict[str, Dict[str, Dict[str, Dict[int, Dict[str, float]]]]]


def build_ld_maps(rows: List[dict]) -> Tuple[Dict[Tuple[str, int], float], CellMap]:
    """base_ld[(set, prompt_idx)]; cell[set][class][cond][prompt_idx][vec_key] -> ld."""
    base_ld: Dict[Tuple[str, int], float] = {}
    cell: CellMap = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    )
    for r in rows:
        s, p = r["set"], int(r["prompt_idx"])
        cond = r.get("condition", "")
        rk = row_kind(r)
        if rk == "baseline" or cond == "base":
            base_ld[(s, p)] = fl(r.get("ld"))
            continue
        if rk not in CLASSES:
            continue
        vk = vec_key(r)
        if vk is None:
            continue
        cell[s][rk][cond][p][vk] = fl(r.get("ld"))
    return base_ld, cell


def cell_get(cell: CellMap, s: str, k: str, cond: str, p: int, v: str) -> Optional[float]:
    """Lookup LD keyed by (set, class, cond, prompt_idx, vector_key)."""
    d = cell.get(s, {}).get(k, {}).get(cond, {}).get(p, {}).get(v)
    return None if d is None else float(d)


def matrix_for(
    base_ld: Dict[Tuple[str, int], float],
    cell: CellMap,
    s: str,
    k: str,
    cond: str,
    prompts: List[int],
    vectors: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (dLD, LD, p_LDneg indicator) arrays [P,V]."""
    P, V = len(prompts), len(vectors)
    dld = np.full((P, V), np.nan)
    ld = np.full((P, V), np.nan)
    pi = {p: i for i, p in enumerate(prompts)}
    vi = {v: j for j, v in enumerate(vectors)}
    for p in prompts:
        bl = base_ld.get((s, p), float("nan"))
        for v, j in vi.items():
            lv = cell_get(cell, s, k, cond, p, v)
            if lv is None:
                continue
            i = pi[p]
            ld[i, j] = lv
            if np.isfinite(bl):
                dld[i, j] = lv - bl
    return dld, ld, (ld < 0).astype(float)


def lopo_means(M: np.ndarray) -> Dict[str, float]:
    """Leave-one-prompt-out nanmeans of row means."""
    out: Dict[str, float] = {}
    P = M.shape[0]
    for i in range(P):
        rows = np.delete(M, i, axis=0)
        rm = np.nanmean(rows, axis=0)
        out[str(i)] = nanmean(rm)
    return out


def paired_diff_matrix(
    base_ld: Dict[Tuple[str, int], float],
    cell: CellMap,
    s: str,
    k: str,
    cond_a: str,
    cond_b: str,
    prompts: List[int],
    vectors: List[str],
) -> np.ndarray:
    """mean(ΔLD_a - ΔLD_b) cells; same as dLD_a - dLD_b when base shared."""
    da, _, _ = matrix_for(base_ld, cell, s, k, cond_a, prompts, vectors)
    db, _, _ = matrix_for(base_ld, cell, s, k, cond_b, prompts, vectors)
    return da - db


def groupby_mean_ld_dld(rows: List[dict]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Independent plain group-by: mean LD and mean dLD per (set, class, cond)."""
    base: Dict[Tuple[str, int], float] = {}
    for r in rows:
        if row_kind(r) == "baseline" or r.get("condition") == "base":
            base[(r["set"], int(r["prompt_idx"]))] = fl(r.get("ld"))
    buckets: Dict[Tuple[str, str, str], List[Tuple[float, float]]] = defaultdict(list)
    for r in rows:
        rk = row_kind(r)
        if rk not in CLASSES:
            continue
        cond = r.get("condition", "")
        if not cond or cond == "base":
            continue
        s, p = r["set"], int(r["prompt_idx"])
        ld = fl(r.get("ld"))
        bl = base.get((s, p), float("nan"))
        dld = ld - bl if np.isfinite(bl) else float("nan")
        buckets[(s, rk, cond)].append((ld, dld))
    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, pairs in buckets.items():
        lds = np.array([a for a, _ in pairs], dtype=float)
        dlds = np.array([b for _, b in pairs], dtype=float)
        out[key] = {
            "n": len(pairs),
            "mean_LD": nanmean(lds),
            "mean_dLD": nanmean(dlds),
        }
    return out


def bootstrap_ratio(num_boot: np.ndarray, den_boot: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den_boot) > 1e-12, num_boot / den_boot, np.nan)


def ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    x, y = x[m], y[m]
    xm, ym = x.mean(), y.mean()
    den = np.sum((x - xm) ** 2)
    if den <= 0:
        return float("nan"), float("nan")
    slope = float(np.sum((x - xm) * (y - ym)) / den)
    intercept = float(ym - slope * xm)
    return slope, intercept


def load_dla_cache(raw: Path) -> Optional[Dict[str, Any]]:
    p = raw / "dla_cache.pt"
    if not p.exists():
        return None
    try:
        import torch

        return torch.load(p, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None


def _feat_act_map(r: dict, layer: int) -> Optional[Dict[int, float]]:
    """Prefer feat_act[Lℓ] dict; else map gate_act list via order of gates (caller)."""
    fa = r.get("feat_act")
    if isinstance(fa, dict):
        blk = fa.get(f"L{layer}") or fa.get(str(layer))
        if isinstance(blk, dict):
            return {int(k): fl(v) for k, v in blk.items()}
    return None


def linear_n_features(
    rows: List[dict],
    feature_ids_by_prompt_layer: Dict[Tuple[int, int], List[int]],
    dla_cache: Dict[str, Any],
    s: str,
    prompts: List[int],
    vectors: List[str],
    *,
    fallback_gates: Optional[Dict[int, List[int]]] = None,
) -> Tuple[np.ndarray, str]:
    """N_X^lin = Σ_{ℓ,f} (a_base,f − a_steer,f)·dla_normed[f] / rms_final(steer)."""
    layers = dla_cache.get("layers", {})
    P, V = len(prompts), len(vectors)
    out = np.full((P, V), np.nan)
    pi = {p: i for i, p in enumerate(prompts)}
    vi = {v: j for j, v in enumerate(vectors)}

    base_rows: Dict[int, dict] = {}
    steer_rows: Dict[Tuple[int, str], dict] = {}
    for r in rows:
        if r["set"] != s:
            continue
        p = int(r["prompt_idx"])
        if row_kind(r) == "baseline":
            base_rows[p] = r
        elif row_kind(r) == "concept" and r.get("condition") == "steer":
            vk = vec_key(r)
            if vk in vi:
                steer_rows[(p, vk)] = r

    note_parts: List[str] = []
    for (p, vk), rs in steer_rows.items():
        rb = base_rows.get(p)
        if rb is None:
            continue
        rms = fl(rs.get("rms_final"))
        if not np.isfinite(rms) or rms <= 0:
            continue
        lin = 0.0
        ok = True
        for layer in TAP_LAYERS:
            dla = layers.get(layer, layers.get(str(layer)))
            if dla is None:
                ok = False
                break
            dn = dla["dla_normed"]
            fids = feature_ids_by_prompt_layer.get((p, layer))
            if not fids and fallback_gates is not None:
                fids = fallback_gates.get(layer, [])
            if not fids:
                ok = False
                break
            ab_map = _feat_act_map(rb, layer)
            as_map = _feat_act_map(rs, layer)
            if ab_map is None or as_map is None:
                # Fall back to gate_act lists when feat_act missing (gates only).
                if fallback_gates is not None and fids == list(fallback_gates.get(layer, [])):
                    ab = gate_acts(rb, layer)
                    ast = gate_acts(rs, layer)
                    if ab is None or ast is None or len(ab) != len(fids) or len(ast) != len(fids):
                        ok = False
                        break
                    for gi, gid in enumerate(fids):
                        lin += (ab[gi] - ast[gi]) * float(dn[int(gid)])
                    continue
                ok = False
                break
            for fid in fids:
                if fid not in ab_map or fid not in as_map:
                    ok = False
                    break
                lin += (ab_map[fid] - as_map[fid]) * float(dn[int(fid)])
            if not ok:
                break
        if not ok:
            continue
        out[pi[p], vi[vk]] = lin / rms
    if not np.isfinite(out).any():
        note_parts.append("no paired base/steer feature acts for linear N")
    return out, "; ".join(note_parts) if note_parts else "ok"


def linear_n_gate(
    rows: List[dict],
    base_ld: Dict[Tuple[str, int], float],
    gates: Dict[int, List[int]],
    dla_cache: Dict[str, Any],
    s: str,
    prompts: List[int],
    vectors: List[str],
) -> Tuple[np.ndarray, str]:
    """N_G^lin using gate feature ids."""
    fids = {(p, layer): list(gates.get(layer, [])) for p in prompts for layer in TAP_LAYERS}
    return linear_n_features(
        rows, fids, dla_cache, s, prompts, vectors, fallback_gates=gates
    )


def hydra_l50_fraction(
    rows: List[dict],
    s: str,
    prompts: List[int],
    vectors: List[str],
    layer: int = 50,
) -> np.ndarray:
    """Per-cell fraction of L50 gate suppression restored under clampG45."""
    pi = {p: i for i, p in enumerate(prompts)}
    vi = {v: j for j, v in enumerate(vectors)}
    M = np.full((len(prompts), len(vectors)), np.nan)

    by: Dict[Tuple[int, str, str], dict] = {}
    for r in rows:
        if r["set"] != s or row_kind(r) != "concept":
            continue
        vk = vec_key(r)
        if vk is None:
            continue
        c = r.get("condition")
        if c not in ("steer", "clampG45"):
            continue
        by[(int(r["prompt_idx"]), vk, c)] = r

    for p in prompts:
        for vk in vectors:
            rb = None
            for r in rows:
                if (
                    r["set"] == s
                    and int(r["prompt_idx"]) == p
                    and row_kind(r) == "baseline"
                ):
                    rb = r
                    break
            rs = by.get((p, vk, "steer"))
            rc = by.get((p, vk, "clampG45"))
            if rb is None or rs is None or rc is None:
                continue
            ab = gate_acts(rb, layer)
            ast = gate_acts(rs, layer)
            ac = gate_acts(rc, layer)
            if ab is None or ast is None or ac is None:
                continue
            fracs = []
            for a0, a1, a2 in zip(ab, ast, ac):
                denom = a0 - a1
                if abs(denom) < 1e-9:
                    continue
                fracs.append((a2 - a1) / denom)
            if fracs:
                M[pi[p], vi[vk]] = float(np.mean(fracs))
    return M


def decomposition_metrics(raw: Path, prompts_by_set: Dict[str, List[int]]) -> Tuple[Dict[str, Any], str]:
    rows = read_jsonl(raw / "decomposition.jsonl")
    if not rows:
        return {}, "decomposition.jsonl missing or empty"
    parts = ("features_dla_eq", "skip_dla_eq", "error_dla_eq")
    # base keyed by (set, prompt_idx, layer); steer by (set, prompt_idx, layer, vec_key)
    base_acc: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    steer_acc: Dict[Tuple[str, int, int, str], Dict[str, float]] = {}
    for r in rows:
        s = r["set"]
        if s not in DECOMP_SETS:
            continue
        rk = row_kind(r)
        layer = int(r["layer"])
        p = int(r["prompt_idx"])
        entry = {
            "features_dla_eq": fl(r.get("features_dla_eq")),
            "skip_dla_eq": fl(r.get("skip_dla_eq")),
            "error_dla_eq": fl(r.get("error_dla_eq", r.get("proj_delta_u_n"))),
        }
        if rk == "baseline" or r.get("condition") == "base":
            base_acc[(s, p, layer)] = entry
        elif rk in CLASSES and r.get("condition") == "steer":
            vk = vec_key(r)
            if vk is None:
                continue
            steer_acc[(s, p, layer, vk)] = entry

    out: Dict[str, Any] = {}
    for s in DECOMP_SETS:
        if s not in prompts_by_set:
            continue
        for layer in TAP_LAYERS:
            deltas = {p: [] for p in parts}
            for (ss, pp, ll, _vk), ste in steer_acc.items():
                if ss != s or ll != layer:
                    continue
                bas = base_acc.get((ss, pp, ll))
                if bas is None:
                    continue
                for part in parts:
                    deltas[part].append(ste[part] - bas[part])
            out.setdefault(s, {})[f"L{layer}"] = {
                "features": ci_block(nanmean(np.array(deltas["features_dla_eq"])), None),
                "skip": ci_block(nanmean(np.array(deltas["skip_dla_eq"])), None),
                "error": ci_block(nanmean(np.array(deltas["error_dla_eq"])), None),
            }
    return out, "DLA-eq = part·Δu_n / rms_final; reported as steer−base mean change"


def controls_summary(ctrl_rows: List[dict]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for s in SET_ORDER:
        subset = [r for r in ctrl_rows if r.get("set") == s]
        if not subset:
            continue
        for layer in TAP_LAYERS:
            lk = f"L{layer}"
            k_dla, k_frac, k_norm = [], [], []
            for r in subset:
                blk = r.get(lk, {})
                if not blk:
                    md = r.get("M_dla", {}).get(str(layer), [])
                    mn = r.get("M_norm", {}).get(str(layer), [])
                    k_dla.append(len(md))
                    k_norm.append(len(mn))
                    continue
                k_dla.append(len(r.get("M_dla", {}).get(str(layer), [])))
                k_frac.append(fl(blk.get("K_frac_dla")))
                k_norm.append(len(r.get("M_norm", {}).get(str(layer), [])))
            out.setdefault(s, {})[lk] = {
                "K_dla_mean": nanmean(np.array(k_dla, dtype=float)),
                "frac_reached_mean": nanmean(np.array(k_frac, dtype=float))
                if k_frac
                else None,
                "K_norm_mean": nanmean(np.array(k_norm, dtype=float)),
            }
    return out


def make_figures(
    results_dir: Path, metrics: Dict[str, Any], ld_hist: Dict[str, Dict[str, List[dict]]]
) -> None:
    def nz(x: Any) -> float:
        return float("nan") if x is None else float(x)

    # ---- fig_recovery (S1/S3/S8 panels + S4 row)
    rec_sets_main = ["S1", "S3", "S8"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey="row")
    for ax, k in zip(axes[0], CLASSES):
        xs = np.arange(len(rec_sets_main))
        width = 0.11
        for xi, xname in enumerate(NECESSITY_X):
            off = (xi - len(NECESSITY_X) / 2) * width
            ys, ylo, yhi = [], [], []
            for s in rec_sets_main:
                rec = (
                    metrics.get("S", {})
                    .get(s, {})
                    .get(k, {})
                    .get("Rec", {})
                    .get(xname, {})
                )
                ys.append(nz(rec.get("m")))
                ylo.append(nz(rec.get("lo95")))
                yhi.append(nz(rec.get("hi95")))
            ys, ylo, yhi = np.array(ys), np.array(ylo), np.array(yhi)
            ax.errorbar(
                xs + off,
                ys,
                yerr=[np.clip(ys - ylo, 0, None), np.clip(yhi - ys, 0, None)],
                fmt="o",
                ms=3,
                capsize=2,
                label=xname,
            )
        ax.axhline(0.3, color="gray", ls="--", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(rec_sets_main)
        ax.set_title(f"Rec_X ({k})")
        ax.set_ylabel("Rec_X = N_X / N_MLP")
    for ax, k in zip(axes[1], CLASSES):
        ys = [
            nz(
                metrics.get("S", {})
                .get("S4", {})
                .get(k, {})
                .get("Rec", {})
                .get(xname, {})
                .get("m")
            )
            for xname in NECESSITY_X
        ]
        ax.plot(NECESSITY_X, ys, "o-", label=k)
        ax.axhline(0.3, color="gray", ls="--", lw=0.8)
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)
        ax.set_title("S4 Rec_X")
    axes[0][0].legend(fontsize=6, ncol=3)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_recovery.png", dpi=150)
    plt.close(fig)

    # ---- fig_efficiency
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = np.arange(len(FIG_SETS))
    for label, key, col, off in (
        ("Eff_G", "G", "C0", -0.15),
        ("Eff_Mdla", "Mdla", "C1", 0.15),
    ):
        ys, ylo, yhi = [], [], []
        for s in FIG_SETS:
            eff = metrics.get("S", {}).get(s, {}).get("concept", {}).get("Eff", {}).get(key, {})
            ys.append(nz(eff.get("m")))
            ylo.append(nz(eff.get("lo95")))
            yhi.append(nz(eff.get("hi95")))
        ys, ylo, yhi = np.array(ys), np.array(ylo), np.array(yhi)
        ax.errorbar(
            xs + off,
            ys,
            yerr=[np.clip(ys - ylo, 0, None), np.clip(yhi - ys, 0, None)],
            fmt="s",
            color=col,
            capsize=3,
            label=label,
        )
    ax.axhspan(0.6, 1.4, color="gray", alpha=0.15)
    ax.set_xticks(xs)
    ax.set_xticklabels(FIG_SETS)
    ax.set_ylabel("Efficiency N / N_lin")
    ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "fig_efficiency.png", dpi=150)
    plt.close(fig)

    # ---- fig_compression (from raw rows — passed via metrics compression block)
    comp = metrics.get("compression", {})
    pts = comp.get("scatter_points", [])
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = {"S1": "C0", "S2v": "C1", "S3": "C2", "S4": "C3", "S8": "C4"}
    for pt in pts:
        ax.scatter(pt["ld_base"], pt["ld_steer"], s=8, alpha=0.5, c=colors.get(pt["set"], "k"))
    lims = comp.get("axis_limits")
    if lims:
        ax.plot([lims[0], lims[1]], [lims[0], lims[1]], "k--", lw=0.7, label="identity")
    for k, fit in (comp.get("fit") or {}).items():
        if fit.get("slope") is None or not np.isfinite(fit.get("slope")):
            continue
        x0, x1 = ax.get_xlim()
        xs = np.array([x0, x1])
        ax.plot(xs, fit["slope"] * xs + fit["intercept"], lw=1.2, label=f"S4 fit ({k})")
    ax.set_xlabel("LD_base")
    ax.set_ylabel("LD_steer")
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_compression.png", dpi=150)
    plt.close(fig)

    # ---- fig_s8
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, s in zip(axes, ["S8", "S1"]):
        for cond, col in (("base", "C0"), ("steer", "C3")):
            vals = [pt["ld"] for pt in ld_hist.get(s, {}).get(cond, [])]
            if vals:
                ax.hist(vals, bins=20, alpha=0.55, color=col, label=cond)
        ax.set_title(s)
        ax.set_xlabel("LD")
        ax.legend()
    fig.tight_layout()
    fig.savefig(results_dir / "fig_s8.png", dpi=150)
    plt.close(fig)


def main(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    cfg_path = REPO_ROOT / "configs" / "wo_004.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    gates = {int(k): [int(g) for g in v] for k, v in cfg.get("gates", {45: [9959, 74631], 50: [167]}).items()}

    summary: Dict[str, Any] = {}
    if (results_dir / "run_summary.json").exists():
        summary = json.loads((results_dir / "run_summary.json").read_text())

    rows = read_jsonl(raw / "rows.jsonl")
    ctrl_rows = read_jsonl(raw / "controls.jsonl")
    base_ld, cell = build_ld_maps(rows)

    prompts_by_set: Dict[str, List[int]] = {}
    vectors_by_class: Dict[str, List[str]] = {k: [] for k in CLASSES}
    for r in rows:
        s, p = r["set"], int(r["prompt_idx"])
        prompts_by_set.setdefault(s, set()).add(p)  # type: ignore[arg-type]
        rk = row_kind(r)
        if rk in CLASSES:
            vk = vec_key(r)
            if vk:
                vectors_by_class[rk].append(vk)
    prompts_by_set = {s: sorted(list(ps)) for s, ps in prompts_by_set.items()}  # type: ignore[arg-type]
    for k in CLASSES:
        vectors_by_class[k] = sorted(set(vectors_by_class[k]))

    set_p = {s: len(prompts_by_set[s]) for s in prompts_by_set}
    class_v = {k: len(vectors_by_class[k]) for k in CLASSES if vectors_by_class[k]}
    boot = Boot(N_BOOT, BOOT_SEED, set_p, class_v)

    dla_cache = load_dla_cache(raw)
    eff_notes: List[str] = []
    if dla_cache is None:
        eff_notes.append("raw/dla_cache.pt missing")

    S_out: Dict[str, Any] = {}
    scatter_points: List[dict] = []
    ld_hist: Dict[str, Dict[str, List[dict]]] = defaultdict(lambda: defaultdict(list))

    for s in SET_ORDER:
        if s not in prompts_by_set:
            continue
        prompts = prompts_by_set[s]
        S_out.setdefault(s, {})
        for k in CLASSES:
            vecs = vectors_by_class.get(k, [])
            if not vecs:
                continue
            S_out[s][k] = {}
            conds = CONCEPT_CONDS if k == "concept" else RANDOM_CONDS
            for cond in conds:
                dld_m, ld_m, pneg_m = matrix_for(base_ld, cell, s, k, cond, prompts, vecs)
                ent = {
                    "dLD": ci_block(nanmean(dld_m), boot.means(dld_m, s, k)),
                    "LD": ci_block(nanmean(ld_m), boot.means(ld_m, s, k)),
                    "p_LDneg": ci_block(nanmean(pneg_m), boot.means(pneg_m, s, k)),
                }
                S_out[s][k][cond] = ent
            # Necessity / Rec (concept only for full X; random: G and MLP)
            x_list = NECESSITY_X if k == "concept" else ["G", "MLP"]
            N_blk: Dict[str, Any] = {}
            Rec_blk: Dict[str, Any] = {}
            for xname in x_list:
                ccond = COND_FOR_X[xname]
                diff = paired_diff_matrix(
                    base_ld, cell, s, k, ccond, "steer", prompts, vecs
                )
                n_point = nanmean(diff)
                n_boot = boot.means(diff, s, k)
                N_blk[xname] = ci_block(n_point, n_boot)
            if N_blk and "MLP" in N_blk:
                n_mlp_mat = paired_diff_matrix(
                    base_ld, cell, s, k, "clampMLP", "steer", prompts, vecs
                )
                n_mlp_boot = boot.means(n_mlp_mat, s, k)
                for xname in x_list:
                    n_boot = boot.means(
                        paired_diff_matrix(
                            base_ld, cell, s, k, COND_FOR_X[xname], "steer", prompts, vecs
                        ),
                        s,
                        k,
                    )
                    n_point = N_blk[xname]["m"]
                    rec_boot = bootstrap_ratio(n_boot, n_mlp_boot)
                    rec_point = (
                        n_point / N_blk["MLP"]["m"]
                        if N_blk.get("MLP", {}).get("m") not in (None, 0)
                        and np.isfinite(N_blk["MLP"]["m"])
                        else float("nan")
                    )
                    Rec_blk[xname] = ci_block(rec_point, rec_boot)
            if N_blk:
                S_out[s][k]["N"] = N_blk
            if Rec_blk:
                S_out[s][k]["Rec"] = Rec_blk

            d_steer, _, _ = matrix_for(base_ld, cell, s, k, "steer", prompts, vecs)
            neg_mean = -nanmean(d_steer)
            if N_blk:
                alt: Dict[str, Any] = {}
                for xname, nb in N_blk.items():
                    m = nb.get("m")
                    alt[xname] = (
                        m / neg_mean
                        if m is not None and np.isfinite(neg_mean) and neg_mean != 0
                        else None
                    )
                S_out[s][k]["N_over_neg_dLD_steer"] = alt

            if k == "concept":
                d_sham, _, _ = matrix_for(base_ld, cell, s, k, "shamG", prompts, vecs)
                suf = nanmean(d_sham) / nanmean(d_steer) if np.isfinite(nanmean(d_steer)) else float("nan")
                S_out[s][k]["Suf_G"] = suf
                rec_g = Rec_blk.get("G", {}).get("m")
                n_mlp = N_blk.get("MLP", {}).get("m")
                nonadd = float("nan")
                if (
                    suf is not None
                    and rec_g is not None
                    and n_mlp is not None
                    and np.isfinite(neg_mean)
                    and neg_mean != 0
                ):
                    nonadd = abs(suf - rec_g * n_mlp / neg_mean)
                S_out[s][k]["nonadd"] = nonadd

                hydra_m = hydra_l50_fraction(rows, s, prompts, vecs)
                S_out[s][k]["hydra_L50_restored"] = ci_block(
                    nanmean(hydra_m), boot.means(hydra_m, s, k)
                )

                # Efficiency
                Eff: Dict[str, Any] = {}
                if dla_cache is not None:
                    lin_g, note_g = linear_n_gate(
                        rows, base_ld, gates, dla_cache, s, prompts, vecs
                    )
                    if note_g != "ok":
                        eff_notes.append(f"{s}: Eff_G {note_g}")
                    n_g = paired_diff_matrix(
                        base_ld, cell, s, k, "clampG", "steer", prompts, vecs
                    )
                    ng_m = nanmean(n_g)
                    lg_m = nanmean(lin_g)
                    eg_boot = bootstrap_ratio(
                        boot.means(n_g, s, k), boot.means(lin_g, s, k)
                    )
                    Eff["G"] = ci_block(
                        ng_m / lg_m if np.isfinite(lg_m) and lg_m != 0 else float("nan"),
                        eg_boot,
                    )
                    mdla_fids: Dict[Tuple[int, int], List[int]] = {}
                    for cr in ctrl_rows:
                        if cr.get("set") != s:
                            continue
                        pi_ = int(cr["prompt_idx"])
                        for layer in TAP_LAYERS:
                            mdla_fids[(pi_, layer)] = [
                                int(f) for f in cr.get("M_dla", {}).get(str(layer), [])
                            ]
                    lin_m, note_m = linear_n_features(
                        rows, mdla_fids, dla_cache, s, prompts, vecs
                    )
                    if note_m != "ok":
                        eff_notes.append(f"{s}: Eff_Mdla {note_m}")
                    n_m = paired_diff_matrix(
                        base_ld, cell, s, k, "clampMdla", "steer", prompts, vecs
                    )
                    nm_m = nanmean(n_m)
                    lm_m = nanmean(lin_m)
                    em_boot = bootstrap_ratio(
                        boot.means(n_m, s, k), boot.means(lin_m, s, k)
                    )
                    Eff["Mdla"] = ci_block(
                        nm_m / lm_m if np.isfinite(lm_m) and lm_m != 0 else float("nan"),
                        em_boot,
                    )
                    if (
                        Eff.get("G", {}).get("m") is not None
                        and Eff["Mdla"].get("m") is not None
                        and np.isfinite(Eff["G"]["m"])
                        and np.isfinite(Eff["Mdla"]["m"])
                    ):
                        Eff["diff"] = ci_block(
                            Eff["G"]["m"] - Eff["Mdla"]["m"], eg_boot - em_boot
                        )
                    else:
                        Eff["diff"] = {"m": None, "lo95": None, "hi95": None}
                else:
                    Eff = {
                        "G": {"m": None, "lo95": None, "hi95": None},
                        "Mdla": {"m": None, "lo95": None, "hi95": None},
                        "diff": {"m": None, "lo95": None, "hi95": None},
                    }
                S_out[s][k]["Eff"] = Eff

            if s in LOPO_SETS:
                d_steer, _, _ = matrix_for(base_ld, cell, s, k, "steer", prompts, vecs)
                S_out[s][k]["lopo"] = {
                    "dLD_steer": lopo_means(d_steer),
                    "mean_dLD_steer": nanmean(np.nanmean(d_steer, axis=1)),
                }

            # scatter for compression figure (steer points)
            _, ld_steer_mat, _ = matrix_for(base_ld, cell, s, k, "steer", prompts, vecs)
            for i, p in enumerate(prompts):
                bl = base_ld.get((s, p))
                if bl is None:
                    continue
                for j in range(len(vecs)):
                    ls = ld_steer_mat[i, j]
                    if np.isfinite(ls):
                        scatter_points.append(
                            {"set": s, "class": k, "ld_base": bl, "ld_steer": ls}
                        )

    for s in ("S8", "S1"):
        if s not in prompts_by_set:
            continue
        for p in prompts_by_set[s]:
            bl = base_ld.get((s, p))
            if bl is not None and np.isfinite(bl):
                ld_hist[s]["base"].append({"ld": bl, "prompt_idx": p})
        for k in CLASSES:
            vecs = vectors_by_class.get(k, [])
            if not vecs:
                continue
            _, ld_steer, _ = matrix_for(
                base_ld, cell, s, k, "steer", prompts_by_set[s], vecs
            )
            for row in ld_steer:
                for v in row[np.isfinite(row)]:
                    ld_hist[s]["steer"].append({"ld": float(v)})

    # Margin compression
    compression: Dict[str, Any] = {"fit": {}, "pred_resid": {}, "ratio": {}, "scatter_points": scatter_points}
    for k in CLASSES:
        vecs = vectors_by_class.get(k, [])
        if not vecs or "S4" not in prompts_by_set:
            compression["fit"][k] = {"slope": None, "intercept": None, "ci": None}
            continue
        prompts_s4 = prompts_by_set["S4"]
        x_list, y_list = [], []
        for p in prompts_s4:
            bl = base_ld.get(("S4", p))
            if bl is None:
                continue
            for v in vecs:
                lv = cell_get(cell, "S4", k, "steer", p, v)
                if lv is not None:
                    x_list.append(bl)
                    y_list.append(lv)
        x_a, y_a = np.array(x_list, dtype=float), np.array(y_list, dtype=float)
        slope, intercept = ols_fit(x_a, y_a)
        # bootstrap over S4 prompts
        P4 = len(prompts_s4)
        boot_slopes = []
        rng = np.random.default_rng([BOOT_SEED, 44, 0])
        for _ in range(N_BOOT):
            idx = rng.integers(0, P4, size=P4)
            xp, yp = [], []
            for pi in idx:
                p = prompts_s4[pi]
                bl = base_ld.get(("S4", p))
                if bl is None:
                    continue
                for v in vecs:
                    lv = cell_get(cell, "S4", k, "steer", p, v)
                    if lv is not None:
                        xp.append(bl)
                        yp.append(lv)
            sl, _ = ols_fit(np.array(xp), np.array(yp))
            boot_slopes.append(sl)
        compression["fit"][k] = {
            "slope": slope,
            "intercept": intercept,
            "ci": ci_block(slope, np.array(boot_slopes)),
        }
        for s in LOPO_SETS:
            if s not in prompts_by_set:
                continue
            obs_vals = []
            pred_vals = []
            for p in prompts_by_set[s]:
                bl = base_ld.get((s, p))
                if bl is None or not np.isfinite(slope):
                    continue
                pred = slope * bl + intercept
                for v in vecs:
                    lv = cell_get(cell, s, k, "steer", p, v)
                    if lv is not None:
                        obs_vals.append(lv)
                        pred_vals.append(pred)
            obs_a = np.array(obs_vals)
            resid = obs_a - np.array(pred_vals) if obs_vals else np.array([])
            compression.setdefault("pred_resid", {}).setdefault(s, {})[k] = ci_block(
                nanmean(resid), None
            )
        compression.setdefault("ratio", {})
        for s in SET_ORDER:
            if s not in prompts_by_set:
                continue
            num, den = [], []
            for p in prompts_by_set[s]:
                bl = base_ld.get((s, p))
                if bl is None:
                    continue
                for v in vecs:
                    lv = cell_get(cell, s, k, "steer", p, v)
                    if lv is not None:
                        num.append(lv)
                        den.append(bl)
            compression["ratio"].setdefault(s, {})[k] = (
                nanmean(np.array(num)) / nanmean(np.array(den))
                if den and nanmean(np.array(den)) != 0
                else None
            )

    if scatter_points:
        all_ld = [pt["ld_base"] for pt in scatter_points] + [pt["ld_steer"] for pt in scatter_points]
        fin = [v for v in all_ld if np.isfinite(v)]
        if fin:
            lo, hi = min(fin), max(fin)
            pad = 0.05 * (hi - lo + 1e-9)
            compression["axis_limits"] = [lo - pad, hi + pad]

    decomp, decomp_note = decomposition_metrics(raw, prompts_by_set)
    ctrl_sum = controls_summary(ctrl_rows)

    n_rows = len(rows)
    assert n_rows == 39646, f"row count {n_rows} != 39646"

    # Real-data group-by check (independent of matrix_for)
    gb = groupby_mean_ld_dld(rows)
    gb_rows_csv: List[dict] = []
    for s in SET_ORDER:
        if s not in S_out:
            continue
        for k in CLASSES:
            if k not in S_out[s]:
                continue
            conds = CONCEPT_CONDS if k == "concept" else RANDOM_CONDS
            n_expected = set_p[s] * class_v[k]
            for cond in conds:
                key = (s, k, cond)
                g = gb.get(key)
                assert g is not None, f"missing group-by cell {key}"
                assert g["n"] == n_expected, (
                    f"{key}: n={g['n']} != prompts×vectors={n_expected}"
                )
                m_ld = S_out[s][k][cond]["LD"]["m"]
                m_dld = S_out[s][k][cond]["dLD"]["m"]
                assert abs(m_ld - g["mean_LD"]) < 1e-6, (
                    f"{key} LD metrics={m_ld} groupby={g['mean_LD']}"
                )
                assert abs(m_dld - g["mean_dLD"]) < 1e-6, (
                    f"{key} dLD metrics={m_dld} groupby={g['mean_dLD']}"
                )
                gb_rows_csv.append(
                    {
                        "set": s,
                        "class": k,
                        "cond": cond,
                        "n": g["n"],
                        "mean_LD": g["mean_LD"],
                        "mean_dLD": g["mean_dLD"],
                        "metrics_LD_m": m_ld,
                        "metrics_dLD_m": m_dld,
                    }
                )
    gb_path = raw / "groupby_check.csv"
    with gb_path.open("w") as f:
        f.write("set,class,cond,n,mean_LD,mean_dLD,metrics_LD_m,metrics_dLD_m\n")
        for r in gb_rows_csv:
            f.write(
                f"{r['set']},{r['class']},{r['cond']},{r['n']},"
                f"{r['mean_LD']},{r['mean_dLD']},{r['metrics_LD_m']},{r['metrics_dLD_m']}\n"
            )

    sanity = {
        "run_status": summary.get("status"),
        "run_errors": summary.get("errors"),
        "n_rows_logged": n_rows,
        "replication_max_abs_ld": summary.get("replication_max_abs_ld"),
        "projection": summary.get("projection"),
        "efficiency_notes": eff_notes,
        "decomposition_note": decomp_note,
        "prompts_per_set": set_p,
        "vectors_per_class": class_v,
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED, "ci95_pctl": CI95},
        "groupby_check": "PASS",
        "groupby_check_path": str(gb_path.name),
        "wo004a": True,
    }

    metrics: Dict[str, Any] = {
        "wo_id": "WO-004",
        "S": S_out,
        "compression": compression,
        "decomposition": decomp,
        "controls_summary": ctrl_sum,
        "sanity": sanity,
        "git_hash": summary.get("git_hash"),
        "model_revision": summary.get("model_revision"),
        "transcoder_revision": summary.get("transcoder_revision") or cfg.get("transcoder_revision"),
        "batch_size": summary.get("batch_size") or cfg.get("batch_size"),
    }
    (results_dir / "metrics.json").write_text(json.dumps(clean(metrics), indent=2))
    make_figures(results_dir, metrics, ld_hist)
    print(f"Wrote {results_dir / 'metrics.json'} and figures under {results_dir}")
    print(f"groupby_check PASS → {gb_path}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "WO-004")
