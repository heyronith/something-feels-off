"""WO-003 metrics.json + figures (numbers only; no interpretation).

Reads ``<results_dir>/raw`` written by ``scripts/wo_003_run.py``:

    gate_scalars.jsonl      one row per (set, prompt_idx, cid):
                            {set, prompt_idx, cid, B, alpha, has_baseline,
                             "L45": {gate_ids, pre, act, thr, mean20_pre, mean20_act,
                                     dpre, direct_pre, cos_wv}, "L50": {...}}
    residual_scalars.jsonl  {set, prompt_idx, cid, B, "L45": {resid_layer, reach,
                             reach_indirect, dirproj}, "L50": {...}}
    logits_ld.jsonl         {set, prompt_idx, cid, ld, logit_yes, logit_no, top1_id}
                            (full-forward sets S1-S4 only; LD = logit(No) - logit(Yes))
    conditions.json         {cid: {class, alpha, norm, batch_slot, concept | seed}}
    captures_sparse_L{45,50}.pt   {(set, prompt_idx, cid): (indices int32, values fp16)}
    ../run_summary.json     provenance + sanity block

Statistics
----------
* D(S) = mean over (prompt, vector) of the per-row shift (e.g. dpre).
* Hierarchical bootstrap: resample vectors with replacement, then (independently) prompts
  with replacement; 2000 resamples, seed 0.  The vector resample is shared across sets /
  alphas / gates of the same class (so ratios between sets are paired on vectors); the prompt
  resample is drawn independently per set.  95% CI = [2.5, 97.5] percentiles, 99.44% CI =
  [0.28, 99.72] percentiles.
"""

from __future__ import annotations

import ast
import json
import math
import os
import re
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
sys.path.insert(0, str(REPO_ROOT / "src"))

SET_ORDER = ["S1", "S2", "S2v", "S3", "S4", "S5", "S6", "S7"]
FULL_SETS = ["S1", "S2", "S2v", "S3", "S4"]
CLASSES = ["concept", "random"]
RATIO_REFS = {"R1": "S1", "R2": "S2", "R2v": "S2v"}
N_BOOT = 2000
BOOT_SEED = 0
CI95 = (2.5, 97.5)
CI9944 = (0.28, 99.72)
N_NULL = 200
N_INSPECT = 20
INSPECT_ALPHA = 4.0
SPARSE_MAX_GB = float(os.environ.get("SFO_SPARSE_MAX_GB", "8"))

# alias lists so the script tolerates the WO's names as well as the run script's names
ALIASES = {
    "pre": ("pre", "pre_g"),
    "act": ("act", "act_g"),
    "thr": ("thr", "threshold_g", "threshold"),
    "dpre": ("dpre",),
    "dact": ("dact",),
    "direct": ("direct_pre", "pre_direct"),
    "cos": ("cos_wv", "cos_wenc_v"),
}


# ----------------------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------------------
def clean(x: Any) -> Any:
    """Recursively convert numpy types and replace NaN/inf with None for strict JSON."""
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


def akey(a: float) -> str:
    return f"{float(a):g}"


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


def pick(d: dict, name: str) -> Any:
    for k in ALIASES[name]:
        if k in d:
            return d[k]
    return None


def pick_mean_last(d: dict, kind: str) -> Any:
    """mean{N}_pre / mean{N}_act (run script) or pre_mean_last20 / act_mean_last20 (WO)."""
    for k, v in d.items():
        if re.fullmatch(rf"mean\d+_{kind}", k) or re.fullmatch(rf"{kind}_mean_last\d+", k):
            return v
    return None


def as_arr(v: Any, n: int) -> np.ndarray:
    if v is None:
        return np.full(n, np.nan)
    return np.array([np.nan if e is None else e for e in v], dtype=float)


# ----------------------------------------------------------------------------------------
# hierarchical bootstrap
# ----------------------------------------------------------------------------------------
def _counts(rng: np.random.Generator, n_boot: int, n: int) -> np.ndarray:
    idx = rng.integers(0, n, size=(n_boot, n))
    return (idx[:, :, None] == np.arange(n)[None, None, :]).sum(axis=1).astype(np.float64)


class Boot:
    """Pre-drawn resample count matrices: Cp[set] [n_boot,P], Cv[class] [n_boot,V]."""

    def __init__(self, n_boot: int, seed: int, set_p: Dict[str, int], class_v: Dict[str, int]):
        self.n_boot = n_boot
        self.Cp: Dict[str, np.ndarray] = {}
        self.Cv: Dict[str, np.ndarray] = {}
        for s, p in set_p.items():
            rng = np.random.default_rng([seed, SET_ORDER.index(s) if s in SET_ORDER else 99, 1])
            self.Cp[s] = _counts(rng, n_boot, p)
        for ci, k in enumerate(CLASSES):
            if k in class_v:
                rng = np.random.default_rng([seed, 100 + ci, 2])
                self.Cv[k] = _counts(rng, n_boot, class_v[k])

    def means(self, M: np.ndarray, s: str, k: str) -> np.ndarray:
        """Bootstrap distribution of the mean of M [P,V] (NaN entries masked)."""
        fin = np.isfinite(M)
        M0 = np.where(fin, M, 0.0)
        W = fin.astype(np.float64)
        Cp, Cv = self.Cp[s], self.Cv[k]
        num = np.einsum("bp,pv,bv->b", Cp, M0, Cv)
        den = np.einsum("bp,pv,bv->b", Cp, W, Cv)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def prompt_means(self, r: np.ndarray, s: str) -> np.ndarray:
        """Bootstrap (prompts only) of nanmean of per-prompt vector r [P]."""
        fin = np.isfinite(r)
        Cp = self.Cp[s]
        num = np.einsum("bp,p->b", Cp, np.where(fin, r, 0.0))
        den = np.einsum("bp,p->b", Cp, fin.astype(np.float64))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(den > 0, num / den, np.nan)


def ci_block(point: float, boots: Optional[np.ndarray]) -> Dict[str, Any]:
    """WO schema: {m, lo95, hi95, lo9944, hi9944}."""
    if boots is None:
        return {"m": point, "lo95": None, "hi95": None, "lo9944": None, "hi9944": None}
    lo95, hi95 = pct(boots, CI95)
    lo9944, hi9944 = pct(boots, CI9944)
    return {"m": point, "lo95": lo95, "hi95": hi95, "lo9944": lo9944, "hi9944": hi9944}


# ----------------------------------------------------------------------------------------
# data loading
# ----------------------------------------------------------------------------------------
def load_conditions(raw: Path) -> Tuple[List[str], Dict[str, dict]]:
    conds = json.loads((raw / "conditions.json").read_text())
    return list(conds.keys()), conds


def vec_id(c: dict) -> str:
    if c["class"] == "concept":
        return str(c.get("concept"))
    if c["class"] == "random":
        return f"s{c.get('seed')}"
    return ""


class GateData:
    """Dense arrays [n_conds, P] per (set, quantity, gate)."""

    def __init__(self) -> None:
        self.Q: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.gates: Dict[str, Tuple[int, int]] = {}  # gate key -> (layer, feature id)
        self.P: Dict[str, int] = {}
        self.thr: Dict[str, float] = {}


def load_gate_data(path: Path, cids: List[str]) -> GateData:
    rows = read_jsonl(path)
    cid_i = {c: i for i, c in enumerate(cids)}
    gd = GateData()
    for r in rows:
        gd.P[r["set"]] = max(gd.P.get(r["set"], 0), int(r["prompt_idx"]) + 1)
    n = len(cids)
    for r in rows:
        s, p, ci = r["set"], int(r["prompt_idx"]), cid_i.get(r["cid"])
        if ci is None:
            continue
        P = gd.P[s]
        for lk, d in r.items():
            if not re.fullmatch(r"L\d+", lk) or not isinstance(d, dict):
                continue
            layer = int(lk[1:])
            ids = d["gate_ids"]
            ng = len(ids)
            vals = {
                "pre": as_arr(pick(d, "pre"), ng),
                "act": as_arr(pick(d, "act"), ng),
                "thr": as_arr(pick(d, "thr"), ng),
                "dpre": as_arr(pick(d, "dpre"), ng),
                "direct": as_arr(pick(d, "direct"), ng),
                "cos": as_arr(pick(d, "cos"), ng),
                "mpre": as_arr(pick_mean_last(d, "pre"), ng),
                "mact": as_arr(pick_mean_last(d, "act"), ng),
            }
            for gi, gid in enumerate(ids):
                gk = f"L{layer}_g{gid}"
                gd.gates[gk] = (layer, int(gid))
                qs = gd.Q.setdefault(s, {})
                for q, v in vals.items():
                    arr = qs.setdefault(q, {}).get(gk)
                    if arr is None:
                        arr = np.full((n, P), np.nan)
                        qs[q][gk] = arr
                    arr[ci, p] = v[gi]
    return gd


def load_residual(path: Path, cids: List[str], P: Dict[str, int]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """out[set][quantity][layer-key] -> [n_conds, P]."""
    rows = read_jsonl(path)
    cid_i = {c: i for i, c in enumerate(cids)}
    out: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    n = len(cids)
    for r in rows:
        s, p, ci = r["set"], int(r["prompt_idx"]), cid_i.get(r["cid"])
        if ci is None or s not in P:
            continue
        for lk, d in r.items():
            if not re.fullmatch(r"L\d+", lk) or not isinstance(d, dict):
                continue
            for q in ("reach", "reach_indirect", "dirproj"):
                arr = out.setdefault(s, {}).setdefault(q, {}).get(lk)
                if arr is None:
                    arr = np.full((n, P[s]), np.nan)
                    out[s][q][lk] = arr
                arr[ci, p] = fl(d.get(q))
    return out


def load_ld(path: Path, cids: List[str], P: Dict[str, int]) -> Dict[str, np.ndarray]:
    rows = read_jsonl(path)
    cid_i = {c: i for i, c in enumerate(cids)}
    out: Dict[str, np.ndarray] = {}
    for r in rows:
        s, p, ci = r["set"], int(r["prompt_idx"]), cid_i.get(r["cid"])
        if ci is None or s not in P:
            continue
        if s not in out:
            out[s] = np.full((len(cids), P[s]), np.nan)
        out[s][ci, p] = fl(r.get("ld"))
    return out


# ----------------------------------------------------------------------------------------
# sparse captures
# ----------------------------------------------------------------------------------------
def parse_sparse_key(k: Any) -> Optional[Tuple[str, int, str]]:
    if isinstance(k, (tuple, list)) and len(k) == 3:
        return str(k[0]), int(k[1]), str(k[2])
    if isinstance(k, str):
        try:
            v = ast.literal_eval(k)
            if isinstance(v, (tuple, list)) and len(v) == 3:
                return str(v[0]), int(v[1]), str(v[2])
        except (ValueError, SyntaxError):
            pass
        for sep in ("|", "::", ":", ","):
            parts = k.split(sep, 2)
            if len(parts) == 3:
                try:
                    return parts[0].strip(), int(parts[1]), parts[2].strip()
                except ValueError:
                    continue
    return None


def _to_np(t: Any, dtype: Any) -> np.ndarray:
    if hasattr(t, "detach"):
        t = t.detach().cpu().float().numpy() if np.issubdtype(np.dtype(dtype), np.floating) \
            else t.detach().cpu().numpy()
    return np.asarray(t).astype(dtype)


def split_sparse_value(v: Any) -> Tuple[np.ndarray, np.ndarray]:
    if isinstance(v, dict):
        idx = v.get("indices", v.get("idx"))
        val = v.get("values", v.get("val"))
    else:
        idx, val = v[0], v[1]
    return _to_np(idx, np.int64), _to_np(val, np.float64)


def load_sparse_layer(
    path: Path, cond_meta: Dict[str, dict], sets: List[str], P: Dict[str, int]
) -> Dict[str, Any]:
    """Accumulate per-feature dense statistics from one captures_sparse_L*.pt.

    Returns dict with, per set s: active[s] bool [W], base_sum[s], base_nz[s], n_base[s];
    and Dact[(s, alpha, class)] float32 [W] = mean steered act - mean baseline act.
    """
    import torch  # local import: only needed here

    d = torch.load(path, map_location="cpu", weights_only=False)
    parsed: List[Tuple[Tuple[str, int, str], Any]] = []
    width = 0
    for k, v in d.items():
        pk = parse_sparse_key(k)
        if pk is None or pk[2] not in cond_meta or pk[0] not in sets:
            continue
        idx, val = split_sparse_value(v)
        if idx.size:
            width = max(width, int(idx.max()) + 1)
        parsed.append((pk, (idx, val)))
    del d
    width = max(width, 262144)

    active = {s: np.zeros(width, dtype=bool) for s in sets}
    base_sum = {s: np.zeros(width) for s in sets}
    base_nz = {s: np.zeros(width) for s in sets}
    n_base = {s: 0 for s in sets}
    acc: Dict[Tuple[str, float, str], np.ndarray] = {}
    n_acc: Dict[Tuple[str, float, str], int] = defaultdict(int)
    n_keys = 0
    for (s, _p, cid), (idx, val) in parsed:
        n_keys += 1
        m = cond_meta[cid]
        nzmask = val != 0
        active[s][idx[nzmask]] = True
        if m["class"] == "baseline":
            base_sum[s][idx] += val
            base_nz[s][idx[nzmask]] += 1
            n_base[s] += 1
        elif m["class"] in CLASSES:
            key = (s, float(m["alpha"]), m["class"])
            if key not in acc:
                acc[key] = np.zeros(width)
            acc[key][idx] += val
            n_acc[key] += 1
    Dact: Dict[Tuple[str, float, str], np.ndarray] = {}
    for key, a in acc.items():
        s = key[0]
        if n_base[s] == 0 or n_acc[key] == 0:
            continue
        Dact[key] = (a / n_acc[key] - base_sum[s] / n_base[s]).astype(np.float32)
    # steered mean act (needed for relative change): keep alongside Dact
    steer_mean = {key: (acc[key] / n_acc[key]).astype(np.float32) for key in Dact}
    return {
        "width": width, "active": active, "base_sum": base_sum, "base_nz": base_nz,
        "n_base": n_base, "Dact": Dact, "steer_mean": steer_mean, "n_keys": n_keys,
    }


def _ranks01(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    return r / max(len(x) - 1, 1)


def decile_edges(x: np.ndarray) -> np.ndarray:
    return np.quantile(x, np.linspace(0.1, 0.9, 9))


def matched_null_pool(
    sp: Dict[str, Any], s: str, layer: int, gid: int, set_idx: int
) -> Optional[Dict[str, Any]]:
    """200 features (same layer): same decile of base_active_frac AND of mean base act on S,
    sampled (seed 0) from the active-in-S features; fall back to nearest in percentile space."""
    nb = sp["n_base"][s]
    # Spec: features active at baseline on S (not merely active in any steered row).
    base_active = sp["base_nz"][s] > 0
    if nb == 0 or not base_active[gid]:
        return None
    ids = np.flatnonzero(base_active)
    frac = sp["base_nz"][s][ids] / nb
    mact = sp["base_sum"][s][ids] / nb
    fe, me = decile_edges(frac), decile_edges(mact)
    fd, md = np.digitize(frac, fe), np.digitize(mact, me)
    pos_g = int(np.searchsorted(ids, gid))
    if pos_g >= len(ids) or ids[pos_g] != gid:
        return None
    gfd, gmd = fd[pos_g], md[pos_g]
    cand = np.ones(len(ids), dtype=bool)
    cand[pos_g] = False
    exact = np.flatnonzero(cand & (fd == gfd) & (md == gmd))
    rng = np.random.default_rng([BOOT_SEED, layer, gid, set_idx])
    if len(exact) >= N_NULL:
        chosen = rng.choice(exact, size=N_NULL, replace=False)
        n_exact = N_NULL
    else:
        n_exact = len(exact)
        rf, rm = _ranks01(frac), _ranks01(mact)
        dist = np.hypot(rf - rf[pos_g], rm - rm[pos_g])
        dist[pos_g] = np.inf
        dist[exact] = np.inf
        need = min(N_NULL - n_exact, int(np.isfinite(dist).sum()))
        near = np.argsort(dist, kind="stable")[:need]
        chosen = np.concatenate([exact, near])
    return {"feat": ids[chosen], "n_exact_decile": int(n_exact),
            "n_matched": int(len(chosen)), "gate_decile": [int(gfd), int(gmd)]}


def sparse_gate_metrics(
    sp: Dict[str, Any], pools: Dict[Tuple[str, int], Optional[Dict[str, Any]]],
    s: str, a: float, k: str, layer: int, gid: int,
) -> Dict[str, Any]:
    key = (s, float(a), k)
    out: Dict[str, Any] = {"pct_rank_active": None, "n_active": None, "abs_rank": None,
                           "matched_null": None}
    if key not in sp["Dact"] or s not in sp["active"]:
        return out
    act = sp["active"][s]
    ids = np.flatnonzero(act)
    D = sp["Dact"][key].astype(np.float64)
    out["n_active"] = int(len(ids))
    if act[gid]:
        d = D[ids]
        g = D[gid]
        # 0 = most negative among active features (percentile on 0–100 scale).
        out["pct_rank_active"] = float(100.0 * np.mean(d <= g))
        # Absolute rank among active: 1 = most negative D_act.
        out["abs_rank"] = int(1 + np.sum(d < g))
    pool = pools.get((s, gid))
    if pool is not None:
        nb = sp["n_base"][s]
        B = sp["base_sum"][s] / nb
        M = sp["steer_mean"][key].astype(np.float64)
        f = pool["feat"]
        ok = B[f] > 0
        rel = (M[f][ok] - B[f][ok]) / B[f][ok]
        gate_rel = float((M[gid] - B[gid]) / B[gid]) if B[gid] > 0 else None
        q = pct(rel, (25, 50, 75)) if rel.size else [None] * 3
        out["matched_null"] = {
            "median": q[1], "q25": q[0], "q75": q[2], "gate_rel": gate_rel,
            "n_matched": pool["n_matched"], "n_exact_decile": pool["n_exact_decile"],
            "gate_decile_frac_meanact": pool["gate_decile"], "n_valid_rel": int(ok.sum()),
            "gate_rel_pctl_in_null": (float(100.0 * np.mean(rel <= gate_rel))
                                      if (gate_rel is not None and rel.size) else None),
        }
    return out


def sparse_inspection(sp: Dict[str, Any], s: str, a: float, layer: int,
                      gate_ids: Sequence[int]) -> Optional[Dict[str, Any]]:
    key = (s, float(a), "concept")
    if key not in sp["Dact"]:
        return None
    act = sp["active"][s]
    ids = np.flatnonzero(act)
    D = sp["Dact"][key].astype(np.float64)[ids]
    nb = max(sp["n_base"][s], 1)
    bm = sp["base_sum"][s][ids] / nb
    bf = sp["base_nz"][s][ids] / nb
    gset = set(int(g) for g in gate_ids)

    def entry(j: int) -> Dict[str, Any]:
        return {"id": int(ids[j]), "D_act": float(D[j]), "base_mean": float(bm[j]),
                "base_active_frac": float(bf[j]), "is_gate": int(ids[j]) in gset}

    order = np.argsort(D, kind="stable")
    neg = [entry(int(j)) for j in order[:N_INSPECT] if D[j] < 0]
    pos = [entry(int(j)) for j in order[::-1][:N_INSPECT] if D[j] > 0]
    return {"n_active": int(len(ids)), "most_neg": neg, "most_pos": pos}


# ----------------------------------------------------------------------------------------
# main metric computation
# ----------------------------------------------------------------------------------------
def cond_groups(cids: List[str], cmeta: Dict[str, dict]) -> Dict[Tuple[str, float], List[int]]:
    """(class, alpha) -> cond indices sorted by vector id; plus ('noise_floor'/'baseline', 0)."""
    g: Dict[Tuple[str, float], List[Tuple[str, int]]] = defaultdict(list)
    for i, c in enumerate(cids):
        m = cmeta[c]
        g[(m["class"], float(m["alpha"]))].append((vec_id(m), i))
    return {k: [i for _, i in sorted(v)] for k, v in g.items()}


def main(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    cfg_path = REPO_ROOT / "configs" / "wo_003.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    summary: Dict[str, Any] = {}
    if (results_dir / "run_summary.json").exists():
        summary = json.loads((results_dir / "run_summary.json").read_text())

    cids, cmeta = load_conditions(raw)
    groups = cond_groups(cids, cmeta)
    base_i = cids.index("baseline")
    nf_idx = groups.get(("noise_floor", 0.0), [])
    alphas = sorted({a for (k, a) in groups if k in CLASSES and a > 0})
    class_v = {k: len(groups[(k, alphas[0])]) for k in CLASSES if (k, alphas[0]) in groups}
    for k in class_v:  # all alphas must share the same vector count
        for a in alphas:
            assert len(groups.get((k, a), [])) == class_v[k], (k, a)

    gd = load_gate_data(raw / "gate_scalars.jsonl", cids)
    sets = [s for s in SET_ORDER if s in gd.P]
    P = {s: gd.P[s] for s in sets}
    boot = Boot(N_BOOT, BOOT_SEED, P, class_v)
    gate_keys = sorted(gd.gates, key=lambda g: (gd.gates[g][0], gd.gates[g][1]))

    res_data = load_residual(raw / "residual_scalars.jsonl", cids, P)
    ld_data = load_ld(raw / "logits_ld.jsonl", cids, P)

    # ---------------------------------------------------------------- sparse (optional)
    sparse: Dict[int, Dict[str, Any]] = {}
    sparse_status: Dict[str, Any] = {}
    layers = sorted({gd.gates[g][0] for g in gate_keys})
    for l in layers:
        p = raw / f"captures_sparse_L{l}.pt"
        if not p.exists():
            sparse_status[f"L{l}"] = "missing"
            continue
        gb = p.stat().st_size / 1e9
        if gb > SPARSE_MAX_GB:
            sparse_status[f"L{l}"] = f"skipped (file {gb:.2f} GB > SFO_SPARSE_MAX_GB={SPARSE_MAX_GB})"
            continue
        try:
            sparse[l] = load_sparse_layer(p, cmeta, sets, P)
            sparse_status[f"L{l}"] = {"loaded": True, "n_keys": sparse[l]["n_keys"],
                                      "width": sparse[l]["width"], "file_gb": gb}
        except Exception as e:  # noqa: BLE001
            sparse_status[f"L{l}"] = f"failed: {type(e).__name__}: {e}"

    pools: Dict[int, Dict[Tuple[str, int], Optional[Dict[str, Any]]]] = {}
    for l, sp in sparse.items():
        pools[l] = {}
        for gk in gate_keys:
            gl, gid = gd.gates[gk]
            if gl != l:
                continue
            for s in sets:
                pools[l][(s, gid)] = matched_null_pool(sp, s, l, gid, SET_ORDER.index(s))

    # ---------------------------------------------------------------- gate metrics
    gates_out: Dict[str, Any] = {}
    boot_d: Dict[Tuple[str, float, str, str], np.ndarray] = {}
    point_d: Dict[Tuple[str, float, str, str], float] = {}
    dpre_mats: Dict[Tuple[str, float, str, str], np.ndarray] = {}

    for s in sets:
        Q = gd.Q[s]
        for gk in gate_keys:
            if gk not in Q["pre"]:
                continue
            layer, gid = gd.gates[gk]
            pre, act, thr = Q["pre"][gk], Q["act"][gk], Q["thr"][gk]
            base_pre, base_act = pre[base_i], act[base_i]
            base_thr = thr[base_i]
            dpre = np.where(np.isnan(Q["dpre"][gk]), pre - base_pre[None, :], Q["dpre"][gk])
            dact = act - base_act[None, :]
            dlast = Q["mpre"][gk] - Q["mpre"][gk][base_i][None, :]
            direct = Q["direct"][gk]
            cos = Q["cos"][gk]
            margin = base_pre - base_thr
            base_info = {
                "base_pre_mean": nanmean(base_pre),
                "base_margin_mean": nanmean(margin),
                "base_active_frac": nanmean(np.where(np.isfinite(base_act),
                                                     (base_act > 0).astype(float), np.nan)),
            }
            nf = nanmean(np.abs(dpre[nf_idx])) if nf_idx else float("nan")
            abs_margin = nanmean(np.abs(margin))
            for a in alphas:
                for k in CLASSES:
                    ci = groups.get((k, a))
                    if not ci:
                        continue
                    M = dpre[ci].T                     # [P, V]
                    b = boot.means(M, s, k)
                    D = nanmean(M)
                    boot_d[(s, a, k, gk)] = b
                    point_d[(s, a, k, gk)] = D
                    dpre_mats[(s, a, k, gk)] = M
                    Ma = dact[ci].T
                    Ml = dlast[ci].T
                    Md = direct[ci].T.copy()
                    Mp = M.copy()
                    both = np.isfinite(Md) & np.isfinite(Mp)
                    Md[~both] = np.nan
                    Mp[~both] = np.nan
                    bd, bp = boot.means(Md, s, k), boot.means(Mp, s, k)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        bfrac = bd / bp
                        frac = nanmean(Md) / nanmean(Mp) if nanmean(Mp) != 0 else float("nan")
                    d_blk = ci_block(D, b)
                    cell: Dict[str, Any] = dict(base_info)
                    cell.update({
                        "layer": layer, "gate_id": gid, "n_prompts": P[s], "n_vectors": len(ci),
                        "D": d_blk,
                        "D_act": ci_block(nanmean(Ma), boot.means(Ma, s, k)),
                        "D_last20": ci_block(nanmean(Ml), boot.means(Ml, s, k)),
                        "noise_floor": nf,
                        "rel_margin": {
                            **{
                                k2: (None if (v is None or abs_margin <= 0) else v / abs_margin)
                                for k2, v in d_blk.items()
                            },
                            "m": (D / abs_margin) if abs_margin > 0 else None,
                            "denominator_mean_abs_base_margin": abs_margin,
                        },
                        "direct_frac": frac,
                        "direct_frac_ci": ci_block(frac, bfrac),
                        "mean_direct": nanmean(Md),
                        "mean_dpre": nanmean(Mp),
                        "mean_cos_wenc_v": nanmean(cos[ci]),
                    })
                    if layer in sparse:
                        cell.update(sparse_gate_metrics(sparse[layer], pools[layer], s, a, k,
                                                        layer, gid))
                    else:
                        cell.update({"pct_rank_active": None, "n_active": None,
                                     "abs_rank": None, "matched_null": None})
                    gates_out.setdefault(s, {}).setdefault(akey(a), {}).setdefault(k, {})[gk] = cell

    # ratios (need all sets' bootstrap means)
    for s in sets:
        for a in alphas:
            for k in CLASSES:
                for gk in gate_keys:
                    cell = gates_out.get(s, {}).get(akey(a), {}).get(k, {}).get(gk)
                    if cell is None:
                        continue
                    for rname, ref in RATIO_REFS.items():
                        if ref == s or (ref, a, k, gk) not in boot_d:
                            cell[rname] = None
                            continue
                        d0, d1 = point_d[(s, a, k, gk)], point_d[(ref, a, k, gk)]
                        with np.errstate(divide="ignore", invalid="ignore"):
                            br = boot_d[(s, a, k, gk)] / boot_d[(ref, a, k, gk)]
                            lg = np.log(np.abs(br))
                        lg = lg[np.isfinite(lg)]
                        lo95, hi95 = pct(br, CI95)
                        lo9944, hi9944 = pct(br, CI9944)
                        cell[rname] = {
                            "m": (d0 / d1) if d1 != 0 else None,
                            "lo95": lo95, "hi95": hi95, "lo9944": lo9944, "hi9944": hi9944,
                            "ref": ref,
                            "log_abs_ratio_pctl": {
                                "p0.28": pct(lg, [0.28])[0], "p2.5": pct(lg, [2.5])[0],
                                "p25": pct(lg, [25])[0], "p50": pct(lg, [50])[0],
                                "p75": pct(lg, [75])[0], "p97.5": pct(lg, [97.5])[0],
                                "p99.72": pct(lg, [99.72])[0],
                            },
                            "n_boot_finite_ratio": int(np.isfinite(br).sum()),
                        }

    # ---------------------------------------------------------------- residual
    residual_out: Dict[str, Any] = {}
    for s in sets:
        for q, per_layer in res_data.get(s, {}).items():
            for lk, arr in per_layer.items():
                nfm = nanmean(arr[nf_idx]) if nf_idx else float("nan")
                for a in alphas:
                    for k in CLASSES:
                        ci = groups.get((k, a))
                        if not ci:
                            continue
                        M = arr[ci].T
                        residual_out.setdefault(s, {}).setdefault(akey(a), {}).setdefault(k, {}) \
                            .setdefault(lk, {})[q] = {
                                **ci_block(nanmean(M), boot.means(M, s, k)),
                                "noise_floor_mean": nfm,
                            }

    # ---------------------------------------------------------------- logits
    logits_out: Dict[str, Any] = {}
    for s in sets:
        if s not in ld_data:
            continue
        L = ld_data[s]
        base_ld = L[base_i]
        dld = L - base_ld[None, :]
        nf_ld = nanmean(np.abs(dld[nf_idx])) if nf_idx else float("nan")
        for a in alphas:
            for k in CLASSES:
                ci = groups.get((k, a))
                if not ci:
                    continue
                M = dld[ci].T
                ent: Dict[str, Any] = {
                    "base_ld_mean": nanmean(base_ld),
                    "noise_floor_mean_abs_dld": nf_ld,
                    "dLD": ci_block(nanmean(M), boot.means(M, s, k)),
                    "corr_dpre_dLD": {},
                }
                for gk in gate_keys:
                    key = (s, a, k, gk)
                    if key not in dpre_mats:
                        continue
                    X = dpre_mats[key]                  # [P, V]
                    r = np.full(X.shape[0], np.nan)
                    for pi in range(X.shape[0]):
                        x, y = X[pi], M[pi]
                        msk = np.isfinite(x) & np.isfinite(y)
                        if msk.sum() >= 3 and np.std(x[msk]) > 0 and np.std(y[msk]) > 0:
                            r[pi] = float(np.corrcoef(x[msk], y[msk])[0, 1])
                    # Spec: per-gate Pearson r (mean over prompts); keep CI under *_ci.
                    ent["corr_dpre_dLD"][gk] = nanmean(r)
                    ent.setdefault("corr_dpre_dLD_ci", {})[gk] = {
                        **ci_block(nanmean(r), boot.prompt_means(r, s)),
                        "n_prompts_with_corr": int(np.isfinite(r).sum()),
                        "ci_note": "bootstrap over prompts only",
                    }
                logits_out.setdefault(s, {}).setdefault(akey(a), {})[k] = ent

    # ---------------------------------------------------------------- inspection
    inspection: Dict[str, Any] = {}
    ia = INSPECT_ALPHA if INSPECT_ALPHA in alphas else (max(alphas) if alphas else None)
    for l, sp in sparse.items():
        gids = [gd.gates[g][1] for g in gate_keys if gd.gates[g][0] == l]
        for s in sets:
            blk = sparse_inspection(sp, s, ia, l, gids) if ia is not None else None
            if blk is not None:
                inspection.setdefault(s, {})[f"L{l}"] = {"alpha": ia, "class": "concept", **blk}

    # ---------------------------------------------------------------- sanity
    n_prompts_total = sum(P.values())
    n_gate_rows = sum(1 for _ in open(raw / "gate_scalars.jsonl")) if (raw / "gate_scalars.jsonl").exists() else 0
    nfd = {}
    for gk in gate_keys:
        vals = []
        for s in sets:
            if gk in gd.Q[s]["pre"]:
                pre = gd.Q[s]["pre"][gk]
                d = np.where(np.isnan(gd.Q[s]["dpre"][gk]), pre - pre[base_i][None, :],
                             gd.Q[s]["dpre"][gk])
                vals.append(np.abs(d[nf_idx]).ravel())
        v = np.concatenate(vals) if vals else np.array([])
        nfd[gk] = {"mean_abs": nanmean(v), "max_abs": float(np.nanmax(v)) if v.size and np.isfinite(v).any() else None}
    computed = {
        "n_conditions": len(cids),
        "expected_rows": n_prompts_total * len(cids),
        "n_gate_rows": n_gate_rows,
        "n_residual_rows": sum(1 for _ in open(raw / "residual_scalars.jsonl")) if (raw / "residual_scalars.jsonl").exists() else 0,
        "n_ld_rows": sum(1 for _ in open(raw / "logits_ld.jsonl")) if (raw / "logits_ld.jsonl").exists() else 0,
        "n_failed_rows": sum(1 for _ in open(raw / "failed_rows.jsonl")) if (raw / "failed_rows.jsonl").exists() else None,
        "prompts_per_set": P,
        "vectors_per_class": class_v,
        "alphas": alphas,
        "noise_floor_conditions": [cids[i] for i in nf_idx],
        "noise_floor_abs_dpre": nfd,
        "baseline_missing_prompts": {
            s: int(np.sum(~np.isfinite(gd.Q[s]["pre"][gate_keys[0]][base_i]))) for s in sets
        } if gate_keys else {},
        "n_nan_dpre_cells": int(sum(
            np.sum(~np.isfinite(gd.Q[s]["pre"][gk])) for s in sets for gk in gate_keys
            if gk in gd.Q[s]["pre"])),
        "sparse_status": sparse_status,
        "bootstrap": {"n_resamples": N_BOOT, "seed": BOOT_SEED, "ci95_pctl": CI95,
                      "ci99_44_pctl": CI9944, "n_null_features": N_NULL},
    }
    sanity = {
        "run_status": summary.get("status"),
        "run_errors": summary.get("errors"),
        "run_rows": summary.get("rows"),
        "run_summary_sanity": summary.get("sanity"),
        "run_projection": summary.get("projection"),
        "s2v_texts_provisional": summary.get("s2v_texts_provisional"),
        "computed": computed,
    }

    transcoder_rev = summary.get("transcoder_revision") or os.environ.get("SFO_TRANSCODER_REVISION")
    if not transcoder_rev:
        trev_path = results_dir / "transcoder_revision.json"
        if trev_path.exists():
            transcoder_rev = json.loads(trev_path.read_text()).get("revision")
    batch_size = (summary.get("config") or {}).get("batch_size", cfg.get("batch_size"))
    metrics = {
        "wo_id": "WO-003",
        "gates": gates_out,
        "residual": residual_out,
        "logits": logits_out,
        "inspection": inspection,
        "sanity": sanity,
        "git_hash": summary.get("git_hash"),
        "model_revision": summary.get("model_revision"),
        "transcoder_revision": transcoder_rev,
        "dataset_revisions": summary.get("dataset_revisions"),
        "batch_size": batch_size,
    }
    (results_dir / "metrics.json").write_text(json.dumps(clean(metrics), indent=2))

    # ---------------------------------------------------------------- figures
    make_figures(results_dir, gates_out, residual_out, sets, gate_keys, alphas)
    print(f"Wrote {results_dir / 'metrics.json'} and figures")


# ----------------------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------------------
def _g(d: Any, *path: Any) -> Any:
    for p in path:
        if d is None:
            return None
        d = d.get(p) if isinstance(d, dict) else None
    return d


def _nz(x: Any) -> float:
    return float("nan") if x is None else float(x)


def make_figures(results_dir: Path, gates: dict, residual: dict, sets: List[str],
                 gate_keys: List[str], alphas: List[float]) -> None:
    ng = max(len(gate_keys), 1)
    x = np.arange(len(sets))
    a_hi = akey(INSPECT_ALPHA if INSPECT_ALPHA in alphas else (max(alphas) if alphas else 4))
    a_lo = akey(min(alphas)) if alphas else "2"

    # ---- fig_gate_dpre
    fig, axes = plt.subplots(1, ng, figsize=(5.2 * ng, 4), squeeze=False)
    specs = [("concept", a_lo, "C0", -0.2), ("concept", a_hi, "C1", 0.0), ("random", a_hi, "C2", 0.2)]
    for ax, gk in zip(axes[0], gate_keys):
        for k, a, col, off in specs:
            y = np.array([_nz(_g(gates, s, a, k, gk, "D", "m")) for s in sets])
            lo = np.array([_nz(_g(gates, s, a, k, gk, "D", "lo95")) for s in sets])
            hi = np.array([_nz(_g(gates, s, a, k, gk, "D", "hi95")) for s in sets])
            ax.errorbar(x + off, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                        fmt="o", color=col, ms=4, capsize=2, label=f"{k} α={a}")
        nf = np.array([_nz(_g(gates, s, a_hi, "concept", gk, "noise_floor")) for s in sets])
        ax.fill_between(x, -nf, nf, color="gray", alpha=0.25, label="noise floor ±mean|Δpre|")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(sets, fontsize=8)
        ax.set_ylabel("D(S) = mean Δpre")
        ax.set_title(gk)
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_gate_dpre.png", dpi=150)
    plt.close(fig)

    # ---- fig_engagement
    fig, axes = plt.subplots(1, ng, figsize=(5.2 * ng, 4), squeeze=False)
    for ax, gk in zip(axes[0], gate_keys):
        xs, ys, lo99, hi99 = [], [], [], []
        for i, s in enumerate(sets):
            r = _g(gates, s, a_hi, "concept", gk, "R2")
            if not r or r.get("m") is None:
                continue
            xs.append(i)
            ys.append(r["m"])
            lo99.append(_nz(r.get("lo9944")))
            hi99.append(_nz(r.get("hi9944")))
        ys_a, lo_a, hi_a = np.array(ys), np.array(lo99), np.array(hi99)
        if len(xs):
            ax.errorbar(xs, ys_a, yerr=[np.clip(ys_a - lo_a, 0, None), np.clip(hi_a - ys_a, 0, None)],
                        fmt="o", color="C1", capsize=3)
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axhline(0.25, color="gray", ls="--", lw=0.8)
        ax.axhline(-0.25, color="gray", ls="--", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(sets, fontsize=8)
        ax.set_ylabel("R2 = D(S)/D(S2)  [99.44% CI]")
        ax.set_title(f"{gk} (concept α={a_hi})")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_engagement.png", dpi=150)
    plt.close(fig)

    # ---- fig_direct_indirect
    fig, axes = plt.subplots(1, ng, figsize=(5.2 * ng, 4.4), squeeze=False)
    for ax, gk in zip(axes[0], gate_keys):
        px, py = [], []
        for i, s in enumerate(sets):
            c = _g(gates, s, a_hi, "concept", gk)
            if not c:
                continue
            dx = _nz(c.get("mean_direct"))
            dy = _nz(_g(c, "D", "m"))
            lo, hi = _nz(_g(c, "D", "lo95")), _nz(_g(c, "D", "hi95"))
            ax.errorbar(dx, dy, yerr=[[max(dy - lo, 0)], [max(hi - dy, 0)]], fmt="o",
                        color=f"C{i % 10}", capsize=2, label=s)
            px.append(dx)
            py.append(dy)
        allv = [v for v in px + py if np.isfinite(v)]
        if allv:
            lim = (min(allv), max(allv))
            pad = 0.05 * (lim[1] - lim[0] + 1e-9)
            ax.plot([lim[0] - pad, lim[1] + pad], [lim[0] - pad, lim[1] + pad], "k--", lw=0.7,
                    label="y = x")
        ax.set_xlabel("mean direct-path Δpre")
        ax.set_ylabel("mean actual Δpre")
        ax.set_title(f"{gk} (concept α={a_hi})")
    axes[0][0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_direct_indirect.png", dpi=150)
    plt.close(fig)

    # ---- fig_reach
    rl = sorted({lk for s in residual for a in residual[s] for k in residual[s][a]
                 for lk in residual[s][a][k]})
    rl = [l for l in rl if l in ("L45", "L50")] or rl
    fig, axes = plt.subplots(1, max(len(rl), 1), figsize=(5.5 * max(len(rl), 1), 4), squeeze=False)
    for ax, lk in zip(axes[0], rl):
        for q, col, off in (("reach", "C0", -0.15), ("reach_indirect", "C3", 0.15)):
            for k, filled in (("concept", True), ("random", False)):
                y = np.array([_nz(_g(residual, s, a_hi, k, lk, q, "m")) for s in sets])
                lo = np.array([_nz(_g(residual, s, a_hi, k, lk, q, "lo95")) for s in sets])
                hi = np.array([_nz(_g(residual, s, a_hi, k, lk, q, "hi95")) for s in sets])
                dx = off + (0.0 if filled else 0.07)
                ax.errorbar(x + dx, y, yerr=[np.clip(y - lo, 0, None), np.clip(hi - y, 0, None)],
                            fmt="o" if filled else "s", mfc=col if filled else "none", color=col,
                            ms=4, capsize=2, label=f"{q} ({k})")
        ax.set_xticks(x)
        ax.set_xticklabels(sets, fontsize=8)
        ax.set_ylabel("‖Δh‖ / ‖h0‖")
        ax.set_title(f"{lk} (α={a_hi})")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_reach.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "WO-003")
