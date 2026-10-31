"""WO-005 / WO-005a metrics.json + figures (numbers only; no interpretation).

WO-005a adds cluster-bootstrap and Wilson CIs; point estimators are unchanged.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.prompts_polarity import PAIR_IDS  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 0
CI95 = (2.5, 97.5)
STEERED_SETS = ["F", "I", "P0F", "P0I", "S3", "S3T", "S4"]
COMPRESS_SETS = ["F", "I", "P0F", "P0I", "S3", "S3T"]
CLASSES = ["concept", "random"]
ALPHAS = [2, 4]
PAIR_SETS = {"F", "I"}
TRIAL_SETS = {"P0F", "P0I"}
PROMPT_SETS = {"S3", "S3T", "S4"}


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
    if boots is None or (isinstance(boots, np.ndarray) and boots.size == 0):
        return {"m": point, "lo95": None, "hi95": None}
    lo95, hi95 = pct(boots, CI95)
    return {"m": point, "lo95": lo95, "hi95": hi95}


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        return None, None
    phat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (phat + z2 / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1.0 - phat) / n + z2 / (4.0 * n * n))
    return centre - half, centre + half


def wording_valid(
    *,
    base_ld: float,
    c_inject_mean_ld: float,
    c_noinject_mean_ld: float,
    polarity: str,
    abs_min: float = 3.0,
) -> bool:
    """Validity rule for one wording (F or I member of a pair)."""
    if polarity == "F":
        a = base_ld > abs_min
        b = c_inject_mean_ld < -abs_min
        c = c_noinject_mean_ld > abs_min
    elif polarity == "I":
        a = base_ld < -abs_min
        b = c_inject_mean_ld > abs_min
        c = c_noinject_mean_ld < -abs_min
    else:
        raise ValueError(polarity)
    return bool(a and b and c)


def _resample_idx(rng: np.random.Generator, n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    return rng.integers(0, n, size=n)


def cluster_boot_means(
    dF: Dict[str, np.ndarray],
    dI: Dict[str, np.ndarray],
    pairs: Sequence[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> np.ndarray:
    """Cluster bootstrap of D = mean_p(mean(dF_p) - mean(dI_p)) over pairs.

    Each replicate resamples pairs with replacement; within each drawn pair,
    resamples the flattened prompt×vector cells of Δclaim_F and Δclaim_I.
    Kept for tests / simple Δ arrays; full gen bootstrap uses base resampling.
    """
    rng = np.random.default_rng(seed)
    out = np.full(n_boot, np.nan)
    pairs = list(pairs)
    P = len(pairs)
    if P == 0:
        return out
    for b in range(n_boot):
        drawn = [pairs[i] for i in rng.integers(0, P, size=P)]
        ds = []
        for p in drawn:
            a = np.asarray(dF.get(p, []), dtype=float)
            c = np.asarray(dI.get(p, []), dtype=float)
            a = a[np.isfinite(a)]
            c = c[np.isfinite(c)]
            if a.size == 0 or c.size == 0:
                continue
            a2 = a[rng.integers(0, a.size, size=a.size)]
            c2 = c[rng.integers(0, c.size, size=c.size)]
            ds.append(float(a2.mean()) - float(c2.mean()))
        if ds:
            out[b] = float(np.mean(ds))
    return out


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


def vec_key(r: dict) -> Optional[str]:
    if r.get("concept") is not None:
        return str(r["concept"])
    if r.get("seed") is not None:
        return f"s{int(r['seed'])}"
    return None


def prompt_key(r: dict) -> Tuple[Any, ...]:
    return (
        r.get("set"),
        r.get("pair"),
        r.get("prompt_idx"),
        r.get("n"),
        r.get("polarity"),
    )


def _ld_stats(pairs: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return mean_base, mean_steer, mean_dld, ratio from (steer, base) pairs."""
    if not pairs:
        return float("nan"), float("nan"), float("nan"), float("nan")
    steers = np.array([s for s, _ in pairs], dtype=float)
    bases = np.array([b for _, b in pairs], dtype=float)
    mean_steer = nanmean(steers)
    mean_base = nanmean(bases)
    dlds = steers - bases
    mean_dld = nanmean(dlds)
    ratio = (
        mean_steer / mean_base
        if np.isfinite(mean_base) and mean_base != 0
        else float("nan")
    )
    return mean_base, mean_steer, mean_dld, ratio


def _cluster_key_for_set(s: str, r: dict) -> Any:
    if s in PAIR_SETS:
        return str(r.get("pair"))
    if s in TRIAL_SETS:
        return int(r.get("n", -1))
    return prompt_key(r)


# Nested: cluster -> prompt_id -> list[(steer_ld, base_ld)]
LdTree = Dict[Any, Dict[Any, List[Tuple[float, float]]]]


def _build_ld_tree(rows: List[dict], bases: Dict[Tuple[Any, ...], float], s: str) -> LdTree:
    tree: LdTree = defaultdict(lambda: defaultdict(list))
    for r in rows:
        pk = prompt_key(r)
        bl = bases.get(pk, float("nan"))
        ld = fl(r.get("ld"))
        ck = _cluster_key_for_set(s, r)
        # prompt id within cluster
        if s in PAIR_SETS:
            pid = (r.get("prompt_idx"), r.get("n"))
        elif s in TRIAL_SETS:
            pid = ("trial", int(r.get("n", -1)))
        else:
            pid = pk
        tree[ck][pid].append((ld, bl))
    return tree


def _boot_ld_tree(
    tree: LdTree,
    *,
    paired_clusters: bool,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap means: base, steer, dLD, ratio."""
    rng = np.random.default_rng(seed)
    b_base = np.full(n_boot, np.nan)
    b_steer = np.full(n_boot, np.nan)
    b_dld = np.full(n_boot, np.nan)
    b_ratio = np.full(n_boot, np.nan)
    clusters = list(tree.keys())
    if not clusters:
        return b_base, b_steer, b_dld, b_ratio
    for b in range(n_boot):
        sample: List[Tuple[float, float]] = []
        drawn_c = [clusters[i] for i in _resample_idx(rng, len(clusters))]
        for c in drawn_c:
            prompts = list(tree[c].keys())
            if not prompts:
                continue
            drawn_p = [prompts[i] for i in _resample_idx(rng, len(prompts))]
            for p in drawn_p:
                rows = tree[c][p]
                if not rows:
                    continue
                for j in _resample_idx(rng, len(rows)):
                    sample.append(rows[j])
        mb, ms, md, mr = _ld_stats(sample)
        b_base[b], b_steer[b], b_dld[b], b_ratio[b] = mb, ms, md, mr
    return b_base, b_steer, b_dld, b_ratio


def compute_validity(
    logit_rows: List[dict], c_rows: List[dict], abs_min: float
) -> Dict[str, Any]:
    base_n1: Dict[Tuple[str, str], float] = {}
    for r in logit_rows:
        if r.get("set") not in ("F", "I"):
            continue
        if r.get("condition") != "base":
            continue
        if int(r.get("n", -1)) != 1:
            continue
        base_n1[(str(r["pair"]), str(r["polarity"]))] = fl(r.get("ld"))

    c_acc: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in c_rows:
        mode = str(r.get("c_mode"))
        key = (str(r["pair"]), str(r["polarity"]), mode)
        c_acc[key].append(fl(r.get("ld")))

    per_wording: Dict[str, Any] = {}
    valid_pairs: List[str] = []
    for pid in PAIR_IDS:
        pair_ok = True
        for pol in ("F", "I"):
            bl = base_n1.get((pid, pol), float("nan"))
            inj = nanmean(np.array(c_acc.get((pid, pol, "inject"), []), dtype=float))
            noj = nanmean(np.array(c_acc.get((pid, pol, "noinject"), []), dtype=float))
            ok = wording_valid(
                base_ld=bl,
                c_inject_mean_ld=inj,
                c_noinject_mean_ld=noj,
                polarity=pol,
                abs_min=abs_min,
            )
            per_wording[f"{pid}_{pol}"] = {
                "valid": ok,
                "base_ld_n1": bl,
                "c_inject_mean_ld": inj,
                "c_noinject_mean_ld": noj,
            }
            pair_ok = pair_ok and ok
        per_wording[pid] = {"valid": pair_ok}
        if pair_ok:
            valid_pairs.append(pid)
    return {"per_wording": per_wording, "valid_pairs": valid_pairs, "all_pairs": list(PAIR_IDS)}


def logit_block(rows: List[dict], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Aggregate logit metrics keyed by (set, alpha, class, cond) with cluster CIs."""
    bases: Dict[Tuple[Any, ...], float] = {}
    cells: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for r in rows:
        if r.get("set") == "C":
            continue
        pk = prompt_key(r)
        cond = r.get("condition")
        if cond == "base":
            bases[pk] = fl(r.get("ld"))
            continue
        if cond in ("alpha0",):
            continue
        cls = r.get("row_kind") or r.get("class")
        alpha = r.get("alpha")
        cells[(r.get("set"), cls, alpha, cond)].append(r)

    out: Dict[str, Any] = {}
    for (s, cls, alpha, cond), lst in cells.items():
        flat = [(fl(r.get("ld")), bases.get(prompt_key(r), float("nan"))) for r in lst]
        mean_base, mean_ld, mean_dld, ratio = _ld_stats(flat)

        top_s = {"Yes": 0, "No": 0, "other": 0, "n": 0}
        for r in lst:
            t1 = r.get("top1_class") or "other"
            top_s[t1] = top_s.get(t1, 0) + 1
            top_s["n"] += 1

        tree = _build_ld_tree(lst, bases, str(s))
        paired = str(s) in PAIR_SETS or str(s) in TRIAL_SETS
        # S3/S3T/S4: tree keys are prompt_keys; resampling clusters == resampling prompts
        b_base, b_steer, b_dld, b_ratio = _boot_ld_tree(
            tree, paired_clusters=paired, n_boot=n_boot, seed=seed
        )

        key = f"{s}|{cls}|a{alpha}|{cond}"
        out[key] = {
            "set": s,
            "class": cls,
            "alpha": alpha,
            "condition": cond,
            "LD_base": ci_block(mean_base, b_base),
            "LD_steer": ci_block(mean_ld, b_steer),
            "dLD": ci_block(mean_dld, b_dld),
            "ratio_means": ci_block(ratio, b_ratio),
            "top1_steer": {
                k: (top_s[k] / top_s["n"] if top_s["n"] else None)
                for k in ("Yes", "No", "other")
            },
            "n": len(lst),
        }

    base_top: Dict[str, Dict[str, float]] = {}
    for r in rows:
        if r.get("condition") != "base" or r.get("set") == "C":
            continue
        s = r["set"]
        base_top.setdefault(s, {"Yes": 0, "No": 0, "other": 0, "n": 0})
        t1 = r.get("top1_class") or "other"
        base_top[s][t1] = base_top[s].get(t1, 0) + 1
        base_top[s]["n"] += 1
    for s, d in base_top.items():
        out[f"{s}|base_top1"] = {
            k: (d[k] / d["n"] if d["n"] else None) for k in ("Yes", "No", "other")
        }
    return out


def compression_block(
    rows: List[dict], *, n_boot: int = N_BOOT, seed: int = BOOT_SEED
) -> Dict[str, Any]:
    bases: Dict[Tuple[Any, ...], float] = {}
    for r in rows:
        if r.get("condition") == "base" and r.get("set") != "C":
            bases[prompt_key(r)] = fl(r.get("ld"))

    def _steer_rows(s: str, alpha: int, cls: str) -> List[dict]:
        return [
            r
            for r in rows
            if r.get("set") == s
            and r.get("condition") == "steer"
            and r.get("row_kind") == cls
            and int(r.get("alpha", -1)) == alpha
        ]

    fit: Dict[str, Any] = {}
    resid: Dict[str, Any] = {}
    for alpha in ALPHAS:
        for cls in CLASSES:
            s4_rows = _steer_rows("S4", alpha, cls)
            xs, ys = [], []
            for r in s4_rows:
                bl = bases.get(prompt_key(r), float("nan"))
                if np.isfinite(bl):
                    xs.append(bl)
                    ys.append(fl(r.get("ld")))
            slope, intercept = ols_fit(np.array(xs), np.array(ys))
            fit[f"a{alpha}_{cls}"] = {"slope": slope, "intercept": intercept, "n": len(xs)}

            s4_tree = _build_ld_tree(s4_rows, bases, "S4")

            for s in COMPRESS_SETS:
                tgt_rows = _steer_rows(s, alpha, cls)
                obs, pred = [], []
                for r in tgt_rows:
                    bl = bases.get(prompt_key(r), float("nan"))
                    if not np.isfinite(bl) or not np.isfinite(slope):
                        continue
                    obs.append(fl(r.get("ld")))
                    pred.append(slope * bl + intercept)
                if obs:
                    residual = nanmean(np.array(obs)) - nanmean(np.array(pred))
                    yesward = -residual
                else:
                    yesward = float("nan")

                tgt_tree = _build_ld_tree(tgt_rows, bases, s)
                paired = s in PAIR_SETS or s in TRIAL_SETS
                rng = np.random.default_rng(seed)
                boots = np.full(n_boot, np.nan)
                s4_clusters = list(s4_tree.keys())
                tgt_clusters = list(tgt_tree.keys())
                for b in range(n_boot):
                    # resample S4 and refit
                    s4_sample: List[Tuple[float, float]] = []
                    if s4_clusters:
                        for c in [s4_clusters[i] for i in _resample_idx(rng, len(s4_clusters))]:
                            prompts = list(s4_tree[c].keys())
                            for p in [prompts[i] for i in _resample_idx(rng, len(prompts))]:
                                rows_p = s4_tree[c][p]
                                for j in _resample_idx(rng, len(rows_p)):
                                    s4_sample.append(rows_p[j])
                    if len(s4_sample) < 2:
                        continue
                    bx = np.array([b0 for _, b0 in s4_sample], dtype=float)
                    by = np.array([st for st, _ in s4_sample], dtype=float)
                    sl, ic = ols_fit(bx, by)
                    if not np.isfinite(sl):
                        continue
                    # resample target set
                    tgt_sample: List[Tuple[float, float]] = []
                    if not tgt_clusters:
                        continue
                    for c in [tgt_clusters[i] for i in _resample_idx(rng, len(tgt_clusters))]:
                        prompts = list(tgt_tree[c].keys())
                        for p in [prompts[i] for i in _resample_idx(rng, len(prompts))]:
                            rows_p = tgt_tree[c][p]
                            for j in _resample_idx(rng, len(rows_p)):
                                tgt_sample.append(rows_p[j])
                    if not tgt_sample:
                        continue
                    o = np.array([st for st, _ in tgt_sample], dtype=float)
                    pv = np.array([sl * b0 + ic for _, b0 in tgt_sample], dtype=float)
                    boots[b] = -(nanmean(o) - nanmean(pv))

                resid[f"{s}|a{alpha}|{cls}"] = {
                    "yesward_resid": ci_block(yesward, boots),
                    "n": len(obs),
                }
    return {"fit": fit, "yesward_resid": resid}


# Gen structures: pair -> prompt_idx -> {"base": [claims], "steer": [claims]}
GenPrompt = Dict[str, List[float]]
GenPair = Dict[int, GenPrompt]


def _cond_key(r: dict) -> str:
    c = r.get("condition")
    if c == "base":
        return "base"
    if c == "concept":
        return f"concept_a{int(r['alpha'])}"
    if c == "random":
        return f"random_a{int(r['alpha'])}"
    return str(c)


def _build_gen_pair_maps(
    cells: Dict[Tuple[str, str, str], List[dict]], s: str, ck: str
) -> Dict[str, GenPair]:
    out: Dict[str, GenPair] = {}
    for pid in PAIR_IDS:
        base_lst = cells.get((s, pid, "base"), [])
        ste_lst = cells.get((s, pid, ck), [])
        by_p: GenPair = {}
        for r in base_lst:
            if r.get("claims") is None:
                continue
            p = int(r["prompt_idx"])
            by_p.setdefault(p, {"base": [], "steer": []})
            by_p[p]["base"].append(float(r["claims"]))
        for r in ste_lst:
            if r.get("claims") is None:
                continue
            p = int(r["prompt_idx"])
            by_p.setdefault(p, {"base": [], "steer": []})
            by_p[p]["steer"].append(float(r["claims"]))
        out[pid] = by_p
    return out


def _point_deltas_from_map(gmap: GenPair) -> np.ndarray:
    """Point estimator matching WO-005: for each steer row, claim - base_mean(prompt)."""
    deltas = []
    for p, d in gmap.items():
        if not d["base"] or not d["steer"]:
            continue
        bm = float(np.mean(d["base"]))
        for c in d["steer"]:
            deltas.append(c - bm)
    return np.array(deltas, dtype=float)


def _boot_pair_delta(gmap: GenPair, rng: np.random.Generator) -> float:
    """Within-pair resample prompts → base samples → vectors; return mean Δclaim."""
    prompts = [p for p, d in gmap.items() if d["base"] and d["steer"]]
    if not prompts:
        return float("nan")
    drawn_p = [prompts[i] for i in _resample_idx(rng, len(prompts))]
    deltas = []
    for p in drawn_p:
        base = gmap[p]["base"]
        steer = gmap[p]["steer"]
        b_s = [base[j] for j in _resample_idx(rng, len(base))]
        s_s = [steer[j] for j in _resample_idx(rng, len(steer))]
        bm = float(np.mean(b_s))
        for c in s_s:
            deltas.append(c - bm)
    return float(np.mean(deltas)) if deltas else float("nan")


def _boot_D_and_deltas(
    dF_maps: Dict[str, GenPair],
    dI_maps: Dict[str, GenPair],
    pairs: Sequence[str],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Joint cluster bootstrap: returns boots of D, mean_dF, mean_dI."""
    rng = np.random.default_rng(seed)
    pairs = list(pairs)
    P = len(pairs)
    bD = np.full(n_boot, np.nan)
    bF = np.full(n_boot, np.nan)
    bI = np.full(n_boot, np.nan)
    if P == 0:
        return bD, bF, bI
    for b in range(n_boot):
        drawn = [pairs[i] for i in rng.integers(0, P, size=P)]
        dfs, dis = [], []
        for p in drawn:
            df = _boot_pair_delta(dF_maps.get(p, {}), rng)
            di = _boot_pair_delta(dI_maps.get(p, {}), rng)
            if np.isfinite(df) and np.isfinite(di):
                dfs.append(df)
                dis.append(di)
        if dfs:
            bF[b] = float(np.mean(dfs))
            bI[b] = float(np.mean(dis))
            bD[b] = bF[b] - bI[b]
    return bD, bF, bI


def gen_metrics(
    gens: List[dict],
    validity: Dict[str, Any],
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any], List[dict]]:
    valid_pairs = set(validity.get("valid_pairs") or [])

    cells: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in gens:
        cells[(r["set"], str(r.get("pair")), _cond_key(r))].append(r)

    def rates(lst: List[dict]) -> Dict[str, Any]:
        if not lst:
            return {
                "claim": None,
                "claim_nondegen": None,
                "degenerate": None,
                "parse_fail": None,
                "n": 0,
                "wilson_unclustered": None,
            }
        claims, nondeg_claims, degen, pf = [], [], [], []
        for r in lst:
            cl = r.get("claims")
            if cl is None:
                pf.append(1)
            else:
                claims.append(int(cl))
                pf.append(0)
                if not r.get("degenerate"):
                    nondeg_claims.append(int(cl))
            degen.append(1 if r.get("degenerate") else 0)
        k = int(sum(claims)) if claims else 0
        n_c = len(claims)
        lo, hi = wilson_ci(k, n_c) if n_c else (None, None)
        claim_m = nanmean(np.array(claims, dtype=float)) if claims else None
        return {
            "claim": claim_m,
            "claim_nondegen": nanmean(np.array(nondeg_claims, dtype=float))
            if nondeg_claims
            else None,
            "degenerate": nanmean(np.array(degen, dtype=float)),
            "parse_fail": nanmean(np.array(pf, dtype=float)),
            "n": len(lst),
            "wilson_unclustered": {
                "k": k,
                "n": n_c,
                "rate": claim_m,
                "lo95": lo,
                "hi95": hi,
            }
            if n_c
            else None,
        }

    gen_out: Dict[str, Any] = {}
    csv_rows: List[dict] = []
    for (s, pair, ck), lst in cells.items():
        rt = rates(lst)
        gen_out[f"{s}|{pair}|{ck}"] = rt
        w = rt.get("wilson_unclustered") or {}
        csv_rows.append(
            {
                "set": s,
                "pair": pair,
                "cond": ck,
                "k": w.get("k", ""),
                "n": w.get("n", rt.get("n", 0)),
                "rate": w.get("rate", rt.get("claim")),
                "wilson_lo95": w.get("lo95"),
                "wilson_hi95": w.get("hi95"),
            }
        )

    delta_claim: Dict[str, Any] = {}
    D_block: Dict[str, Any] = {}

    for ck in ("concept_a2", "concept_a4", "random_a4"):
        dF_maps = _build_gen_pair_maps(cells, "F", ck)
        dI_maps = _build_gen_pair_maps(cells, "I", ck)
        dF_raw = {p: _point_deltas_from_map(dF_maps[p]) for p in PAIR_IDS}
        dI_raw = {p: _point_deltas_from_map(dI_maps[p]) for p in PAIR_IDS}

        per_pair: Dict[str, Any] = {}
        for pid in PAIR_IDS:
            # Within-pair CIs
            rng = np.random.default_rng(seed)
            bF = np.full(n_boot, np.nan)
            bI = np.full(n_boot, np.nan)
            bD = np.full(n_boot, np.nan)
            for b in range(n_boot):
                df = _boot_pair_delta(dF_maps[pid], rng)
                di = _boot_pair_delta(dI_maps[pid], rng)
                bF[b], bI[b] = df, di
                if np.isfinite(df) and np.isfinite(di):
                    bD[b] = df - di
            mF = nanmean(dF_raw[pid])
            mI = nanmean(dI_raw[pid])
            mD = (
                mF - mI
                if dF_raw[pid].size and dI_raw[pid].size
                else float("nan")
            )
            per_pair[pid] = {
                "delta_claim_F": ci_block(mF, bF),
                "delta_claim_I": ci_block(mI, bI),
                "D": ci_block(mD, bD),
                "n_F": int(dF_raw[pid].size),
                "n_I": int(dI_raw[pid].size),
            }
        delta_claim[ck] = per_pair

        point_D_all = nanmean(
            np.array([nanmean(dF_raw[p]) - nanmean(dI_raw[p]) for p in PAIR_IDS])
        )
        point_F_all = nanmean(np.array([nanmean(dF_raw[p]) for p in PAIR_IDS]))
        point_I_all = nanmean(np.array([nanmean(dI_raw[p]) for p in PAIR_IDS]))
        boots_D, boots_F, boots_I = _boot_D_and_deltas(
            dF_maps, dI_maps, PAIR_IDS, n_boot=n_boot, seed=seed
        )

        dF_v = {p: dF_raw[p] for p in PAIR_IDS if p in valid_pairs}
        dI_v = {p: dI_raw[p] for p in PAIR_IDS if p in valid_pairs}
        maps_Fv = {p: dF_maps[p] for p in dF_v}
        maps_Iv = {p: dI_maps[p] for p in dI_v}
        point_D_v = (
            nanmean(np.array([nanmean(dF_v[p]) - nanmean(dI_v[p]) for p in dF_v]))
            if dF_v
            else float("nan")
        )
        boots_Dv, _, _ = (
            _boot_D_and_deltas(maps_Fv, maps_Iv, list(dF_v.keys()), n_boot=n_boot, seed=seed)
            if dF_v
            else (np.array([]), np.array([]), np.array([]))
        )

        D_block[ck] = {
            "pooled_all": ci_block(point_D_all, boots_D),
            "pooled_valid": ci_block(point_D_v, boots_Dv if boots_Dv.size else None),
            "per_pair": {p: per_pair[p]["D"] for p in PAIR_IDS},
            "delta_claim_F_pooled_all": ci_block(point_F_all, boots_F),
            "delta_claim_I_pooled_all": ci_block(point_I_all, boots_I),
        }

    # LOPO on concept_a4 with CIs
    ck = "concept_a4"
    dF_maps = _build_gen_pair_maps(cells, "F", ck)
    dI_maps = _build_gen_pair_maps(cells, "I", ck)
    dF_raw = {p: _point_deltas_from_map(dF_maps[p]) for p in PAIR_IDS}
    dI_raw = {p: _point_deltas_from_map(dI_maps[p]) for p in PAIR_IDS}
    lopo: Dict[str, Any] = {}
    for leave in PAIR_IDS:
        keep = [p for p in PAIR_IDS if p != leave]
        dF_m = nanmean(np.array([nanmean(dF_raw[p]) for p in keep]))
        dI_m = nanmean(np.array([nanmean(dI_raw[p]) for p in keep]))
        boots_D, boots_F, boots_I = _boot_D_and_deltas(
            {p: dF_maps[p] for p in keep},
            {p: dI_maps[p] for p in keep},
            keep,
            n_boot=n_boot,
            seed=seed,
        )
        lopo[leave] = {
            "delta_claim_F": ci_block(dF_m, boots_F),
            "delta_claim_I": ci_block(dI_m, boots_I),
            "D": ci_block(dF_m - dI_m, boots_D),
        }

    return gen_out, delta_claim, D_block, {"lopo_concept_a4": lopo}, csv_rows


def contingency_table(gens: List[dict], logit_rows: List[dict]) -> Dict[str, Any]:
    """Join steered gens to logit top1 for same (set, prompt_idx, vector)."""
    top1: Dict[Tuple[Any, ...], str] = {}
    for r in logit_rows:
        if r.get("set") not in ("F", "I"):
            continue
        if r.get("condition") not in ("steer", "late"):
            continue
        if int(r.get("n", -1)) != 1:
            continue
        vk = vec_key(r)
        if vk is None:
            continue
        key = (r["set"], int(r["prompt_idx"]), r.get("row_kind"), r.get("alpha"), vk)
        top1[key] = r.get("top1_class") or "other"

    out: Dict[str, Any] = {}
    for s in ("F", "I"):
        tab: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for r in gens:
            if r.get("set") != s or r.get("condition") == "base":
                continue
            vk = vec_key(r)
            key2 = (
                s,
                int(r["prompt_idx"]),
                "concept" if r.get("condition") == "concept" else "random",
                r.get("alpha"),
                vk,
            )
            t1 = top1.get(key2) or "other"
            cl = r.get("claims")
            clabel = "null" if cl is None else ("YES" if int(cl) == 1 else "NO")
            tab[t1][clabel] += 1
        out[s] = {t: dict(v) for t, v in tab.items()}
    return out


def make_figures(results_dir: Path, metrics: Dict[str, Any], c_rows: List[dict]) -> None:
    D = metrics.get("D", {})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, pool in zip(axes, ("pooled_valid", "pooled_all")):
        xs, ys, ylo, yhi = [], [], [], []
        for i, ck in enumerate(("concept_a2", "concept_a4", "random_a4")):
            blk = (D.get(ck) or {}).get(pool) or {}
            xs.append(i)
            ys.append(blk.get("m") if blk.get("m") is not None else float("nan"))
            ylo.append(blk.get("lo95") if blk.get("lo95") is not None else float("nan"))
            yhi.append(blk.get("hi95") if blk.get("hi95") is not None else float("nan"))
        ys, ylo, yhi = np.array(ys, float), np.array(ylo, float), np.array(yhi, float)
        yerr_lo = np.where(np.isfinite(ylo), np.clip(ys - ylo, 0, None), np.nan)
        yerr_hi = np.where(np.isfinite(yhi), np.clip(yhi - ys, 0, None), np.nan)
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o", capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels(["cα2", "cα4", "rα4"])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"D ({pool})")
        ax.set_ylabel("D = Δclaim_F − Δclaim_I")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_claims.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([-40, 40], [-40, 40], "k--", lw=0.7, label="identity")
    fit = metrics.get("compression", {}).get("fit", {})
    for key, f in fit.items():
        if f.get("slope") is None or not np.isfinite(f.get("slope", float("nan"))):
            continue
        xs = np.array([-40.0, 40.0])
        ax.plot(xs, f["slope"] * xs + f["intercept"], lw=1.0, label=key)
    ax.set_xlabel("LD_base")
    ax.set_ylabel("LD_steer")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(results_dir / "fig_ld.png", dpi=150)
    plt.close(fig)

    resid = metrics.get("compression", {}).get("yesward_resid", {})
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = [k for k in resid if "|a4|concept" in k]
    ys = np.array(
        [
            resid[k]["yesward_resid"]["m"]
            if resid[k]["yesward_resid"]["m"] is not None
            else float("nan")
            for k in keys
        ],
        dtype=float,
    )
    ylo = np.array(
        [
            resid[k]["yesward_resid"].get("lo95")
            if resid[k]["yesward_resid"].get("lo95") is not None
            else float("nan")
            for k in keys
        ],
        dtype=float,
    )
    yhi = np.array(
        [
            resid[k]["yesward_resid"].get("hi95")
            if resid[k]["yesward_resid"].get("hi95") is not None
            else float("nan")
            for k in keys
        ],
        dtype=float,
    )
    x = np.arange(len(keys))
    ax.bar(x, ys, yerr=[np.clip(ys - ylo, 0, None), np.clip(yhi - ys, 0, None)], capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels([k.split("|")[0] for k in keys], rotation=45, ha="right")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Yes-ward residual (α=4 concept)")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_residual.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    acc: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in c_rows:
        acc[(str(r["pair"]), str(r["polarity"]), str(r["c_mode"]))].append(fl(r.get("ld")))
    labels, vals = [], []
    for pid in PAIR_IDS:
        for pol in ("F", "I"):
            for mode in ("inject", "noinject"):
                labels.append(f"{pid}{pol[0]}/{mode[:3]}")
                vals.append(nanmean(np.array(acc.get((pid, pol, mode), []), dtype=float)))
    ax.bar(range(len(labels)), vals)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.axhline(3, color="gray", ls="--", lw=0.7)
    ax.axhline(-3, color="gray", ls="--", lw=0.7)
    ax.set_ylabel("set-C LD")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_comprehension.png", dpi=150)
    plt.close(fig)


def _extract_points(obj: Any, path: str = "") -> Dict[str, float]:
    """Flatten comparable point estimates from metrics (noci or new)."""
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        if set(obj.keys()) >= {"m"} and all(
            k in ("m", "lo95", "hi95") or k in obj for k in ("m",)
        ):
            # CI block or legacy {"m": ...}
            if "lo95" in obj or "hi95" in obj or list(obj.keys()) == ["m"]:
                if obj["m"] is not None and isinstance(obj["m"], (int, float)):
                    out[path + ".m" if path else "m"] = float(obj["m"])
                # still walk? no
                return out
        for k, v in obj.items():
            if k in ("wilson_unclustered", "lo95", "hi95", "k"):
                continue
            p = f"{path}.{k}" if path else str(k)
            out.update(_extract_points(v, p))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        if obj is not None and math.isfinite(float(obj)):
            out[path] = float(obj)
    elif isinstance(obj, list):
        pass
    return out


def assert_points_match(noci: dict, new: dict, tol: float = 1e-9) -> None:
    """SANITY1: every noci point estimate equals new counterpart within tol."""
    # Special-case LOPO bare floats vs CI blocks
    a = _extract_points(noci)
    b = _extract_points(new)
    # Map old lopo.P1.D -> lopo.P1.D.m style: extract_points on new already uses .m
    # For noci lopo values are bare floats at lopo.P1.D; new at lopo.P1.D.m
    mismatches = []
    for k, va in a.items():
        # try direct
        candidates = [k]
        if not k.endswith(".m"):
            candidates.append(k + ".m")
        found = None
        for c in candidates:
            if c in b:
                found = c
                break
        if found is None:
            # optional keys only in new (wilson) — skip missing in new for keys we don't require
            mismatches.append((k, va, "MISSING"))
            continue
        vb = b[found]
        if abs(va - vb) > tol:
            mismatches.append((k, va, vb))
    # Filter: ignore keys that are only structural lists etc.
    real = [(k, va, vb) for k, va, vb in mismatches if vb != "MISSING" or k.startswith(
        ("logit.", "compression.", "delta_claim.", "D.", "lopo.", "gen.")
    ) and not k.endswith(".valid") and "wilson" not in k]
    # Missing in new is OK only if noci didn't have CI fields; require present for known point paths
    hard = []
    for k, va, vb in mismatches:
        if vb == "MISSING":
            # try .m
            if k + ".m" in b:
                if abs(va - b[k + ".m"]) > tol:
                    hard.append((k, va, b[k + ".m"]))
            elif any(
                k.startswith(p)
                for p in ("logit.", "compression.yesward", "delta_claim.", "D.", "lopo.", "gen.")
            ) and not any(
                x in k
                for x in (
                    "top1",
                    "valid",
                    "n_F",
                    "n_I",
                    "n.",
                    ".n",
                    "wilson",
                    "per_wording",
                    "sanity",
                    "judge",
                    "git",
                    "model",
                    "batch",
                    "all_pairs",
                    "valid_pairs",
                    "contingency",
                    "fit.",
                )
            ):
                # gen claim rates etc should exist
                if ".claim" in k or ".degenerate" in k or ".parse_fail" in k or k.endswith(".m"):
                    hard.append((k, va, "MISSING"))
        else:
            hard.append((k, va, vb))
    # Simpler strict check on known sections:
    hard2 = []
    for k, va in a.items():
        if not any(
            k.startswith(p)
            for p in (
                "logit.",
                "compression.yesward_resid",
                "delta_claim.",
                "D.",
                "lopo.",
                "gen.",
            )
        ):
            continue
        if any(
            x in k
            for x in (
                "top1",
                "wilson",
                ".n",
                "n_F",
                "n_I",
                "valid",
                "fit.",
            )
        ):
            continue
        vb = b.get(k)
        if vb is None:
            vb = b.get(k + ".m")
        if vb is None:
            hard2.append((k, va, None))
        elif abs(va - vb) > tol:
            hard2.append((k, va, vb))
    if hard2:
        preview = hard2[:20]
        raise AssertionError(
            f"SANITY1 point mismatch n={len(hard2)}; sample={preview}"
        )


def assert_ci_ordered(obj: Any, path: str = "") -> None:
    """SANITY2: every CI satisfies lo95 ≤ m ≤ hi95."""
    if isinstance(obj, dict):
        if "m" in obj and ("lo95" in obj or "hi95" in obj):
            m, lo, hi = obj.get("m"), obj.get("lo95"), obj.get("hi95")
            if m is not None and lo is not None and hi is not None:
                if not (lo - 1e-12 <= m <= hi + 1e-12):
                    raise AssertionError(f"SANITY2 CI order fail at {path}: {obj}")
        for k, v in obj.items():
            assert_ci_ordered(v, f"{path}.{k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            assert_ci_ordered(v, f"{path}[{i}]")


def write_claims_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["set", "pair", "cond", "k", "n", "rate", "wilson_lo95", "wilson_hi95"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["set"], x["pair"], x["cond"])):
            w.writerow(r)


def main(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "wo_005.yaml").read_text())
    abs_min = float(cfg.get("validity", {}).get("base_ld_abs_min", 3.0))
    n_boot = int(cfg.get("bootstrap", {}).get("n_resamples", N_BOOT))
    seed = int(cfg.get("bootstrap", {}).get("seed", BOOT_SEED))

    summary = {}
    if (results_dir / "run_summary.json").exists():
        summary = json.loads((results_dir / "run_summary.json").read_text())
    judge_summary = {}
    if (results_dir / "judge_summary.json").exists():
        judge_summary = json.loads((results_dir / "judge_summary.json").read_text())

    noci_path = results_dir / "metrics_noci_aba5637.json"
    if not noci_path.exists() and (results_dir / "metrics.json").exists():
        # freeze current metrics as no-CI baseline once
        noci_path.write_text((results_dir / "metrics.json").read_text())

    logit_rows = read_jsonl(raw / "logit_rows.jsonl")
    gens = read_jsonl(raw / "gens.jsonl")
    c_rows = [r for r in logit_rows if r.get("set") == "C"]
    steered = [r for r in logit_rows if r.get("set") != "C"]

    validity = compute_validity(steered, c_rows, abs_min)
    logit = logit_block(steered, n_boot=n_boot, seed=seed)
    compression = compression_block(steered, n_boot=n_boot, seed=seed)
    gen_out, delta_claim, D_block, extra, csv_rows = gen_metrics(
        gens, validity, n_boot=n_boot, seed=seed
    )
    contingency = contingency_table(gens, steered)

    metrics = {
        "wo_id": "WO-005",
        "wo_005a": True,
        "validity": validity,
        "logit": logit,
        "compression": compression,
        "gen": gen_out,
        "delta_claim": delta_claim,
        "D": D_block,
        "contingency": contingency,
        "lopo": extra.get("lopo_concept_a4"),
        "sanity": {
            "n_logit_rows": len(logit_rows),
            "n_steered": len(steered),
            "n_C": len(c_rows),
            "n_gens": len(gens),
            "run_status": summary.get("status"),
            "replication_max_abs_ld": summary.get("replication_max_abs_ld"),
            "projection": summary.get("projection"),
            "n_boot": n_boot,
            "boot_seed": seed,
        },
        "judge": {
            "calls": judge_summary.get("n_calls"),
            "usd": judge_summary.get("est_usd"),
            "parse_failures": judge_summary.get("parse_failures"),
        },
        "git_hash": summary.get("git_hash"),
        "model_revision": summary.get("model_revision"),
        "batch_size": summary.get("batch_size") or cfg.get("batch_size"),
    }
    metrics = clean(metrics)

    if noci_path.exists():
        noci = json.loads(noci_path.read_text())
        assert_points_match(noci, metrics)
        print("SANITY1 point estimates match metrics_noci_aba5637.json")
    assert_ci_ordered(metrics)
    print("SANITY2 all CIs satisfy lo95 ≤ m ≤ hi95")

    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_claims_csv(raw / "claims_by_pair.csv", csv_rows)
    make_figures(results_dir, metrics, c_rows)
    print(f"Wrote {results_dir / 'metrics.json'}, figures, and {raw / 'claims_by_pair.csv'}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "WO-005")
