"""WO-005 metrics.json + figures (numbers only; no interpretation)."""

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
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.prompts_polarity import PAIR_IDS  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 0
CI95 = (2.5, 97.5)
STEERED_SETS = ["F", "I", "P0F", "P0I", "S3", "S3T", "S4"]
COMPRESS_SETS = ["F", "I", "P0F", "P0I", "S3", "S3T"]
CLASSES = ["concept", "random"]
ALPHAS = [2, 4]


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


def compute_validity(
    logit_rows: List[dict], c_rows: List[dict], abs_min: float
) -> Dict[str, Any]:
    # base LD at n=1 for F/I wordings keyed by (pair, polarity)
    base_n1: Dict[Tuple[str, str], float] = {}
    for r in logit_rows:
        if r.get("set") not in ("F", "I"):
            continue
        if r.get("condition") != "base":
            continue
        if int(r.get("n", -1)) != 1:
            continue
        base_n1[(str(r["pair"]), str(r["polarity"]))] = fl(r.get("ld"))

    # C means per (pair, polarity, mode)
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


def logit_block(rows: List[dict]) -> Dict[str, Any]:
    """Aggregate logit metrics keyed by (set, alpha, class, cond)."""
    # Index: base by prompt_key; steer by prompt_key + class + alpha + cond
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
        if cls not in CLASSES and cond != "late":
            # late is concept α=4
            pass
        lds, dlds = [], []
        top_b = {"Yes": 0, "No": 0, "other": 0, "n": 0}
        top_s = {"Yes": 0, "No": 0, "other": 0, "n": 0}
        for r in lst:
            pk = prompt_key(r)
            bl = bases.get(pk, float("nan"))
            ld = fl(r.get("ld"))
            lds.append(ld)
            if np.isfinite(bl):
                dlds.append(ld - bl)
            t1 = r.get("top1_class") or "other"
            top_s[t1] = top_s.get(t1, 0) + 1
            top_s["n"] += 1
        # base top-1 shares over unique prompts in this set
        seen = set()
        for r in lst:
            pk = prompt_key(r)
            if pk in seen:
                continue
            seen.add(pk)
            # find base row top1 from original rows via ld match — skip; filled below
        mean_ld = nanmean(np.array(lds))
        mean_dld = nanmean(np.array(dlds))
        mean_base = nanmean(np.array([bases.get(prompt_key(r), float("nan")) for r in lst]))
        ratio = mean_ld / mean_base if np.isfinite(mean_base) and mean_base != 0 else float("nan")
        key = f"{s}|{cls}|a{alpha}|{cond}"
        out[key] = {
            "set": s,
            "class": cls,
            "alpha": alpha,
            "condition": cond,
            "LD_base": {"m": mean_base},
            "LD_steer": {"m": mean_ld},
            "dLD": {"m": mean_dld},
            "ratio_means": {"m": ratio},
            "top1_steer": {
                k: (top_s[k] / top_s["n"] if top_s["n"] else None)
                for k in ("Yes", "No", "other")
            },
            "n": len(lst),
        }
    # base top1 per set
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


def compression_block(rows: List[dict]) -> Dict[str, Any]:
    bases: Dict[Tuple[Any, ...], float] = {}
    for r in rows:
        if r.get("condition") == "base" and r.get("set") != "C":
            bases[prompt_key(r)] = fl(r.get("ld"))

    fit: Dict[str, Any] = {}
    resid: Dict[str, Any] = {}
    for alpha in ALPHAS:
        for cls in CLASSES:
            # S4 fit
            xs, ys = [], []
            for r in rows:
                if (
                    r.get("set") != "S4"
                    or r.get("condition") != "steer"
                    or r.get("row_kind") != cls
                    or int(r.get("alpha", -1)) != alpha
                ):
                    continue
                bl = bases.get(prompt_key(r), float("nan"))
                if np.isfinite(bl):
                    xs.append(bl)
                    ys.append(fl(r.get("ld")))
            slope, intercept = ols_fit(np.array(xs), np.array(ys))
            fit[f"a{alpha}_{cls}"] = {"slope": slope, "intercept": intercept, "n": len(xs)}
            for s in COMPRESS_SETS:
                obs, pred = [], []
                for r in rows:
                    if (
                        r.get("set") != s
                        or r.get("condition") != "steer"
                        or r.get("row_kind") != cls
                        or int(r.get("alpha", -1)) != alpha
                    ):
                        continue
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
                resid[f"{s}|a{alpha}|{cls}"] = {
                    "yesward_resid": {"m": yesward},
                    "n": len(obs),
                }
    return {"fit": fit, "yesward_resid": resid}


def gen_metrics(
    gens: List[dict], validity: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    valid_pairs = set(validity.get("valid_pairs") or [])
    # group by (set, pair, cond_key)
    # cond_key: base | concept_a2 | concept_a4 | random_a4
    def cond_key(r: dict) -> str:
        c = r.get("condition")
        if c == "base":
            return "base"
        if c == "concept":
            return f"concept_a{int(r['alpha'])}"
        if c == "random":
            return f"random_a{int(r['alpha'])}"
        return str(c)

    cells: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in gens:
        cells[(r["set"], str(r.get("pair")), cond_key(r))].append(r)

    def rates(lst: List[dict]) -> Dict[str, Any]:
        if not lst:
            return {"claim": None, "claim_nondegen": None, "degenerate": None, "parse_fail": None, "n": 0}
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
        return {
            "claim": nanmean(np.array(claims, dtype=float)) if claims else None,
            "claim_nondegen": nanmean(np.array(nondeg_claims, dtype=float)) if nondeg_claims else None,
            "degenerate": nanmean(np.array(degen, dtype=float)),
            "parse_fail": nanmean(np.array(pf, dtype=float)),
            "n": len(lst),
        }

    gen_out: Dict[str, Any] = {}
    for (s, pair, ck), lst in cells.items():
        gen_out[f"{s}|{pair}|{ck}"] = rates(lst)

    # Δclaim per pair × set, then D
    def claim_by_prompt_vec(lst: List[dict], is_base: bool) -> Dict[Tuple[Any, ...], List[float]]:
        acc: Dict[Tuple[Any, ...], List[float]] = defaultdict(list)
        for r in lst:
            if r.get("claims") is None:
                continue
            if is_base:
                key = (r.get("prompt_idx"), r.get("sample_k"))
            else:
                key = (r.get("prompt_idx"), vec_key(r))
            acc[key].append(float(r["claims"]))
        return acc

    def _deltas_for(s: str, pid: str, ck: str) -> np.ndarray:
        base_lst = cells.get((s, pid, "base"), [])
        ste_lst = cells.get((s, pid, ck), [])
        base_by_p: Dict[int, List[float]] = defaultdict(list)
        for r in base_lst:
            if r.get("claims") is not None:
                base_by_p[int(r["prompt_idx"])].append(float(r["claims"]))
        base_mean = {p: float(np.mean(v)) for p, v in base_by_p.items() if v}
        deltas = []
        for r in ste_lst:
            if r.get("claims") is None:
                continue
            p = int(r["prompt_idx"])
            if p not in base_mean:
                continue
            deltas.append(float(r["claims"]) - base_mean[p])
        return np.array(deltas, dtype=float)

    delta_claim: Dict[str, Any] = {}
    D_block: Dict[str, Any] = {}

    for ck in ("concept_a2", "concept_a4", "random_a4"):
        dF_raw: Dict[str, np.ndarray] = {}
        dI_raw: Dict[str, np.ndarray] = {}
        per_pair: Dict[str, Any] = {}
        for pid in PAIR_IDS:
            dF_raw[pid] = _deltas_for("F", pid, ck)
            dI_raw[pid] = _deltas_for("I", pid, ck)
            per_pair[pid] = {
                "delta_claim_F": {"m": nanmean(dF_raw[pid])},
                "delta_claim_I": {"m": nanmean(dI_raw[pid])},
                "D": {
                    "m": nanmean(dF_raw[pid]) - nanmean(dI_raw[pid])
                    if dF_raw[pid].size and dI_raw[pid].size
                    else float("nan")
                },
                "n_F": int(dF_raw[pid].size),
                "n_I": int(dI_raw[pid].size),
            }
        delta_claim[ck] = per_pair
        point_D_all = nanmean(
            np.array([nanmean(dF_raw[p]) - nanmean(dI_raw[p]) for p in PAIR_IDS])
        )
        boots_all = cluster_boot_means(dF_raw, dI_raw, PAIR_IDS)
        dF_v = {p: dF_raw[p] for p in PAIR_IDS if p in valid_pairs}
        dI_v = {p: dI_raw[p] for p in PAIR_IDS if p in valid_pairs}
        point_D_v = (
            nanmean(np.array([nanmean(dF_v[p]) - nanmean(dI_v[p]) for p in dF_v]))
            if dF_v
            else float("nan")
        )
        boots_v = (
            cluster_boot_means(dF_v, dI_v, list(dF_v.keys())) if dF_v else np.array([])
        )
        D_block[ck] = {
            "pooled_all": ci_block(point_D_all, boots_all),
            "pooled_valid": ci_block(point_D_v, boots_v if boots_v.size else None),
            "per_pair": {p: per_pair[p]["D"] for p in PAIR_IDS},
            "delta_claim_F_pooled_all": {
                "m": nanmean(np.array([nanmean(dF_raw[p]) for p in PAIR_IDS]))
            },
            "delta_claim_I_pooled_all": {
                "m": nanmean(np.array([nanmean(dI_raw[p]) for p in PAIR_IDS]))
            },
        }

    # LOPO on concept_a4
    lopo = {}
    dF_raw, dI_raw = {}, {}
    ck = "concept_a4"
    for pid in PAIR_IDS:
        for s in ("F", "I"):
            base_lst = cells.get((s, pid, "base"), [])
            ste_lst = cells.get((s, pid, ck), [])
            base_by_p: Dict[int, List[float]] = defaultdict(list)
            for r in base_lst:
                if r.get("claims") is not None:
                    base_by_p[int(r["prompt_idx"])].append(float(r["claims"]))
            base_mean = {p: float(np.mean(v)) for p, v in base_by_p.items() if v}
            deltas = [
                float(r["claims"]) - base_mean[int(r["prompt_idx"])]
                for r in ste_lst
                if r.get("claims") is not None and int(r["prompt_idx"]) in base_mean
            ]
            if s == "F":
                dF_raw[pid] = np.array(deltas, dtype=float)
            else:
                dI_raw[pid] = np.array(deltas, dtype=float)
    for leave in PAIR_IDS:
        keep = [p for p in PAIR_IDS if p != leave]
        dF_k = {p: dF_raw[p] for p in keep}
        dI_k = {p: dI_raw[p] for p in keep}
        dF_m = nanmean(np.array([nanmean(dF_k[p]) for p in keep]))
        dI_m = nanmean(np.array([nanmean(dI_k[p]) for p in keep]))
        lopo[leave] = {
            "delta_claim_F": dF_m,
            "delta_claim_I": dI_m,
            "D": dF_m - dI_m,
        }

    return gen_out, delta_claim, D_block, {"lopo_concept_a4": lopo}


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
            key = (s, int(r["prompt_idx"]), r.get("condition"), r.get("alpha"), vk)
            # condition in gens is concept/random; logit uses steer
            key2 = (s, int(r["prompt_idx"]), "concept" if r.get("condition") == "concept" else "random", r.get("alpha"), vk)
            t1 = top1.get(key2) or top1.get(key) or "other"
            cl = r.get("claims")
            clabel = "null" if cl is None else ("YES" if int(cl) == 1 else "NO")
            tab[t1][clabel] += 1
        out[s] = {t: dict(v) for t, v in tab.items()}
    return out


def make_figures(results_dir: Path, metrics: Dict[str, Any], c_rows: List[dict]) -> None:
    # fig_claims
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
        ax.errorbar(
            xs,
            ys,
            yerr=[np.clip(ys - ylo, 0, None), np.clip(yhi - ys, 0, None)],
            fmt="o",
            capsize=3,
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(["cα2", "cα4", "rα4"])
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_title(f"D ({pool})")
        ax.set_ylabel("D = Δclaim_F − Δclaim_I")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_claims.png", dpi=150)
    plt.close(fig)

    # fig_ld — S4 scatter + identity (use compression fit)
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

    # fig_residual
    resid = metrics.get("compression", {}).get("yesward_resid", {})
    fig, ax = plt.subplots(figsize=(8, 4))
    keys = [k for k in resid if "|a4|concept" in k]
    ys = [resid[k]["yesward_resid"]["m"] for k in keys]
    ax.bar(range(len(keys)), [y if y is not None else float("nan") for y in ys])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.split("|")[0] for k in keys], rotation=45, ha="right")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("Yes-ward residual (α=4 concept)")
    fig.tight_layout()
    fig.savefig(results_dir / "fig_residual.png", dpi=150)
    plt.close(fig)

    # fig_comprehension
    fig, ax = plt.subplots(figsize=(10, 4))
    # mean LD per pair×polarity×mode
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


def main(results_dir: Path) -> None:
    results_dir = Path(results_dir)
    raw = results_dir / "raw"
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "wo_005.yaml").read_text())
    abs_min = float(cfg.get("validity", {}).get("base_ld_abs_min", 3.0))

    summary = {}
    if (results_dir / "run_summary.json").exists():
        summary = json.loads((results_dir / "run_summary.json").read_text())
    judge_summary = {}
    if (results_dir / "judge_summary.json").exists():
        judge_summary = json.loads((results_dir / "judge_summary.json").read_text())

    logit_rows = read_jsonl(raw / "logit_rows.jsonl")
    gens = read_jsonl(raw / "gens.jsonl")
    c_rows = [r for r in logit_rows if r.get("set") == "C"]
    steered = [r for r in logit_rows if r.get("set") != "C"]

    validity = compute_validity(steered, c_rows, abs_min)
    logit = logit_block(steered)
    compression = compression_block(steered)
    gen_out, delta_claim, D_block, extra = gen_metrics(gens, validity)
    contingency = contingency_table(gens, steered)

    metrics = {
        "wo_id": "WO-005",
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
    (results_dir / "metrics.json").write_text(json.dumps(clean(metrics), indent=2))
    make_figures(results_dir, metrics, c_rows)
    print(f"Wrote {results_dir / 'metrics.json'} and figures")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO_ROOT / "results" / "WO-005")
