"""WO-003a: rank-matched gate specificity + damping + DLA (CPU; no forwards)."""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sfo.budget import append_ledger_row  # noqa: E402
from sfo.damping import (  # noqa: E402
    affine_s,
    matched_window,
    midrank_percentile,
    n_at_signed_frac,
    proportional_k,
    ratio_of_sums,
    rho_from_acts,
    rho_prime,
    r2_actual_pred,
    slope_actual_on_pred,
    zscore,
)
from sfo.ld import yes_no_token_ids  # noqa: E402
from wo_003_metrics import (  # noqa: E402
    FULL_SETS,
    SET_ORDER,
    akey,
    load_conditions,
    parse_sparse_key,
    split_sparse_value,
    vec_id,
)

WIDTH = 262144
GATES = {45: {9959, 74631}, 50: {167}}
GATE_KEYS = {45: ["L45_g9959", "L45_g74631"], 50: ["L50_g167"]}
CLASSES = ("concept", "random")
ALPHAS = (2.0, 4.0)


def clean(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    return x


def ci_block(point: float, boots: Optional[np.ndarray]) -> Dict[str, Any]:
    if boots is None or len(boots) == 0 or not np.isfinite(boots).any():
        return {"m": point, "lo95": None, "hi95": None}
    b = boots[np.isfinite(boots)]
    return {
        "m": point,
        "lo95": float(np.percentile(b, 2.5)),
        "hi95": float(np.percentile(b, 97.5)),
    }


def slog(msg: str = "") -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------------------------
# loading helpers
# --------------------------------------------------------------------------------------
def sparse_to_dict(idx: np.ndarray, val: np.ndarray) -> Dict[int, float]:
    return {int(i): float(v) for i, v in zip(idx.tolist(), val.tolist()) if float(v) != 0.0}


def load_sparse_maps(path: Path) -> Dict[Tuple[str, int, str], Dict[int, float]]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    out: Dict[Tuple[str, int, str], Dict[int, float]] = {}
    for k, v in raw.items():
        pk = parse_sparse_key(k)
        if pk is None:
            continue
        idx, val = split_sparse_value(v)
        out[pk] = sparse_to_dict(idx, val)
    del raw
    return out


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_b_enc(folder: Path) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
        return f.get_tensor("b_enc").float().numpy()


def load_w_dec_stats(folder: Path, delta_u: np.ndarray, gamma: np.ndarray) -> Dict[str, np.ndarray]:
    """Stream w_dec to compute dla_raw, dla_normed, and ‖w_dec‖."""
    from safetensors import safe_open

    du = delta_u.astype(np.float64)
    du_n = ((1.0 + gamma) * delta_u).astype(np.float64)
    dla_raw = np.empty(WIDTH, dtype=np.float64)
    dla_norm = np.empty(WIDTH, dtype=np.float64)
    norms = np.empty(WIDTH, dtype=np.float64)
    chunk = 4096
    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
        # w_dec [width, d]
        for i0 in range(0, WIDTH, chunk):
            i1 = min(WIDTH, i0 + chunk)
            w = f.get_slice("w_dec")[i0:i1].float().numpy().astype(np.float64)
            dla_raw[i0:i1] = w @ du
            dla_norm[i0:i1] = w @ du_n
            norms[i0:i1] = np.linalg.norm(w, axis=1)
    return {"dla_raw": dla_raw, "dla_normed": dla_norm, "w_norm": norms}


def find_weight_keys(index: dict) -> Tuple[List[str], List[str], List[str]]:
    weight_map = index.get("weight_map", index)
    unembed_re = re.compile(r"^(language_model\.)?(model\.)?lm_head\.weight$")
    tied_re = re.compile(r"^(language_model\.)?model\.embed_tokens\.weight$")
    norm_re = re.compile(r"^(language_model\.)?model\.norm\.weight$")
    u, t, n = [], [], []
    for k in weight_map:
        if unembed_re.match(k):
            u.append(k)
        if tied_re.match(k):
            t.append(k)
        if norm_re.match(k):
            n.append(k)
    return u, t, n


def load_rows_from_shard(model_dir: Path, shard: str, key: str, rows: Sequence[int]) -> np.ndarray:
    from safetensors import safe_open

    path = model_dir / shard
    with safe_open(str(path), framework="pt", device="cpu") as f:
        # weight [vocab, d] or [d] for norm
        t = f.get_slice(key)
        if len(rows) == 0:
            return f.get_tensor(key).float().numpy()
        # gather rows
        out = []
        for r in rows:
            out.append(np.asarray(t[int(r)].float().numpy(), dtype=np.float64))
        return np.stack(out, axis=0)


def load_delta_u_and_gamma(model_dir: Path, yes_id: int, no_id: int) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        # single-file fallback
        raise FileNotFoundError(f"missing {index_path}")
    index = json.loads(index_path.read_text())
    u_keys, tied_keys, norm_keys = find_weight_keys(index)
    if len(norm_keys) != 1:
        raise RuntimeError(f"norm key match count={len(norm_keys)} keys={norm_keys}")
    if u_keys:
        uk = u_keys[0]
        source = "lm_head"
    elif tied_keys:
        uk = tied_keys[0]
        source = "tied_embed"
    else:
        raise RuntimeError("no unembed / tied embed key matched")
    weight_map = index["weight_map"]
    u_shard = weight_map[uk]
    n_shard = weight_map[norm_keys[0]]
    rows = load_rows_from_shard(model_dir, u_shard, uk, [yes_id, no_id])
    u_yes, u_no = rows[0], rows[1]
    delta_u = u_no - u_yes
    gamma = load_rows_from_shard(model_dir, n_shard, norm_keys[0], [])
    if gamma.ndim != 1:
        gamma = gamma.reshape(-1)
    cfg = json.loads((model_dir / "config.json").read_text())
    softcap = cfg.get("final_logit_softcapping")
    if softcap is None and isinstance(cfg.get("text_config"), dict):
        softcap = cfg["text_config"].get("final_logit_softcapping")
    return {
        "unembed_key": uk,
        "unembed_source": source,
        "norm_key": norm_keys[0],
        "delta_u": delta_u.astype(np.float64),
        "gamma": gamma.astype(np.float64),
        "softcap": softcap,
        "u_yes": u_yes,
        "u_no": u_no,
    }


def assert_yes_no(model_dir: Path, yes_id: int, no_id: int) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    y, n = yes_no_token_ids(tok)
    assert y == yes_id and n == no_id, ((y, n), (yes_id, no_id))
    del tok


# --------------------------------------------------------------------------------------
# per-prompt / per-row core
# --------------------------------------------------------------------------------------
def baseline_ranking(a0: Dict[int, float]) -> Tuple[List[int], Dict[int, int]]:
    """Return feature ids sorted by a0 desc, and 1-based ranks."""
    ids = sorted(a0.keys(), key=lambda f: (-a0[f], f))
    ranks = {f: i + 1 for i, f in enumerate(ids)}
    return ids, ranks


def build_match_sets(
    ranks_desc: List[int],
    ranks: Dict[int, int],
    gate_id: int,
    layer_gates: set,
    w_norm: np.ndarray,
    a0: Dict[int, float],
    dla_normed: np.ndarray,
    p99_dla: float,
    n_match: int,
    n_m50: int,
) -> Optional[Dict[str, Any]]:
    if gate_id not in ranks:
        return None
    gpos = ranks[gate_id] - 1
    M, above, below = matched_window(ranks_desc, gpos, n=n_match, exclude=layer_gates)
    # M50: top 50 non-gate
    M50 = [f for f in ranks_desc if f not in layer_gates][:n_m50]
    # M_noNo: same window but exclude strong No-writers
    strong_no = {f for f in ranks_desc if dla_normed[f] > p99_dla}
    excl = set(layer_gates) | strong_no
    M_noNo, _, _ = matched_window(ranks_desc, gpos, n=n_match, exclude=excl)
    # M_norm: rank by a0 * ||w||
    scored = sorted(
        ((f, a0[f] * float(w_norm[f])) for f in a0 if f not in layer_gates),
        key=lambda t: (-t[1], t[0]),
    )
    # include gate in ranking for its position
    all_scored = sorted(
        ((f, a0[f] * float(w_norm[f])) for f in a0),
        key=lambda t: (-t[1], t[0]),
    )
    ids_norm = [f for f, _ in all_scored]
    try:
        gpos_n = ids_norm.index(gate_id)
    except ValueError:
        return None
    M_norm, _, _ = matched_window(ids_norm, gpos_n, n=n_match, exclude=layer_gates)
    return {
        "r_g": ranks[gate_id],
        "M": M,
        "M50": M50,
        "M_noNo": M_noNo,
        "M_norm": M_norm,
        "window_above": above,
        "window_below": below,
        "n_strong_no_excluded": len(strong_no & set(ranks_desc)),
    }


def act_get(d: Dict[int, float], f: int) -> float:
    return float(d.get(f, 0.0))


def row_gate_stats(
    a0: Dict[int, float],
    a1: Dict[int, float],
    gate_id: int,
    match: Dict[str, Any],
    b_enc: np.ndarray,
) -> Dict[str, Any]:
    a0g, a1g = act_get(a0, gate_id), act_get(a1, gate_id)
    rho_g = rho_from_acts(a0g, a1g)
    out: Dict[str, Any] = {"r_g": match["r_g"], "a0": a0g, "a1": a1g, "rho": rho_g}

    def pack(name: str, feats: List[int], use_prime: bool = False) -> Dict[str, Any]:
        rhos = []
        used = []
        for f in feats:
            if use_prime:
                if act_get(a0, f) <= 0 or act_get(a1, f) <= 0:
                    continue
                r = rho_prime(act_get(a0, f), act_get(a1, f), float(b_enc[f]))
            else:
                if act_get(a0, f) <= 0:
                    continue
                r = rho_from_acts(act_get(a0, f), act_get(a1, f))
            if np.isfinite(r):
                rhos.append(r)
                used.append(f)
        if use_prime:
            rho_use = rho_prime(a0g, a1g, float(b_enc[gate_id])) if (a0g > 0 and a1g > 0) else float("nan")
        else:
            rho_use = rho_g
        p = midrank_percentile(rho_use, rhos) if rhos else float("nan")
        z = zscore(rho_use, rhos) if rhos else float("nan")
        d: Dict[str, Any] = {"p": p, "z": z, "n_match": len(rhos)}
        if use_prime:
            d["n_used"] = len(used)
            d["rho"] = rho_use
        return d

    out["M"] = pack("M", match["M"])
    out["M50"] = pack("M50", match["M50"])
    out["M_noNo"] = pack("M_noNo", match["M_noNo"])
    out["M_norm"] = pack("M_norm", match["M_norm"])
    out["M_pre"] = pack("M_pre", match["M"], use_prime=True)
    out["window_above"] = match["window_above"]
    out["window_below"] = match["window_below"]
    return out


def damping_layer(
    a0: Dict[int, float],
    a1: Dict[int, float],
    layer_gates: set,
    b_enc: np.ndarray,
    gate_id: Optional[int],
) -> Dict[str, Any]:
    # proportional: baseline-active non-gate
    ids0 = [f for f in a0 if f not in layer_gates and a0[f] > 0]
    a0v = np.array([a0[f] for f in ids0], dtype=np.float64)
    a1v = np.array([act_get(a1, f) for f in ids0], dtype=np.float64)
    da = a1v - a0v
    k, k_r2 = proportional_k(a0v, da) if len(ids0) else (float("nan"), float("nan"))
    # affine: active in both, non-gate
    ids_both = [f for f in ids0 if act_get(a1, f) > 0]
    a0b = np.array([a0[f] for f in ids_both], dtype=np.float64)
    a1b = np.array([a1[f] for f in ids_both], dtype=np.float64)
    bb = np.array([b_enc[f] for f in ids_both], dtype=np.float64)
    s, s_r2 = affine_s(a0b, a1b, bb) if len(ids_both) else (float("nan"), float("nan"))
    frac_off = float(np.mean(a1v <= 0)) if len(ids0) else float("nan")
    new_ids = [f for f in a1 if f not in layer_gates and act_get(a0, f) <= 0 and a1[f] > 0]
    new_mass = float(sum(a1[f] for f in new_ids))
    sum_a0 = float(sum(a0[f] for f in ids0)) if ids0 else 0.0
    mass_frac = new_mass / sum_a0 if sum_a0 > 0 else float("nan")
    out: Dict[str, Any] = {
        "k": k, "k_R2": k_r2, "s": s, "s_R2": s_r2,
        "frac_off": frac_off,
        "new_active_count": len(new_ids),
        "new_active_mass_frac": mass_frac,
        "n_base_active": len(ids0),
        "n_both_active": len(ids_both),
    }
    if gate_id is not None and gate_id in a0 and a0[gate_id] > 0:
        a0g = a0[gate_id]
        dag = act_get(a1, gate_id) - a0g
        pred_k = -k * a0g if np.isfinite(k) else float("nan")
        pred_s = (s - 1.0) * (a0g - float(b_enc[gate_id])) if np.isfinite(s) else float("nan")
        out["gate_da"] = dag
        out["pred_da_prop"] = pred_k
        out["pred_da_affine"] = pred_s
    return out


def dla_row(
    a0: Dict[int, float],
    a1: Dict[int, float],
    dla: np.ndarray,
    layer_gates: Sequence[int],
    k: float,
) -> Dict[str, Any]:
    feats = set(a0) | set(a1)
    contrib = []
    ids = []
    for f in feats:
        da = act_get(a1, f) - act_get(a0, f)
        if da == 0.0:
            continue
        c = da * float(dla[f])
        contrib.append(c)
        ids.append(f)
    dDLA = float(sum(contrib)) if contrib else 0.0
    gate_c = float(sum(
        (act_get(a1, g) - act_get(a0, g)) * float(dla[g]) for g in layer_gates
    ))
    # top-20 by |Δa·dla|
    order = np.argsort([-abs(c) for c in contrib])[:20] if contrib else []
    top20 = float(sum(contrib[i] for i in order)) if len(order) else 0.0
    # damping-explained: over baseline-active (incl gates? spec says Σ (-k a0) dla — over features used in fit = non-gate; include all baseline-active non-gate)
    damp = 0.0
    if np.isfinite(k):
        for f, a in a0.items():
            if f in set(layer_gates):
                continue
            if a > 0:
                damp += (-k * a) * float(dla[f])
    n50 = n_at_signed_frac(contrib, dDLA, 0.5)
    n80 = n_at_signed_frac(contrib, dDLA, 0.8)
    share = gate_c / dDLA if dDLA != 0 else float("nan")
    return {
        "dDLA": dDLA,
        "gate_contrib": gate_c,
        "gate_share": share,
        "top20_contrib": top20,
        "damp_contrib": damp,
        "n50": n50,
        "n80": n80,
        "n_contrib": len(contrib),
    }


# --------------------------------------------------------------------------------------
# bootstrap
# --------------------------------------------------------------------------------------
class TwoWayBoot:
    def __init__(self, prompts: List[int], vectors: List[str], n: int, seed: int):
        self.P = len(prompts)
        self.V = len(vectors)
        self.prompts = prompts
        self.vectors = vectors
        rng = np.random.default_rng(seed)
        self.Cp = rng.integers(0, self.P, size=(n, self.P)) if self.P else np.zeros((n, 0), int)
        self.Cv = rng.integers(0, self.V, size=(n, self.V)) if self.V else np.zeros((n, 0), int)

    def mean_of_matrix(self, M: np.ndarray) -> np.ndarray:
        """M [P, V]; return bootstrap means length n."""
        # crossed: for each boot b, mean over Cp[b] × Cv[b]
        n = self.Cp.shape[0]
        out = np.empty(n, dtype=np.float64)
        for b in range(n):
            sub = M[np.ix_(self.Cp[b], self.Cv[b])]
            out[b] = np.nanmean(sub)
        return out


def prompt_only_boot(vals: np.ndarray, n: int, seed: int) -> np.ndarray:
    """vals [P] or [P, n_noise]; resample prompts."""
    rng = np.random.default_rng(seed)
    P = vals.shape[0]
    if P == 0:
        return np.full(n, np.nan)
    idx = rng.integers(0, P, size=(n, P))
    out = np.empty(n, dtype=np.float64)
    for b in range(n):
        out[b] = np.nanmean(vals[idx[b]])
    return out


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def run_wo_003a(
    *,
    config_path: str,
    results_dir: str,
    ledger_path: str,
    git_hash_str: str,
    wo003_raw: Optional[str] = None,
    model_dir: Optional[str] = None,
    transcoder_root: Optional[str] = None,
    n_cores: float = 8.0,
    mem_gib: float = 64.0,
) -> dict:
    t0 = time.perf_counter()
    start = datetime.now(timezone.utc)
    cfg = yaml.safe_load(Path(config_path).read_text())
    results = Path(results_dir)
    raw_out = results / "raw"
    results.mkdir(parents=True, exist_ok=True)
    raw_out.mkdir(parents=True, exist_ok=True)

    wo_raw = Path(wo003_raw or cfg["wo003_raw"])
    model_dir_p = Path(model_dir or cfg["model_dir"])
    tc_root = Path(transcoder_root or cfg["transcoder_root"])
    yes_id, no_id = int(cfg["yes_id"]), int(cfg["no_id"])
    n_boot = int(cfg["n_boot"])
    boot_seed = int(cfg["boot_seed"])
    n_match = int(cfg["n_match"])
    n_m50 = int(cfg["n_m50"])

    sanity: Dict[str, Any] = {}
    errors: List[str] = []

    # --- token assert
    slog("=== assert Yes/No token ids ===")
    assert_yes_no(model_dir_p, yes_id, no_id)
    slog(f"Yes={yes_id} No={no_id} OK")

    # --- load conditions / prompts / scalars / logits
    cids, cmeta = load_conditions(wo_raw)
    steered = [c for c in cids if cmeta[c]["class"] in CLASSES]
    noise = [c for c in cids if cmeta[c]["class"] == "noise_floor"]
    assert "baseline" in cmeta
    prompts_meta = read_jsonl(wo_raw / "prompts.jsonl")
    prompts_by_set: Dict[str, List[int]] = defaultdict(list)
    for p in prompts_meta:
        prompts_by_set[p["set"]].append(int(p["idx"]))
    for s in prompts_by_set:
        prompts_by_set[s] = sorted(prompts_by_set[s])

    gate_rows = read_jsonl(wo_raw / "gate_scalars.jsonl")
    gate_index = {(r["set"], int(r["prompt_idx"]), r["cid"]): r for r in gate_rows}
    ld_rows = read_jsonl(wo_raw / "logits_ld.jsonl")
    ld_index = {(r["set"], int(r["prompt_idx"]), r["cid"]): r for r in ld_rows}

    # --- SANITY 1: sparse load
    slog("=== SANITY 1: load sparse captures ===")
    sparse: Dict[int, Dict[Tuple[str, int, str], Dict[int, float]]] = {}
    for L in (45, 50):
        path = wo_raw / f"captures_sparse_L{L}.pt"
        sparse[L] = load_sparse_maps(path)
        slog(f"L{L}: {len(sparse[L])} entries")
        assert len(sparse[L]) == 38038, len(sparse[L])
        missing = []
        for p in prompts_meta:
            for c in cids:
                k = (p["set"], int(p["idx"]), c)
                if k not in sparse[L]:
                    missing.append(k)
                    if len(missing) > 5:
                        break
            if len(missing) > 5:
                break
        assert not missing, missing[:5]
    sanity["s1_n_keys"] = {str(L): len(sparse[L]) for L in sparse}
    slog("SANITY1 PASS")

    # --- SANITY 2: sparse vs fp32 gates
    slog("=== SANITY 2: sparse vs fp32 gate agreement ===")
    rng = np.random.default_rng(0)
    keys = list(sparse[45].keys())
    sample = [keys[i] for i in rng.choice(len(keys), size=100, replace=False)]
    max_rel = 0.0
    n_ok = 0
    for set_, pi, cid in sample:
        gr = gate_index[(set_, pi, cid)]
        for L, gids in ((45, [9959, 74631]), (50, [167])):
            acts_fp = gr[f"L{L}"]["act"]
            sp = sparse[L][(set_, pi, cid)]
            for j, g in enumerate(gids):
                a32 = float(acts_fp[j])
                a16 = float(sp.get(g, 0.0))
                if a32 == 0 and a16 == 0:
                    n_ok += 1
                    continue
                rel = abs(a16 - a32) / max(abs(a32), 1e-12)
                max_rel = max(max_rel, rel)
                assert rel <= 0.03 or (a32 == 0 and a16 == 0), (set_, pi, cid, L, g, a32, a16, rel)
                n_ok += 1
    # D_act S1 concept α=4 from sparse
    dacts = {gk: [] for gk in ("L45_g9959", "L45_g74631", "L50_g167")}
    for pi in prompts_by_set["S1"]:
        base45 = sparse[45][("S1", pi, "baseline")]
        base50 = sparse[50][("S1", pi, "baseline")]
        for c in steered:
            if cmeta[c]["class"] != "concept" or float(cmeta[c]["alpha"]) != 4.0:
                continue
            s45, s50 = sparse[45][("S1", pi, c)], sparse[50][("S1", pi, c)]
            dacts["L45_g9959"].append(act_get(s45, 9959) - act_get(base45, 9959))
            dacts["L45_g74631"].append(act_get(s45, 74631) - act_get(base45, 74631))
            dacts["L50_g167"].append(act_get(s50, 167) - act_get(base50, 167))
    d_act_sparse = {k: float(np.mean(v)) for k, v in dacts.items()}
    wo003_m = {}
    mpath_candidates = [
        Path(cfg.get("wo003_metrics", "")),
        wo_raw.parent / "metrics.json",
        Path("/root/results/WO-003/metrics.json"),
        Path("/models/results/WO-003/metrics.json"),
    ]
    mpath = next((p for p in mpath_candidates if p and p.exists()), None)
    if mpath is not None:
        m003 = json.loads(mpath.read_text())
        for gk in d_act_sparse:
            cell = m003["gates"]["S1"]["4"]["concept"][gk]
            wo003_m[gk] = cell["D_act"]["m"]
    sanity["s2"] = {
        "max_rel": max_rel, "n_checked": n_ok,
        "D_act_sparse_S1_c4": d_act_sparse,
        "D_act_wo003_S1_c4": wo003_m,
        "wo003_metrics_path": str(mpath) if mpath else None,
    }
    slog(f"max_rel={max_rel:.4e} D_act_sparse={d_act_sparse} wo003={wo003_m}")
    slog("SANITY2 PASS")

    # --- load b_enc + DLA weights
    slog("=== load b_enc, Δu, γ, w_dec DLA ===")
    b_enc = {}
    dla_stats = {}
    du_info = load_delta_u_and_gamma(model_dir_p, yes_id, no_id)
    slog(f"unembed={du_info['unembed_key']} ({du_info['unembed_source']}) norm={du_info['norm_key']}")
    slog(f"‖Δu‖={np.linalg.norm(du_info['delta_u']):.4f} softcap={du_info['softcap']}")
    for L in (45, 50):
        folder = tc_root / cfg["transcoder_subdir"].format(layer=L)
        b_enc[L] = load_b_enc(folder)
        dla_stats[L] = load_w_dec_stats(folder, du_info["delta_u"], du_info["gamma"])
        slog(f"L{L}: dla_normed mean={dla_stats[L]['dla_normed'].mean():.4e}")

    p99 = {L: float(np.percentile(dla_stats[L]["dla_normed"], 99)) for L in (45, 50)}
    p01 = {L: float(np.percentile(dla_stats[L]["dla_normed"], 1)) for L in (45, 50)}

    # --- SANITY 4: gate dla
    slog("=== SANITY 4: gate DLA ===")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(model_dir_p), local_files_only=True)
    sanity4 = {
        "unembed_key": du_info["unembed_key"],
        "norm_key": du_info["norm_key"],
        "yes": {"id": yes_id, "str": tok.decode([yes_id])},
        "no": {"id": no_id, "str": tok.decode([no_id])},
        "delta_u_norm": float(np.linalg.norm(du_info["delta_u"])),
        "softcap": du_info["softcap"],
        "gates": {},
    }
    del tok
    for L, gset in GATES.items():
        for g in sorted(gset):
            raw = float(dla_stats[L]["dla_raw"][g])
            normed = float(dla_stats[L]["dla_normed"][g])
            pr = float(100.0 * np.mean(dla_stats[L]["dla_raw"] <= raw))
            pn = float(100.0 * np.mean(dla_stats[L]["dla_normed"] <= normed))
            sanity4["gates"][f"L{L}_g{g}"] = {
                "raw": raw, "normed": normed, "pctl_raw": pr, "pctl_normed": pn,
                "high_positive_normed": normed > 0 and pn >= 90.0,
            }
            slog(f"L{L}_g{g}: raw={raw:.4f} normed={normed:.4f} pctl_n={pn:.2f}")
    sanity["s4"] = sanity4

    # --- SANITY 3: ranks
    slog("=== SANITY 3: gate ranks ===")
    s3 = {"S1": {}, "S4": {}}
    for set_name, n_show in (("S1", 6), ("S4", 5)):
        for pi in prompts_by_set[set_name][:n_show]:
            s3[set_name][str(pi)] = {}
            for L in (45, 50):
                a0 = sparse[L][(set_name, pi, "baseline")]
                _, ranks = baseline_ranking(a0)
                for g in sorted(GATES[L]):
                    s3[set_name][str(pi)][f"L{L}_g{g}"] = ranks.get(g)
                    slog(f"{set_name}[{pi}] L{L}_g{g} r={ranks.get(g)}")
    sanity["s3_ranks"] = s3

    # =====================================================================
    # Main per-row loop
    # =====================================================================
    slog("=== MAIN per-row analysis ===")
    per_row_path = raw_out / "per_row.jsonl"
    # collectors for aggregation
    # rank_matched[S][alpha][class][gate][variant] -> list of (prompt, vec, p, z)
    RM: Dict[Any, List[Tuple[int, str, float, float]]] = defaultdict(list)
    RM_pre_n: Dict[Any, List[int]] = defaultdict(list)
    rank_dist: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    win_ab: Dict[Tuple[str, str], List[Tuple[int, int]]] = defaultdict(list)
    NF: Dict[Any, List[Tuple[int, float]]] = defaultdict(list)  # noise floor p
    damp_vals: Dict[Any, List[float]] = defaultdict(list)
    gate_pred: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: {"prop": [], "affine": []}
    )
    gate_pred_set: Dict[Tuple[str, str], Dict[str, List[Tuple[float, float]]]] = defaultdict(
        lambda: {"prop": [], "affine": []}
    )
    dla_acc: Dict[Any, Dict[str, List[float]]] = defaultdict(
        lambda: {"gate_c": [], "dDLA": [], "top20": [], "damp": [], "share": [],
                 "n50": [], "n80": [], "dDLA_raw": []}
    )
    writers_acc: Dict[Any, Dict[str, List[float]]] = defaultdict(
        lambda: {"yes_rho": [], "yes_p": [], "yes_new": [], "no_rho": [], "no_p": []}
    )
    validity_pts: Dict[Any, List[Tuple[float, float, float, float]]] = defaultdict(list)
    # (dDLA45+50, l1, rand_dla, dLD)

    concept_order = sorted(
        [c for c in steered if cmeta[c]["class"] == "concept" and float(cmeta[c]["alpha"]) == 4.0],
        key=lambda c: (vec_id(cmeta[c]), c),
    )[: int(cfg["n_inspect_concepts"])]
    inspection_lines: List[str] = []

    # ρ vs rank bins for fig: (set, layer) -> bin_idx -> list of ρ; plus overlays
    rank_bins_def = [(1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 10**9)]
    rho_bins: Dict[Tuple[str, int], List[List[float]]] = defaultdict(
        lambda: [[] for _ in rank_bins_def]
    )
    rho_gate_pts: Dict[Tuple[str, int], List[Tuple[int, float]]] = defaultdict(list)
    rho_yes_pts: Dict[Tuple[str, int], List[Tuple[int, float]]] = defaultdict(list)
    rho_no_pts: Dict[Tuple[str, int], List[Tuple[int, float]]] = defaultdict(list)

    n_rows_written = 0
    with per_row_path.open("w") as fout:
        for set_name in SET_ORDER:
            pis = prompts_by_set.get(set_name, [])
            for pi in pis:
                base = {L: sparse[L][(set_name, pi, "baseline")] for L in (45, 50)}
                ranks_desc = {}
                ranks = {}
                matches = {}
                for L in (45, 50):
                    ranks_desc[L], ranks[L] = baseline_ranking(base[L])
                    for g in GATES[L]:
                        if g in ranks[L]:
                            rank_dist[(set_name, f"L{L}_g{g}")].append(ranks[L][g])
                        m = build_match_sets(
                            ranks_desc[L], ranks[L], g, GATES[L],
                            dla_stats[L]["w_norm"], base[L],
                            dla_stats[L]["dla_normed"], p99[L], n_match, n_m50,
                        )
                        matches[(L, g)] = m
                        if m is not None:
                            win_ab[(set_name, f"L{L}_g{g}")].append(
                                (m["window_above"], m["window_below"])
                            )

                # Yes/No writer feature sets at this prompt
                yes_feats = {L: [f for f in base[L] if dla_stats[L]["dla_normed"][f] < p01[L]
                                 and f not in GATES[L]] for L in (45, 50)}
                no_feats = {L: [f for f in base[L] if dla_stats[L]["dla_normed"][f] > p99[L]
                                and f not in GATES[L]] for L in (45, 50)}
                writer_match = {}
                for L in (45, 50):
                    for kind, feats in (("yes", yes_feats[L]), ("no", no_feats[L])):
                        for f in feats:
                            try:
                                gpos = ranks[L][f] - 1
                            except KeyError:
                                continue
                            M, _, _ = matched_window(ranks_desc[L], gpos, n=n_match, exclude=GATES[L] | {f})
                            writer_match[(L, kind, f)] = M

                analysed = steered + noise
                for cid in analysed:
                    a1 = {L: sparse[L][(set_name, pi, cid)] for L in (45, 50)}
                    klass = cmeta[cid]["class"]
                    alpha = float(cmeta[cid]["alpha"])
                    vid = vec_id(cmeta[cid]) if klass in CLASSES else cid
                    row_out: Dict[str, Any] = {
                        "set": set_name, "prompt_idx": pi, "cid": cid,
                        "class": klass, "alpha": alpha, "vec": vid, "gates": {},
                        "damping": {}, "dla": {},
                    }

                    # damping per layer (exclude gates from fits)
                    damp_L = {}
                    for L in (45, 50):
                        # use first gate only for pred storage; compute once
                        dmp = damping_layer(base[L], a1[L], GATES[L], b_enc[L], None)
                        damp_L[L] = dmp
                        if klass in CLASSES:
                            key_d = (set_name, akey(alpha), klass, f"L{L}")
                            for fld in ("k", "k_R2", "s", "s_R2", "frac_off",
                                        "new_active_count", "new_active_mass_frac"):
                                damp_vals[(key_d, fld)].append(dmp[fld])
                        row_out["damping"][f"L{L}"] = {
                            k: dmp[k] for k in ("k", "k_R2", "s", "s_R2", "frac_off",
                                                "new_active_count", "new_active_mass_frac")
                        }

                    # gates
                    for L in (45, 50):
                        for g in sorted(GATES[L]):
                            gk = f"L{L}_g{g}"
                            m = matches.get((L, g))
                            if m is None or base[L].get(g, 0) <= 0:
                                row_out["gates"][gk] = {"active_baseline": False}
                                continue
                            gs = row_gate_stats(base[L], a1[L], g, m, b_enc[L])
                            row_out["gates"][gk] = gs
                            # gate predictions
                            a0g = base[L][g]
                            dag = act_get(a1[L], g) - a0g
                            k, s = damp_L[L]["k"], damp_L[L]["s"]
                            pred_k = -k * a0g if np.isfinite(k) else float("nan")
                            pred_s = (s - 1.0) * (a0g - float(b_enc[L][g])) if np.isfinite(s) else float("nan")
                            gate_pred[gk]["prop"].append((dag, pred_k))
                            gate_pred[gk]["affine"].append((dag, pred_s))
                            gate_pred_set[(gk, set_name)]["prop"].append((dag, pred_k))
                            gate_pred_set[(gk, set_name)]["affine"].append((dag, pred_s))
                            row_out["gates"][gk]["pred_prop"] = pred_k
                            row_out["gates"][gk]["pred_affine"] = pred_s

                            if klass in CLASSES:
                                for var in ("M", "M50", "M_noNo", "M_norm", "M_pre"):
                                    RM[(set_name, akey(alpha), klass, gk, var)].append(
                                        (pi, vid, gs[var]["p"], gs[var].get("z", float("nan")))
                                    )
                                    if var == "M_pre":
                                        RM_pre_n[(set_name, akey(alpha), klass, gk)].append(
                                            int(gs[var].get("n_used", 0))
                                        )
                            elif klass == "noise_floor":
                                NF[(set_name, gk)].append((pi, gs["M"]["p"]))

                    # DLA
                    dDLA_sum_n = 0.0
                    dDLA_sum_r = 0.0
                    l1 = 0.0
                    rand_dlas = []
                    for L in (45, 50):
                        d_n = dla_row(base[L], a1[L], dla_stats[L]["dla_normed"],
                                      sorted(GATES[L]), damp_L[L]["k"])
                        d_r = dla_row(base[L], a1[L], dla_stats[L]["dla_raw"],
                                      sorted(GATES[L]), damp_L[L]["k"])
                        row_out["dla"][f"L{L}"] = {**d_n, "dDLA_raw": d_r["dDLA"]}
                        dDLA_sum_n += d_n["dDLA"]
                        dDLA_sum_r += d_r["dDLA"]
                        # L1 null
                        feats = set(base[L]) | set(a1[L])
                        l1 += sum(abs(act_get(a1[L], f) - act_get(base[L], f)) for f in feats)
                        if klass in CLASSES:
                            key = (set_name, akey(alpha), klass, f"L{L}")
                            dla_acc[key]["gate_c"].append(d_n["gate_contrib"])
                            dla_acc[key]["dDLA"].append(d_n["dDLA"])
                            dla_acc[key]["top20"].append(d_n["top20_contrib"])
                            dla_acc[key]["damp"].append(d_n["damp_contrib"])
                            dla_acc[key]["share"].append(d_n["gate_share"])
                            dla_acc[key]["n50"].append(d_n["n50"] if d_n["n50"] is not None else np.nan)
                            dla_acc[key]["n80"].append(d_n["n80"] if d_n["n80"] is not None else np.nan)
                            dla_acc[key]["dDLA_raw"].append(d_r["dDLA"])

                        # writers
                        for kind, feats in (("yes", yes_feats[L]), ("no", no_feats[L])):
                            rhos, ps = [], []
                            for f in feats:
                                r = rho_from_acts(act_get(base[L], f), act_get(a1[L], f))
                                if not np.isfinite(r):
                                    continue
                                rhos.append(r)
                                M = writer_match.get((L, kind, f), [])
                                rhos_m = [rho_from_acts(act_get(base[L], ff), act_get(a1[L], ff))
                                          for ff in M if act_get(base[L], ff) > 0]
                                rhos_m = [x for x in rhos_m if np.isfinite(x)]
                                ps.append(midrank_percentile(r, rhos_m) if rhos_m else float("nan"))
                            new_mass = float(sum(
                                act_get(a1[L], f) for f in a1[L]
                                if act_get(base[L], f) <= 0 and a1[L][f] > 0
                                and dla_stats[L]["dla_normed"][f] < p01[L]
                                and f not in GATES[L]
                            )) if kind == "yes" else float("nan")
                            if klass in CLASSES:
                                wk = (set_name, akey(alpha), klass, f"L{L}")
                                if rhos:
                                    writers_acc[wk][f"{kind}_rho"].append(float(np.mean(rhos)))
                                if ps and np.isfinite(ps).any():
                                    writers_acc[wk][f"{kind}_p"].append(float(np.nanmean(ps)))
                                if kind == "yes":
                                    writers_acc[wk]["yes_new"].append(new_mass)

                        # random subset ΔDLA for validity
                        if set_name in FULL_SETS:
                            active = list(set(base[L]) | set(a1[L]))
                            sz = max(1, len(active) // 10)
                            rng_r = np.random.default_rng(0)
                            vals = []
                            for _ in range(20):
                                if not active:
                                    vals.append(0.0)
                                    continue
                                sub = rng_r.choice(active, size=min(sz, len(active)), replace=False)
                                vals.append(float(sum(
                                    (act_get(a1[L], int(f)) - act_get(base[L], int(f)))
                                    * float(dla_stats[L]["dla_normed"][int(f)])
                                    for f in sub
                                )))
                            rand_dlas.append(float(np.mean(vals)))

                    if set_name in FULL_SETS and klass in CLASSES:
                        base_ld = ld_index[(set_name, pi, "baseline")]["ld"]
                        row_ld = ld_index[(set_name, pi, cid)]["ld"]
                        dLD = float(row_ld) - float(base_ld)
                        validity_pts[(set_name, akey(alpha), klass)].append(
                            (dDLA_sum_n, l1, float(np.sum(rand_dlas)), dLD)
                        )

                    fout.write(json.dumps(clean(row_out)) + "\n")
                    n_rows_written += 1

                # ρ-vs-rank accumulation (S1/S4, concept α=4)
                if set_name in ("S1", "S4"):
                    for cid in steered:
                        if cmeta[cid]["class"] != "concept" or float(cmeta[cid]["alpha"]) != 4.0:
                            continue
                        for L in (45, 50):
                            a0, a1s = base[L], sparse[L][(set_name, pi, cid)]
                            rnk = ranks[L]
                            for f, a0f in a0.items():
                                r = rho_from_acts(a0f, act_get(a1s, f))
                                if not np.isfinite(r):
                                    continue
                                rg = rnk[f]
                                for bi, (lo, hi) in enumerate(rank_bins_def):
                                    if lo <= rg <= hi:
                                        rho_bins[(set_name, L)][bi].append(r)
                                        break
                                if f in GATES[L]:
                                    rho_gate_pts[(set_name, L)].append((rg, r))
                                elif dla_stats[L]["dla_normed"][f] < p01[L]:
                                    rho_yes_pts[(set_name, L)].append((rg, r))
                                elif dla_stats[L]["dla_normed"][f] > p99[L]:
                                    rho_no_pts[(set_name, L)].append((rg, r))

                # inspection
                if (set_name in cfg["inspect_sets"] and pi == int(cfg["inspect_prompt_idx"])):
                    for cid in concept_order:
                        for L in (45, 50):
                            a0, a1s = base[L], sparse[L][(set_name, pi, cid)]
                            ids_sorted, rnk = baseline_ranking(a0)
                            inspection_lines.append(
                                f"=== {set_name} prompt={pi} {cid} L{L} ==="
                            )
                            for f in ids_sorted[: int(cfg["inspect_top_n"])]:
                                mark = " GATE" if f in GATES[L] else ""
                                inspection_lines.append(
                                    f"id={f} rank={rnk[f]} a0={a0[f]:.4f} "
                                    f"a1={act_get(a1s, f):.4f} "
                                    f"rho={rho_from_acts(a0[f], act_get(a1s, f)):.4f} "
                                    f"dla_n={dla_stats[L]['dla_normed'][f]:.6f}{mark}"
                                )

    (raw_out / "inspection_top30.txt").write_text("\n".join(inspection_lines) + "\n")
    slog(f"wrote {n_rows_written} per-row lines")

    # --- SANITY 5 noise floor mean p
    slog("=== SANITY 5: noise-floor mean p ===")
    nf_summary = {}
    for (s, gk), pairs in sorted(NF.items()):
        vals = np.array([p for _, p in pairs], dtype=float)
        nf_summary.setdefault(s, {})[gk] = {"mean_p": float(np.nanmean(vals)), "n": int(np.isfinite(vals).sum())}
        slog(f"noise {s} {gk}: mean_p={nf_summary[s][gk]['mean_p']:.3f} n={nf_summary[s][gk]['n']}")
    sanity["s5_noise_floor"] = nf_summary

    # =====================================================================
    # Aggregate metrics
    # =====================================================================
    slog("=== aggregate metrics + bootstrap ===")

    def matrix_from_tuples(tuples, prompts, vectors):
        """list of (pi, vid, p, z) -> P×V matrix of p (and z)."""
        pmap = {p: i for i, p in enumerate(prompts)}
        vmap = {v: i for i, v in enumerate(vectors)}
        Mp = np.full((len(prompts), len(vectors)), np.nan)
        Mz = np.full((len(prompts), len(vectors)), np.nan)
        for pi, vid, p, z in tuples:
            if pi in pmap and vid in vmap:
                Mp[pmap[pi], vmap[vid]] = p
                Mz[pmap[pi], vmap[vid]] = z
        return Mp, Mz

    rank_matched_out: Dict[str, Any] = {}
    contrast_out: Dict[str, Any] = {}

    for set_name in SET_ORDER:
        pis = prompts_by_set.get(set_name, [])
        for a in ALPHAS:
            for klass in CLASSES:
                # vectors for this class/alpha
                vecs = sorted({
                    vec_id(cmeta[c]) for c in steered
                    if cmeta[c]["class"] == klass and float(cmeta[c]["alpha"]) == a
                })
                boot = TwoWayBoot(pis, vecs, n_boot, boot_seed)
                for L in (45, 50):
                    for g in sorted(GATES[L]):
                        gk = f"L{L}_g{g}"
                        # NA if gate active on <3 prompts
                        n_active_prompts = len(rank_dist.get((set_name, gk), []))
                        # actually rank_dist counts every baseline-active prompt once
                        cell: Dict[str, Any] = {}
                        rd = rank_dist.get((set_name, gk), [])
                        if len(rd) >= 3:
                            cell["rank_dist"] = {
                                "median": float(np.median(rd)),
                                "q25": float(np.percentile(rd, 25)),
                                "q75": float(np.percentile(rd, 75)),
                                "n_prompts_active": len(rd),
                            }
                            wab = win_ab.get((set_name, gk), [])
                            if wab:
                                cell["window_above_below"] = {
                                    "mean_above": float(np.mean([a for a, _ in wab])),
                                    "mean_below": float(np.mean([b for _, b in wab])),
                                }
                            for var in ("M", "M50", "M_noNo", "M_norm", "M_pre"):
                                tups = RM.get((set_name, akey(a), klass, gk, var), [])
                                # filter to prompts where we have data
                                Mp, Mz = matrix_from_tuples(tups, pis, vecs)
                                # prompts with any finite
                                n_prompts = int(np.sum(np.isfinite(Mp).any(axis=1))) if Mp.size else 0
                                if n_prompts < 3:
                                    cell[var] = None
                                    continue
                                mean_p = float(np.nanmean(Mp))
                                boots = boot.mean_of_matrix(Mp)
                                blk = {
                                    "mean_p": ci_block(mean_p, boots),
                                    "mean_z": float(np.nanmean(Mz)),
                                    "frac_p_le10": float(np.nanmean(Mp <= 10.0)),
                                    "n_rows": int(np.isfinite(Mp).sum()),
                                    "n_prompts": n_prompts,
                                }
                                if var == "M_pre":
                                    ns = RM_pre_n.get((set_name, akey(a), klass, gk), [])
                                    blk["n_used"] = float(np.mean(ns)) if ns else None
                                cell[var] = blk
                        else:
                            cell = None
                        rank_matched_out.setdefault(set_name, {}).setdefault(akey(a), {}).setdefault(
                            klass, {}
                        )[gk] = cell

        # concept - random contrast at α=4
        for L in (45, 50):
            for g in sorted(GATES[L]):
                gk = f"L{L}_g{g}"
                tc = RM.get((set_name, "4", "concept", gk, "M"), [])
                tr = RM.get((set_name, "4", "random", gk, "M"), [])
                if not tc or not tr:
                    continue
                vc = sorted({vid for _, vid, _, _ in tc})
                vr = sorted({vid for _, vid, _, _ in tr})
                # shared prompt list
                Mc, _ = matrix_from_tuples(tc, pis, vc)
                Mr, _ = matrix_from_tuples(tr, pis, vr)
                diff = float(np.nanmean(Mc) - np.nanmean(Mr))
                # bootstrap: resample prompts + each vector set
                rngb = np.random.default_rng(boot_seed)
                boots = []
                P = len(pis)
                for _ in range(n_boot):
                    ip = rngb.integers(0, P, size=P)
                    ic = rngb.integers(0, len(vc), size=len(vc)) if vc else []
                    ir = rngb.integers(0, len(vr), size=len(vr)) if vr else []
                    boots.append(float(np.nanmean(Mc[np.ix_(ip, ic)]) - np.nanmean(Mr[np.ix_(ip, ir)])))
                contrast_out.setdefault(set_name, {})[gk] = ci_block(diff, np.array(boots))

    # noise floor block
    noise_out: Dict[str, Any] = {}
    for (s, gk), pairs in NF.items():
        # average the 2 noise rows per prompt then boot prompts
        by_p: Dict[int, List[float]] = defaultdict(list)
        for pi, p in pairs:
            by_p[pi].append(p)
        pis = sorted(by_p)
        vals = np.array([np.nanmean(by_p[p]) for p in pis], dtype=float)
        boots = prompt_only_boot(vals, n_boot, boot_seed)
        noise_out.setdefault(s, {})[gk] = {"mean_p": ci_block(float(np.nanmean(vals)), boots)}

    # damping aggregates
    damp_out: Dict[str, Any] = {}
    damp_keys = {(k[0][0], k[0][1], k[0][2], k[0][3]) for k in damp_vals}
    for set_name, a, klass, Lk in sorted(damp_keys):
        cell: Dict[str, Any] = {}
        for fld in ("k", "k_R2", "s", "s_R2", "frac_off"):
            v = np.array(damp_vals.get(((set_name, a, klass, Lk), fld), []), dtype=float)
            if v.size == 0:
                continue
            cell[fld] = {
                "mean": float(np.nanmean(v)), "sd": float(np.nanstd(v)),
                "q25": float(np.nanpercentile(v, 25)), "q50": float(np.nanpercentile(v, 50)),
                "q75": float(np.nanpercentile(v, 75)),
            }
        vcount = np.array(damp_vals.get(((set_name, a, klass, Lk), "new_active_count"), []), float)
        vmass = np.array(damp_vals.get(((set_name, a, klass, Lk), "new_active_mass_frac"), []), float)
        cell["new_active"] = {
            "count_mean": float(np.nanmean(vcount)) if vcount.size else None,
            "mass_frac_mean": float(np.nanmean(vmass)) if vmass.size else None,
        }
        damp_out.setdefault(set_name, {}).setdefault(a, {}).setdefault(klass, {})[Lk] = cell

    def pred_block(pairs: List[Tuple[float, float]]) -> Dict[str, Any]:
        if not pairs:
            return {"R2": None, "slope": None, "n": 0}
        act = np.array([a for a, p in pairs if np.isfinite(a) and np.isfinite(p)], float)
        pred = np.array([p for a, p in pairs if np.isfinite(a) and np.isfinite(p)], float)
        if act.size == 0:
            return {"R2": None, "slope": None, "n": 0}
        return {"R2": r2_actual_pred(act, pred), "slope": slope_actual_on_pred(act, pred), "n": int(act.size)}

    gate_pred_out: Dict[str, Any] = {}
    for gk, d in gate_pred.items():
        gate_pred_out[gk] = {
            "prop": {"pooled": pred_block(d["prop"]), "by_set": {}},
            "affine": {"pooled": pred_block(d["affine"]), "by_set": {}},
        }
        for set_name in SET_ORDER:
            gate_pred_out[gk]["prop"]["by_set"][set_name] = pred_block(
                gate_pred_set[(gk, set_name)]["prop"]
            )
            gate_pred_out[gk]["affine"]["by_set"][set_name] = pred_block(
                gate_pred_set[(gk, set_name)]["affine"]
            )

    # DLA aggregates
    dla_out: Dict[str, Any] = {
        "keys": {"unembed": du_info["unembed_key"], "norm": du_info["norm_key"]},
        "softcap": du_info["softcap"],
        "gate_dla": sanity4["gates"],
        "p01_dla": {str(L): p01[L] for L in p01},
        "p99_dla": {str(L): p99[L] for L in p99},
    }
    for set_name in SET_ORDER:
        for a in ALPHAS:
            for klass in CLASSES:
                for L in (45, 50):
                    key = (set_name, akey(a), klass, f"L{L}")
                    acc = dla_acc.get(key)
                    if not acc or not acc["dDLA"]:
                        continue
                    gc = np.array(acc["gate_c"], float)
                    dd = np.array(acc["dDLA"], float)
                    t20 = np.array(acc["top20"], float)
                    dm = np.array(acc["damp"], float)
                    # bootstrap ratio-of-sums over rows (treat each row as unit; two-way approx via prompts×vecs hard here — use row resample)
                    rngb = np.random.default_rng(boot_seed)
                    n = len(dd)
                    def ros_boot(num, den):
                        point = ratio_of_sums(num, den)
                        boots = []
                        for _ in range(n_boot):
                            idx = rngb.integers(0, n, size=n)
                            boots.append(ratio_of_sums(num[idx], den[idx]))
                        return ci_block(point, np.array(boots))
                    cell = {
                        "gate_share_ros": ros_boot(gc, dd),
                        "gate_share_median": float(np.nanmedian(acc["share"])),
                        "top20_share_ros": ros_boot(t20, dd),
                        "damping_share_ros": ros_boot(dm, dd),
                        "n50": float(np.nanmean(acc["n50"])),
                        "n80": float(np.nanmean(acc["n80"])),
                        "n50_not_reached_frac": float(np.mean(~np.isfinite(acc["n50"]))),
                        "n80_not_reached_frac": float(np.mean(~np.isfinite(acc["n80"]))),
                        "mean_dDLA": float(np.nanmean(dd)),
                        "mean_dDLA_raw": float(np.nanmean(acc["dDLA_raw"])),
                    }
                    dla_out.setdefault(set_name, {}).setdefault(akey(a), {}).setdefault(
                        klass, {}
                    )[f"L{L}"] = cell

    # writers
    writers_out: Dict[str, Any] = {}
    for key, acc in writers_acc.items():
        set_name, a, klass, Lk = key
        cell = {}
        for kind in ("yes", "no"):
            rh = np.array(acc[f"{kind}_rho"], float)
            pp = np.array(acc[f"{kind}_p"], float)
            boots_r = prompt_only_boot(rh, n_boot, boot_seed) if rh.size else None
            boots_p = prompt_only_boot(pp, n_boot, boot_seed) if pp.size else None
            entry = {
                "mean_rho": ci_block(float(np.nanmean(rh)) if rh.size else float("nan"), boots_r),
                "mean_p": ci_block(float(np.nanmean(pp)) if pp.size else float("nan"), boots_p),
            }
            if kind == "yes":
                entry["new_active_mass"] = float(np.nanmean(acc["yes_new"])) if acc["yes_new"] else None
            cell[kind] = entry
        writers_out.setdefault(set_name, {}).setdefault(a, {}).setdefault(klass, {})[Lk] = cell
    dla_out["writers"] = writers_out

    # validity
    validity_out: Dict[str, Any] = {}
    for (set_name, a, klass), pts in validity_pts.items():
        arr = np.array(pts, float)
        if arr.shape[0] < 3:
            continue
        dDLA, l1v, randv, dLD = arr.T

        def pearson(x, y):
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 3 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                return float("nan")
            return float(np.corrcoef(x[m], y[m])[0, 1])

        validity_out.setdefault(set_name, {}).setdefault(a, {})[klass] = {
            "r_dDLA": pearson(dDLA, dLD),
            "r_l1null": pearson(l1v, dLD),
            "r_randsubset": pearson(randv, dLD),
            "n": int(arr.shape[0]),
        }
    dla_out["validity"] = validity_out

    metrics = {
        "wo_id": "WO-003a",
        "rank_matched": {
            **rank_matched_out,
            "noise_floor": noise_out,
            "contrast_concept_minus_random": contrast_out,
        },
        "damping": {**damp_out, "gate_pred": gate_pred_out},
        "dla": dla_out,
        "sanity": sanity,
        "git_hash": git_hash_str,
        "inputs": {
            "wo003_commit": cfg.get("depends_on_commit"),
            "files": {
                "sparse_L45": str(wo_raw / "captures_sparse_L45.pt"),
                "sparse_L50": str(wo_raw / "captures_sparse_L50.pt"),
                "conditions": str(wo_raw / "conditions.json"),
                "gate_scalars": str(wo_raw / "gate_scalars.jsonl"),
                "logits_ld": str(wo_raw / "logits_ld.jsonl"),
            },
        },
        "errors": errors,
        "n_per_row": n_rows_written,
    }
    (results / "metrics.json").write_text(json.dumps(clean(metrics), indent=2))
    slog("wrote metrics.json")

    # figures
    make_figures(
        results, metrics, gate_pred,
        rho_bins=rho_bins, rank_bins_def=rank_bins_def,
        rho_gate_pts=rho_gate_pts, rho_yes_pts=rho_yes_pts, rho_no_pts=rho_no_pts,
    )
    end = datetime.now(timezone.utc)
    append_ledger_row(
        ledger_path,
        wo_id="WO-003a",
        function="wo_003a_analysis",
        gpu_type="CPU",
        start_utc=start,
        end_utc=end,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
    )
    summary = {
        "status": "COMPLETE",
        "seconds": time.perf_counter() - t0,
        "git_hash": git_hash_str,
        "n_per_row": n_rows_written,
        "errors": errors,
        "sanity": {k: sanity[k] for k in sanity if k.startswith("s")},
    }
    (results / "run_summary.json").write_text(json.dumps(clean(summary), indent=2))
    slog(f"DONE in {summary['seconds']:.1f}s")
    return summary


def make_figures(
    results: Path,
    metrics: dict,
    gate_pred: dict,
    *,
    rho_bins: dict,
    rank_bins_def: list,
    rho_gate_pts: dict,
    rho_yes_pts: dict,
    rho_no_pts: dict,
) -> None:
    sets = [s for s in SET_ORDER if s in metrics["rank_matched"]]
    gates = ["L45_g9959", "L45_g74631", "L50_g167"]
    labels = ["1-5", "6-10", "11-20", "21-50", "51-100", "101-200", ">200"]

    # fig_rankmatched_pctl
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    x = np.arange(len(sets))
    for ax, gk in zip(axes, gates):
        for klass, off, col in (("concept", -0.15, "C0"), ("random", 0.15, "C1")):
            y, lo, hi, ns = [], [], [], []
            for s in sets:
                cell = metrics["rank_matched"].get(s, {}).get("4", {}).get(klass, {}).get(gk)
                if not cell or not cell.get("M"):
                    y.append(np.nan); lo.append(np.nan); hi.append(np.nan); ns.append(0)
                    continue
                mp = cell["M"]["mean_p"]
                y.append(mp["m"]); lo.append(mp["lo95"]); hi.append(mp["hi95"])
                ns.append(cell["M"]["n_rows"])
            y = np.array(y, float)
            ax.errorbar(x + off, y,
                        yerr=[np.clip(y - np.array(lo, float), 0, None),
                              np.clip(np.array(hi, float) - y, 0, None)],
                        fmt="o", color=col, label=f"{klass} α=4", capsize=3)
            for i, n in enumerate(ns):
                if n:
                    ax.text(x[i] + off, y[i], f"n={n}", fontsize=6, ha="center", va="bottom")
        # noise floor
        yn = []
        for s in sets:
            nf = metrics["rank_matched"]["noise_floor"].get(s, {}).get(gk)
            yn.append(nf["mean_p"]["m"] if nf else np.nan)
        ax.plot(x, yn, "k--", marker="x", label="noise floor")
        for lv, ls in ((15, ":"), (20, ":"), (50, "-"), (75, ":"), (80, ":")):
            ax.axhline(lv, color="gray", ls=ls, lw=0.7)
        ax.set_xticks(x); ax.set_xticklabels(sets, fontsize=8)
        ax.set_title(gk); ax.set_ylabel("mean p (0=most suppressed)")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results / "fig_rankmatched_pctl.png", dpi=150)
    plt.close(fig)

    # fig_damping
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    colors = {s: f"C{i}" for i, s in enumerate(SET_ORDER)}
    for ax, gk in zip(axes, gates):
        for kind, marker in (("prop", "o"), ("affine", "s")):
            pairs = gate_pred.get(gk, {}).get(kind, [])
            if not pairs:
                continue
            # color by set unavailable in flat list — plot all
            act = [a for a, p in pairs if np.isfinite(a) and np.isfinite(p)]
            pred = [p for a, p in pairs if np.isfinite(a) and np.isfinite(p)]
            ax.scatter(pred, act, s=6, alpha=0.25, marker=marker, label=kind)
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", lw=0.8)
        ax.set_xlabel("predicted Δa"); ax.set_ylabel("actual Δa"); ax.set_title(gk)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results / "fig_damping.png", dpi=150)
    plt.close(fig)

    # fig_dla_share
    fig, ax = plt.subplots(figsize=(10, 4))
    width = 0.25
    for i, (name, key) in enumerate((
        ("gate", "gate_share_ros"), ("top20", "top20_share_ros"), ("damp", "damping_share_ros")
    )):
        y, lo, hi = [], [], []
        for s in sets:
            cell = metrics["dla"].get(s, {}).get("4", {}).get("concept", {}).get("L45")
            if not cell:
                y.append(np.nan); lo.append(np.nan); hi.append(np.nan); continue
            blk = cell[key]
            y.append(blk["m"]); lo.append(blk["lo95"]); hi.append(blk["hi95"])
        y = np.array(y, float)
        ax.bar(x + (i - 1) * width, y, width=width, yerr=[np.clip(y - np.array(lo, float), 0, None),
                                                          np.clip(np.array(hi, float) - y, 0, None)],
               label=f"{name} (L45)", capsize=2)
    ax.set_xticks(x); ax.set_xticklabels(sets)
    ax.set_ylabel("share (ratio of sums)")
    ax.set_title("DLA shares concept α=4 (L45)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(results / "fig_dla_share.png", dpi=150)
    plt.close(fig)

    # fig_rho_vs_rank
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    panels = [("S1", 45), ("S1", 50), ("S4", 45), ("S4", 50)]
    for ax, (s, L) in zip(axes.ravel(), panels):
        bins_data = rho_bins.get((s, L), [[] for _ in rank_bins_def])
        med = [float(np.median(b)) if b else np.nan for b in bins_data]
        q25 = [float(np.percentile(b, 25)) if b else np.nan for b in bins_data]
        q75 = [float(np.percentile(b, 75)) if b else np.nan for b in bins_data]
        xs = np.arange(len(labels))
        med_a = np.array(med, float)
        ax.errorbar(
            xs, med_a,
            yerr=[np.clip(med_a - np.array(q25, float), 0, None),
                  np.clip(np.array(q75, float) - med_a, 0, None)],
            fmt="o-", color="C0", capsize=3, label="median ρ",
        )
        for pts, col, lab, mk in (
            (rho_gate_pts.get((s, L), []), "C3", "gates", "*"),
            (rho_no_pts.get((s, L), []), "C1", "No-writers", "^"),
            (rho_yes_pts.get((s, L), []), "C2", "Yes-writers", "v"),
        ):
            if not pts:
                continue
            # map rank → bin center x
            xx, yy = [], []
            for rg, r in pts:
                for bi, (lo, hi) in enumerate(rank_bins_def):
                    if lo <= rg <= hi:
                        xx.append(bi); yy.append(r); break
            ax.scatter(xx, yy, s=18, c=col, marker=mk, alpha=0.5, label=lab)
        n_tot = sum(len(b) for b in bins_data)
        ax.set_title(f"{s} L{L} (n_ρ={n_tot})")
        ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=45, fontsize=7)
        ax.set_ylabel("ρ"); ax.axhline(0, color="k", lw=0.5)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(results / "fig_rho_vs_rank.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/wo_003a.yaml"))
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results/WO-003a"))
    ap.add_argument("--ledger", default=str(REPO_ROOT / "results/budget_ledger.csv"))
    ap.add_argument("--git-hash", default="local")
    ap.add_argument("--wo003-raw", default=None)
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--transcoder-root", default=None)
    a = ap.parse_args()
    run_wo_003a(
        config_path=a.config,
        results_dir=a.results_dir,
        ledger_path=a.ledger,
        git_hash_str=a.git_hash,
        wo003_raw=a.wo003_raw,
        model_dir=a.model_dir,
        transcoder_root=a.transcoder_root,
    )
