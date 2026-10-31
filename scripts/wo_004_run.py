"""WO-004: causal gate test via feature clamp / sham on transcoder features (GPU).

86 prompts × (base + concept/random intervention rows). One token sequence per forward
batch; row 0 is always unsteered base. See ``configs/wo_004.yaml``.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sfo.feature_clamp import (  # noqa: E402
    ClampPlan,
    FeatureBundle,
    apply_feature_clamp,
    identity_delta_check,
    load_feature_bundle,
    select_m_dla,
    select_m_norm,
)
from sfo.hooks import _resolve_layers  # noqa: E402
from sfo.ld import yes_no_token_ids  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import format_chat, get_d_model, get_layers, load_model, load_tokenizer  # noqa: E402
from sfo.prompts import (  # noqa: E402
    INTRO_ASSISTANT,
    INTRO_USER_1,
    YESBIAS_QUESTIONS,
    build_introspection_prompt,
    build_yesbias_prompt,
)
from sfo.provenance import collect_env  # noqa: E402
from sfo.transcoder import (  # noqa: E402
    encode_full,
    encode_gates_fp32,
    load_encode_weights,
    load_transcoder_config,
    sparse_from_acts,
)
from sfo.vectors import compute_start_pos, format_chat_text  # noqa: E402
from wo_003_run import (  # noqa: E402
    Logger,
    RunStop,
    SanityAbort,
    _chat_prompt,
    _load_boolq,
    _py,
)

SET_ORDER = ["S1", "S2v", "S3", "S4", "S8"]
EXPECTED_PROMPTS = 86
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
WIDTH = 262144


# ----------------------------------------------------------------------------------------
# Prompts (WO-003 S1/S2v/S3/S4 + S8)
# ----------------------------------------------------------------------------------------
@dataclass
class Prompt:
    pset: str
    idx: int
    ids: torch.Tensor
    start_pos: int
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[1])


def build_prompts(tokenizer: Any, cfg: dict) -> Tuple[List[Prompt], Dict[str, Any]]:
    prompts: List[Prompt] = []
    info: Dict[str, Any] = {"dataset_revisions": {}}
    ge1 = int(cfg.get("start_pos_ge1", 1))
    ns = [int(n) for n in cfg["s1_trial_ns"]]

    for i, n in enumerate(ns):
        text, ids, sp = build_introspection_prompt(tokenizer, n)
        prompts.append(Prompt("S1", i, ids.cpu(), int(sp), text, {"trial_n": n}))
    s2v = list(cfg["s2v"])
    assert len(s2v) == 4
    for i, entry in enumerate(s2v):
        mode = entry["mode"]
        if mode == "three_turn":
            msgs = [
                {"role": "user", "content": INTRO_USER_1},
                {"role": "assistant", "content": INTRO_ASSISTANT},
                {"role": "user", "content": entry["user3"]},
            ]
            meta_txt = entry["user3"]
        elif mode == "single_user":
            msgs = [{"role": "user", "content": entry["user"]}]
            meta_txt = entry["user"]
        else:
            raise ValueError(f"unknown S2v mode {mode!r}")
        text, ids = _chat_prompt(tokenizer, msgs)
        prompts.append(Prompt("S2v", i, ids, ge1, text, {"mode": mode, "paraphrase": meta_txt}))
    tn3 = int(cfg.get("s3_trial_n", 1))
    for i, q in enumerate(YESBIAS_QUESTIONS):
        text, ids, sp = build_yesbias_prompt(tokenizer, tn3, q)
        prompts.append(Prompt("S3", i, ids.cpu(), int(sp), text, {"question": q, "trial_n": tn3}))
    items, rev = _load_boolq(cfg)
    info["dataset_revisions"]["boolq"] = rev
    tmpl = cfg["boolq"]["user_template"]
    for i, it in enumerate(items):
        user = tmpl.format(q=it["question"])
        text, ids = _chat_prompt(tokenizer, [{"role": "user", "content": user}])
        prompts.append(Prompt("S4", i, ids, ge1, text, {"orig_idx": it["orig_idx"]}))
    for i, n in enumerate(cfg["s8_trial_ns"]):
        user = cfg["s8_user_template"].format(n=n)
        msgs = [
            {"role": "user", "content": INTRO_USER_1},
            {"role": "assistant", "content": INTRO_ASSISTANT},
            {"role": "user", "content": user},
        ]
        text, ids = _chat_prompt(tokenizer, msgs)
        sp = compute_start_pos(tokenizer, text, int(n))
        prompts.append(Prompt("S8", i, ids, int(sp), text, {"trial_n": int(n)}))

    assert len(prompts) == EXPECTED_PROMPTS, f"{len(prompts)} != {EXPECTED_PROMPTS}"
    for p in prompts:
        assert 0 <= p.start_pos < p.n_tokens - 1, (p.pset, p.idx, p.start_pos, p.n_tokens)
    return prompts, info


def interleave_prompts(prompts: Sequence[Prompt]) -> List[Prompt]:
    by = {s: [p for p in prompts if p.pset == s] for s in SET_ORDER}
    out: List[Prompt] = []
    k = 0
    while any(k < len(v) for v in by.values()):
        for s in SET_ORDER:
            if k < len(by[s]):
                out.append(by[s][k])
        k += 1
    return out


# ----------------------------------------------------------------------------------------
# DLA cache (CPU)
# ----------------------------------------------------------------------------------------
def _find_weight_keys(index: dict) -> Tuple[List[str], List[str], List[str]]:
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


def _load_rows_from_shard(model_dir: Path, shard: str, key: str, rows: Sequence[int]) -> np.ndarray:
    from safetensors import safe_open

    with safe_open(str(model_dir / shard), framework="pt", device="cpu") as f:
        if len(rows) == 0:
            return f.get_tensor(key).float().numpy()
        t = f.get_slice(key)
        out = [np.asarray(t[int(r)].float().numpy(), dtype=np.float64) for r in rows]
        return np.stack(out, axis=0)


def load_delta_u_and_gamma(model_dir: Path, yes_id: int, no_id: int) -> Dict[str, Any]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.exists():
        raise FileNotFoundError(f"missing {index_path}")
    index = json.loads(index_path.read_text())
    u_keys, tied_keys, norm_keys = _find_weight_keys(index)
    if len(norm_keys) != 1:
        raise RuntimeError(f"norm key count={len(norm_keys)}")
    if u_keys:
        uk = u_keys[0]
    elif tied_keys:
        uk = tied_keys[0]
    else:
        raise RuntimeError("no unembed / tied embed key")
    weight_map = index["weight_map"]
    rows = _load_rows_from_shard(model_dir, weight_map[uk], uk, [yes_id, no_id])
    u_yes, u_no = rows[0], rows[1]
    delta_u = u_no - u_yes
    gamma = _load_rows_from_shard(model_dir, weight_map[norm_keys[0]], norm_keys[0], [])
    if gamma.ndim != 1:
        gamma = gamma.reshape(-1)
    return {
        "delta_u": delta_u.astype(np.float64),
        "gamma": gamma.astype(np.float64),
        "delta_u_n": ((1.0 + gamma) * delta_u).astype(np.float64),
    }


def load_w_dec_stats(folder: Path, delta_u: np.ndarray, gamma: np.ndarray) -> Dict[str, np.ndarray]:
    from safetensors import safe_open

    du = delta_u.astype(np.float64)
    du_n = ((1.0 + gamma) * delta_u).astype(np.float64)
    dla_raw = np.empty(WIDTH, dtype=np.float64)
    dla_norm = np.empty(WIDTH, dtype=np.float64)
    norms = np.empty(WIDTH, dtype=np.float64)
    chunk = 4096
    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
        for i0 in range(0, WIDTH, chunk):
            i1 = min(WIDTH, i0 + chunk)
            w = f.get_slice("w_dec")[i0:i1].float().numpy().astype(np.float64)
            dla_raw[i0:i1] = w @ du
            dla_norm[i0:i1] = w @ du_n
            norms[i0:i1] = np.linalg.norm(w, axis=1)
    return {"dla_raw": dla_raw, "dla_normed": dla_norm, "w_norm": norms}


def build_or_load_dla_cache(
    model_dir: Path,
    transcoder_root: Path,
    cfg: dict,
    raw: Path,
    yes_id: int,
    no_id: int,
    slog: Logger,
) -> Dict[str, Any]:
    cache_path = raw / "dla_cache.pt"
    if cache_path.exists():
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    du_info = load_delta_u_and_gamma(model_dir, yes_id, no_id)
    gates = {int(k): [int(g) for g in v] for k, v in cfg["gates"].items()}
    exp = cfg["expected_gate_dla_normed"]
    per_layer: Dict[int, Dict[str, np.ndarray]] = {}
    for l in cfg["transcoder_layers"]:
        folder = transcoder_root / cfg["transcoder_subdir"].format(layer=int(l))
        stats = load_w_dec_stats(folder, du_info["delta_u"], du_info["gamma"])
        per_layer[int(l)] = stats
        for g in gates[int(l)]:
            key = f"L{l}_g{g}"
            got = float(stats["dla_normed"][g])
            want = float(exp[key])
            if abs(got - want) > 0.01:
                raise SanityAbort(f"gate dla_normed {key}: {got} vs expected {want}")
            slog(f"DLA cache L{l} g{g}: dla_normed={got:.6f} (expected {want})")
    pack = {"du_info": du_info, "layers": per_layer, "gates": gates}
    torch.save(pack, cache_path)
    return pack


# ----------------------------------------------------------------------------------------
# Row specs / batching
# ----------------------------------------------------------------------------------------
@dataclass
class RowSpec:
    row_kind: str  # base | concept | random
    condition: str
    concept: Optional[str] = None
    seed: Optional[int] = None
    vec: Optional[torch.Tensor] = None  # [d] bf16 CPU


def _tl_folder(root: Path, cfg: dict, layer: int) -> Path:
    return Path(root) / cfg["transcoder_subdir"].format(layer=layer)


def build_concept_vecs(path: Path, alpha: float) -> Tuple[List[str], Dict[str, torch.Tensor], Dict[str, float]]:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    concepts = list(pack["concepts"])
    steered: Dict[str, torch.Tensor] = {}
    norms: Dict[str, float] = {}
    for i, c in enumerate(concepts):
        raw = pack["vectors"][i].float()
        norms[c] = float(raw.norm())
        steered[c] = (raw * float(alpha)).to(torch.bfloat16)
    return concepts, steered, norms


def build_random_vecs(
    concepts: Sequence[str],
    concept_norms: Dict[str, float],
    seeds: Sequence[int],
    alpha: float,
    d_model: int,
) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for i, s in enumerate(seeds):
        c = concepts[i]
        target = float(alpha) * concept_norms[c]
        g = torch.Generator(device="cpu")
        g.manual_seed(int(s))
        r = torch.randn(d_model, generator=g, dtype=torch.float32)
        r = r / r.norm() * target
        out[int(s)] = r.to(torch.bfloat16)
    return out


def pack_groups(items: Sequence[Any], group_size: int, batch_size: int) -> List[List[Any]]:
    """Split items into batches of [complete groups]; caller prepends base row."""
    max_groups = (batch_size - 1) // group_size
    assert max_groups >= 1
    batches: List[List[Any]] = []
    i = 0
    while i < len(items):
        chunk = list(items[i : i + max_groups])
        assert len(chunk) * group_size + 1 <= batch_size
        batches.append(chunk)
        i += len(chunk)
    return batches


def _gate_ids(gates: Dict[int, List[int]], layer: int) -> List[int]:
    return [int(g) for g in gates[layer]]


def build_clamp_plan(
    B: int,
    rows: Sequence[RowSpec],
    controls: Dict[str, Any],
    gates: Dict[int, List[int]],
    *,
    layers_override: Optional[Tuple[int, ...]] = None,
) -> ClampPlan:
    assert len(rows) == B
    tr = torch.full((B,), -1, dtype=torch.long)
    feat = torch.zeros(B, dtype=torch.bool)
    mlp = torch.zeros(B, dtype=torch.bool)
    last_only = torch.zeros(B, dtype=torch.bool)
    fid_sets: List[Optional[List[int]]] = [None] * B
    layers = layers_override if layers_override is not None else (45, 50)

    row_layers: List[Optional[Tuple[int, ...]]] = [None] * B
    # Map each non-base row to the steer index of its own group (same concept/seed).
    steer_of: Dict[int, int] = {}
    last_steer_by_key: Dict[Tuple[str, Optional[str], Optional[int]], int] = {}
    for b, r in enumerate(rows):
        if r.row_kind == "base":
            continue
        key = (r.row_kind, r.concept, r.seed)
        if r.condition == "steer":
            last_steer_by_key[key] = b
        elif key in last_steer_by_key:
            steer_of[b] = last_steer_by_key[key]
    for b, r in enumerate(rows):
        if r.row_kind == "base":
            continue
        c = r.condition
        if c == "steer":
            continue
        if c == "clampG":
            feat[b] = True
            tr[b] = 0
            fid_sets[b] = _gate_ids(gates, 45) + _gate_ids(gates, 50)
        elif c == "clampG_last":
            feat[b] = True
            tr[b] = 0
            last_only[b] = True
            fid_sets[b] = _gate_ids(gates, 45) + _gate_ids(gates, 50)
        elif c == "clampG45":
            feat[b] = True
            tr[b] = 0
            fid_sets[b] = _gate_ids(gates, 45)
            row_layers[b] = (45,)
        elif c == "clampMdla":
            feat[b] = True
            tr[b] = 0
            md = controls["M_dla"]["45"] + controls["M_dla"]["50"]
            fid_sets[b] = md
        elif c == "clampMnorm":
            feat[b] = True
            tr[b] = 0
            mn = controls["M_norm"]["45"] + controls["M_norm"]["50"]
            fid_sets[b] = mn
        elif c == "clampMLP":
            mlp[b] = True
            tr[b] = 0
        elif c == "shamG":
            feat[b] = True
            assert b in steer_of, f"shamG row {b} has no in-group steer"
            tr[b] = int(steer_of[b])
            fid_sets[b] = _gate_ids(gates, 45) + _gate_ids(gates, 50)
        else:
            raise ValueError(c)
    return ClampPlan(
        target_row=tr,
        feature_mask=feat,
        mlp_mask=mlp,
        last_only=last_only,
        layers=layers,
        feature_id_sets=fid_sets,
        row_layers=row_layers,
    )


def union_bundle_ids(controls: Dict[str, Any], gates: Dict[int, List[int]], layer: int) -> List[int]:
    s = set(_gate_ids(gates, layer))
    s.update(int(f) for f in controls["M_dla"][str(layer)])
    s.update(int(f) for f in controls["M_norm"][str(layer)])
    return sorted(s)


# ----------------------------------------------------------------------------------------
# Forward
# ----------------------------------------------------------------------------------------
@dataclass
class FwdOut:
    logits: torch.Tensor  # [B,V]
    rms_final: torch.Tensor  # [B] fp32
    gate_act: Dict[int, torch.Tensor]  # layer -> [B,G]
    post_last: Dict[int, torch.Tensor]  # layer -> [B,d] fp32
    x_pre_last: Dict[int, torch.Tensor]
    acts_last: Dict[int, torch.Tensor]  # bundle acts [B,F]


@contextmanager
def _residual_batched_checked(
    model: Any, layer_idx: int, vectors: torch.Tensor, start_positions: torch.Tensor, T: int
) -> Iterator[None]:
    layers = _resolve_layers(model)
    layer = layers[layer_idx]
    state = {"abs_pos": 0}

    def hook(_module: Any, _input: Any, output: Any) -> Any:
        if isinstance(output, tuple):
            hidden, rest = output[0], output[1:]
        else:
            hidden, rest = output, None
        B, tlen, _D = hidden.shape
        device = hidden.device
        vecs = vectors.to(device=device, dtype=hidden.dtype)
        starts = start_positions.to(device=device, dtype=torch.long)
        abs_idx = torch.arange(state["abs_pos"], state["abs_pos"] + tlen, device=device, dtype=torch.long)
        mask = abs_idx.unsqueeze(0) >= starts.unsqueeze(1)
        modified = hidden + vecs.unsqueeze(1) * mask.unsqueeze(-1).to(hidden.dtype)
        state["abs_pos"] += tlen
        if rest is None:
            return modified
        return (modified,) + rest

    handle = layer.register_forward_hook(hook)
    try:
        yield
        assert state["abs_pos"] == T, f"abs_pos={state['abs_pos']} != T={T}"
    finally:
        handle.remove()


def _resolve_norm_module(model: Any) -> Any:
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "norm"):
            return lm.norm
    if hasattr(model, "model") and hasattr(model.model, "norm"):
        return model.model.norm
    raise AttributeError("model norm not found")


@torch.no_grad()
def forward_batch(
    model: Any,
    layers: Any,
    device: torch.device,
    ids1: torch.Tensor,
    start_pos: int,
    row_specs: Sequence[RowSpec],
    bundles: Dict[int, FeatureBundle],
    tw: Dict[int, Any],
    gates: Dict[int, List[int]],
    controls: Dict[str, Any],
    steer_layer: int,
    d_model: int,
    *,
    plan_layers: Optional[Tuple[int, ...]] = None,
    plan_override: Optional[ClampPlan] = None,
    supports_ltk: bool,
) -> Tuple[FwdOut, bool]:
    B = len(row_specs)
    assert B >= 1 and row_specs[0].row_kind == "base"
    ids = ids1.to(device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.expand(B, -1).contiguous()
    T = int(ids.shape[1])
    zd = torch.zeros(d_model, dtype=torch.bfloat16, device="cpu")
    vecs = torch.stack([(r.vec if r.vec is not None else zd).to(device) for r in row_specs])
    for b, r in enumerate(row_specs):
        if r.row_kind == "base" or r.condition == "shamG":
            vecs[b] = 0
    starts = torch.full((B,), int(start_pos), dtype=torch.long, device=device)

    plan = plan_override or build_clamp_plan(
        B, row_specs, controls, gates, layers_override=plan_layers
    )
    rms_holder: Dict[str, torch.Tensor] = {}
    cap_post: Dict[int, torch.Tensor] = {}
    cap_acts: Dict[int, torch.Tensor] = {}

    def norm_pre(_m: Any, args: Tuple[Any, ...]) -> None:
        rms_holder["x"] = args[0][:, -1, :].detach().float()
        rms_holder["rms"] = rms_holder["x"].pow(2).mean(dim=-1).sqrt()

    nh = _resolve_norm_module(model).register_forward_pre_hook(norm_pre)
    clamp_state: Dict[str, Any] = {}
    try:
        with apply_feature_clamp(
            model, bundles, plan, capture_post_out=cap_post, capture_acts_out=cap_acts
        ) as clamp_state:
            with _residual_batched_checked(model, steer_layer, vecs, starts, T):
                kw: Dict[str, Any] = dict(
                    input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
                )
                if supports_ltk:
                    kw["logits_to_keep"] = 1
                try:
                    out = model(**kw)
                except TypeError as e:
                    if "logits_to_keep" in str(e) and supports_ltk:
                        supports_ltk = False
                        kw.pop("logits_to_keep")
                        out = model(**kw)
                    else:
                        raise
        logits = out.logits[:, -1, :].detach()
    finally:
        nh.remove()

    gate_act: Dict[int, torch.Tensor] = {}
    x_pre_last: Dict[int, torch.Tensor] = {}
    for l in tw:
        x_last = clamp_state["x_pre"][l][:, -1, :].detach()
        x_pre_last[l] = x_last.float()
        _, act = encode_gates_fp32(x_last.unsqueeze(1), tw[l])
        gate_act[l] = act[:, 0].detach()

    post_last = {l: cap_post[l][:, -1, :].float().detach() for l in cap_post}
    acts_last = {l: cap_acts[l][:, -1, :].float().detach() for l in cap_acts}
    fo = FwdOut(
        logits=logits,
        rms_final=rms_holder["rms"].detach(),
        gate_act=gate_act,
        post_last=post_last,
        x_pre_last=x_pre_last,
        acts_last=acts_last,
    )
    return fo, supports_ltk


# ----------------------------------------------------------------------------------------
# Controls pre-pass
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def run_controls_for_prompt(
    model: Any,
    layers: Any,
    device: torch.device,
    prompt: Prompt,
    dla: Dict[str, Any],
    gates: Dict[int, List[int]],
    tw: Dict[int, Any],
    cfg: dict,
    set_index: int,
) -> Dict[str, Any]:
    ids = prompt.ids.to(device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    x_by_layer: Dict[int, torch.Tensor] = {}
    handles = []
    for l in cfg["transcoder_layers"]:
        li = int(l)

        def mk(layer_idx=li):
            def hook(_m: Any, _a: Any, output: torch.Tensor) -> None:
                x_by_layer[layer_idx] = output[:, -1, :].detach()

            return hook

        handles.append(layers[li].pre_feedforward_layernorm.register_forward_hook(mk()))
    kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    try:
        model(**kw)
    finally:
        for h in handles:
            h.remove()

    a0_by: Dict[int, Dict[int, float]] = {}
    for l in cfg["transcoder_layers"]:
        li = int(l)
        acts = encode_full(x_by_layer[li], tw[li])[0]
        idx, val = sparse_from_acts(acts)
        a0_by[li] = {int(i): float(v) for i, v in zip(idx.tolist(), val.tolist())}

    out: Dict[str, Any] = {
        "set": prompt.pset,
        "prompt_idx": prompt.idx,
        "M_dla": {"45": [], "50": []},
        "M_norm": {"45": [], "50": []},
    }
    k_max = int(cfg.get("k_max_controls", 50))
    for l in cfg["transcoder_layers"]:
        li = int(l)
        a0 = a0_by[li]
        stats = dla["layers"][li]
        md, kf_d, gt = select_m_dla(a0, stats["dla_raw"], gates[li], k_max=k_max)
        p50 = float(np.percentile(np.abs(stats["dla_normed"]), 50))
        rng = np.random.default_rng([0, set_index, prompt.idx, li])
        mn, kf_n, gt_n = select_m_norm(
            a0,
            stats["w_norm"],
            stats["dla_normed"],
            gates[li],
            exclude=md,
            p50_abs_dla=p50,
            rng=rng,
            k_max=k_max,
        )
        out["M_dla"][str(li)] = md
        out["M_norm"][str(li)] = mn
        out[f"L{li}"] = {
            "K_dla": len(md),
            "K_frac_dla": kf_d,
            "gate_total_dla": gt,
            "K_norm": len(mn),
            "K_frac_norm": kf_n,
            "gate_total_norm": gt_n,
            "p50_abs_dla_normed": p50,
            "n_a0": len(a0),
            "M_dla_detail": [
                {
                    "id": int(f),
                    "a0": float(a0[f]),
                    "dla_raw": float(stats["dla_raw"][f]),
                    "dla_normed": float(stats["dla_normed"][f]),
                }
                for f in md
            ],
            "M_norm_detail": [
                {
                    "id": int(f),
                    "a0": float(a0[f]),
                    "w_norm": float(stats["w_norm"][f]),
                    "dla_normed": float(stats["dla_normed"][f]),
                }
                for f in mn
            ],
            "gate_detail": [
                {
                    "id": int(g),
                    "a0": float(a0.get(g, 0.0)),
                    "dla_raw": float(stats["dla_raw"][g]),
                    "dla_normed": float(stats["dla_normed"][g]),
                    "w_norm": float(stats["w_norm"][g]),
                }
                for g in gates[li]
            ],
        }
    return out


# ----------------------------------------------------------------------------------------
# Decomposition
# ----------------------------------------------------------------------------------------
def _attach_decode(tw: Any, folder: Path, device: torch.device) -> None:
    from safetensors import safe_open

    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
        tw.w_dec = f.get_tensor("w_dec").to(device=device, dtype=torch.bfloat16)
        tw.b_dec = f.get_tensor("b_dec").to(device=device, dtype=torch.float32)
        if "affine_skip_connection" in set(f.keys()):
            tw.affine_skip = f.get_tensor("affine_skip_connection").to(
                device=device, dtype=torch.float32
            )


@torch.no_grad()
def decompose_row(
    x_pre: torch.Tensor,
    post: torch.Tensor,
    tw: Any,
    delta_u_n: np.ndarray,
    rms: float,
) -> Dict[str, Any]:
    """Decompose post_ff_ln into features + skip (W+A) + error; DLA-eq = proj on Δu_n / rms."""
    # Encode on GPU; assemble features/skip on CPU (w_dec/affine_skip live on CPU).
    acts = encode_full(x_pre.unsqueeze(0), tw)[0].detach().cpu()
    idx, val = sparse_from_acts(acts)
    post_c = post.detach().float().cpu()
    x_c = x_pre.detach().float().cpu()
    if idx.numel() == 0:
        feat = torch.zeros_like(post_c)
    else:
        wd = tw.w_dec[idx.long()].float().cpu()
        feat = (val.float().unsqueeze(0) @ wd).squeeze(0) + tw.b_dec.float().cpu()
    skip = torch.zeros_like(post_c)
    if tw.affine_skip is not None:
        skip = x_c @ tw.affine_skip.float().cpu()
    err = post_c - feat - skip
    du_n = torch.as_tensor(delta_u_n, dtype=torch.float32)
    denom = max(float(rms), 1e-12)

    def dla_eq(v: torch.Tensor) -> float:
        return float(v.float() @ du_n / denom)

    return {
        "features_dla_eq": _py(dla_eq(feat)),
        "skip_dla_eq": _py(dla_eq(skip)),
        "error_dla_eq": _py(dla_eq(err)),
        "rms_final": _py(rms),
        "error_norm": _py(float(err.norm())),
        "feat_norm": _py(float(feat.norm())),
        "skip_norm": _py(float(skip.norm())),
        "nnz": int(idx.numel()),
    }


# ----------------------------------------------------------------------------------------
# Sanity
# ----------------------------------------------------------------------------------------
def sanity_provenance(slog: Logger, git_hash_str: str, revision_sha: str, model_dir: str) -> None:
    slog("=== SANITY 1: provenance (caller) ===")
    if not git_hash_str or str(git_hash_str).strip().lower() in ("unknown", "none", ""):
        raise SanityAbort("SANITY1: git hash not supplied")
    if not revision_sha or len(str(revision_sha)) < 7:
        raise SanityAbort("SANITY1: model revision sha not supplied")
    slog(f"git={git_hash_str} revision={revision_sha} model_dir={model_dir}")
    slog("SANITY1 PASS")


def _max_logit_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def _run_logits(
    ctx: Dict[str, Any], p: Prompt, rows: List[RowSpec], plan: Optional[ClampPlan] = None
) -> torch.Tensor:
    fo, ctx["supports_ltk"] = forward_batch(
        ctx["model"],
        ctx["layers"],
        ctx["device"],
        p.ids,
        p.start_pos,
        rows,
        ctx["bundles"],
        ctx["tw"],
        ctx["gates"],
        ctx["ctrl"],
        ctx["steer_layer"],
        ctx["d_model"],
        plan_override=plan,
        supports_ltk=ctx["supports_ltk"],
    )
    return fo.logits


@torch.no_grad()
def sanity_identity(
    ctx: Dict[str, Any],
    prompts: Sequence[Prompt],
    concepts: Sequence[str],
    slog: Logger,
) -> None:
    slog("=== SANITY 2: identity (2 S1 prompts) ===")
    z = ctx["zero_vec"]
    cv = ctx["c_vecs"][concepts[0]]
    for p in [x for x in prompts if x.pset == "S1"][:2]:
        r_clamp_own = [
            RowSpec("base", "base", vec=z),
            RowSpec("concept", "clampG", concept=concepts[0], vec=cv),
        ]
        plan_own = build_clamp_plan(2, r_clamp_own, ctx["ctrl"], ctx["gates"])
        plan_own.target_row[1] = 1
        lg_a = _run_logits(ctx, p, r_clamp_own, plan_own)[1]
        lg_b = _run_logits(
            ctx,
            p,
            [
                RowSpec("base", "base", vec=z),
                RowSpec("concept", "steer", concept=concepts[0], vec=cv),
            ],
        )[1]
        if not torch.equal(lg_a, lg_b):
            raise SanityAbort(f"SANITY2 clamp_own: max={_max_logit_diff(lg_a, lg_b)}")
        slog(f"{p.pset}[{p.idx}] clamp target=own vs none: identical")

        lg_cg = _run_logits(
            ctx,
            p,
            [
                RowSpec("base", "base", vec=z),
                RowSpec("concept", "clampG", concept=concepts[0], vec=z),
            ],
        )[1]
        lg_base = _run_logits(ctx, p, [RowSpec("base", "base", vec=z)])[0]
        if not torch.equal(lg_cg, lg_base):
            raise SanityAbort("SANITY2 clampG unsteered vs base")
        slog(f"{p.pset}[{p.idx}] clampG unsteered vs base: identical")

        lg_mlp = _run_logits(
            ctx,
            p,
            [
                RowSpec("base", "base", vec=z),
                RowSpec("concept", "clampMLP", concept=concepts[0], vec=z),
            ],
        )[1]
        if not torch.equal(lg_mlp, lg_base):
            raise SanityAbort("SANITY2 clampMLP unsteered vs base")
        slog(f"{p.pset}[{p.idx}] clampMLP unsteered vs base: identical")

        r_sham = [
            RowSpec("base", "base", vec=z),
            RowSpec("concept", "steer", concept=concepts[0], vec=cv),
            RowSpec("concept", "shamG", concept=concepts[0], vec=cv),
        ]
        plan_sh = build_clamp_plan(3, r_sham, ctx["ctrl"], ctx["gates"])
        plan_sh.target_row[2] = 0
        lg_sh = _run_logits(ctx, p, r_sham, plan_sh)[2]
        if not torch.equal(lg_sh, lg_base):
            raise SanityAbort("SANITY2 sham source=base vs base")
        slog(f"{p.pset}[{p.idx}] sham source=base vs base: identical")
    slog("SANITY2 PASS")


@torch.no_grad()
def sanity_delta(
    ctx: Dict[str, Any],
    prompt: Prompt,
    concept: str,
    slog: Logger,
    rtol: float,
) -> None:
    slog("=== SANITY 3: delta check (1 S1, 1 concept, L45) ===")
    z = ctx["zero_vec"]
    cv = ctx["c_vecs"][concept]
    rows = [
        RowSpec("base", "base", vec=z),
        RowSpec("concept", "steer", concept=concept, vec=cv),
        RowSpec("concept", "clampG", concept=concept, vec=cv),
    ]
    fo, ctx["supports_ltk"] = forward_batch(
        ctx["model"],
        ctx["layers"],
        ctx["device"],
        prompt.ids,
        prompt.start_pos,
        rows,
        ctx["bundles"],
        ctx["tw"],
        ctx["gates"],
        ctx["ctrl"],
        ctx["steer_layer"],
        ctx["d_model"],
        supports_ltk=ctx["supports_ltk"],
    )
    l = 45
    bun = ctx["bundles"][l]
    gids = ctx["gates"][l]
    cols = [bun.feature_ids.index(int(g)) for g in gids]
    w_g = bun.w_dec[cols]
    ok, rel = identity_delta_check(
        fo.post_last[l][2],
        fo.post_last[l][1],
        fo.acts_last[l][0][cols],
        fo.acts_last[l][1][cols],
        w_g,
        rtol=rtol,
    )
    slog(f"L45 rel={rel:.6f} ok={ok}")
    if not ok:
        raise SanityAbort(f"SANITY3: delta rel {rel} > {rtol}")
    slog("SANITY3 PASS")


def load_wo003_ld(path: Path) -> Dict[Tuple[str, int, str], float]:
    out: Dict[Tuple[str, int, str], float] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out[(r["set"], int(r["prompt_idx"]), r["cid"])] = float(r["ld"])
    return out


def sanity_replication(
    ld_map: Dict[Tuple[str, int, str], float],
    set_name: str,
    prompt_idx: int,
    ld_val: float,
    concept: str,
    alpha: float,
    max_abs: float,
) -> Optional[float]:
    cid = f"concept__{concept}__a{alpha:g}"
    key = (set_name, prompt_idx, cid)
    if key not in ld_map:
        return None
    diff = abs(ld_val - ld_map[key])
    if diff > max_abs:
        raise RunStop(f"SANITY4: |LD-LD003|={diff} > {max_abs} for {key}")
    return diff


def sanity_readout(
    ctx: Dict[str, Any],
    prompts: Sequence[Prompt],
    slog: Logger,
    concepts: Sequence[str],
) -> None:
    slog("=== SANITY 6: readout top-5 ===")
    tok = ctx["tokenizer"]
    z = ctx["zero_vec"]
    yes_id, no_id = yes_no_token_ids(tok)
    # Every S8 base; 2 prompts per other set.
    for s in SET_ORDER:
        take = [x for x in prompts if x.pset == s]
        if s != "S8":
            take = take[:2]
        for p in take:
            lg = _run_logits(ctx, p, [RowSpec("base", "base", vec=z)])[0].float()
            top = torch.topk(lg, 5)
            tops = [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
            ids = p.ids[0].tolist()
            slog(
                f"{p.pset}[{p.idx}] base readout={tok.decode([ids[-1]])!r} top5={tops}"
            )
    # steer / clampG / clampMLP / shamG of first concept on 1 S1 prompt
    p = next(x for x in prompts if x.pset == "S1")
    c0 = concepts[0]
    cv = ctx["c_vecs"][c0]
    rows_ord = [
        RowSpec("base", "base", vec=z),
        RowSpec("concept", "steer", concept=c0, vec=cv),
        RowSpec("concept", "clampG", concept=c0, vec=cv),
        RowSpec("concept", "clampMLP", concept=c0, vec=cv),
        RowSpec("concept", "shamG", concept=c0, vec=cv),
    ]
    fo, ctx["supports_ltk"] = forward_batch(
        ctx["model"],
        ctx["layers"],
        ctx["device"],
        p.ids,
        p.start_pos,
        rows_ord,
        ctx["bundles"],
        ctx["tw"],
        ctx["gates"],
        ctx["ctrl"],
        ctx["steer_layer"],
        ctx["d_model"],
        supports_ltk=ctx["supports_ltk"],
    )
    for b, spec in enumerate(rows_ord):
        if spec.row_kind == "base":
            continue
        lg = fo.logits[b].float()
        top = torch.topk(lg, 5)
        tops = [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
        ld = float(lg[no_id] - lg[yes_id])
        slog(f"S1[0] concept={c0} {spec.condition} LD={ld:.4f} top5={tops}")


def sanity_s8_base_ld(ctx: Dict[str, Any], prompts: Sequence[Prompt], slog: Logger) -> None:
    slog("=== SANITY 7: S8 default-Yes base LD (all 6) ===")
    z = ctx["zero_vec"]
    tok = ctx["tokenizer"]
    yes_id, no_id = yes_no_token_ids(tok)
    for p in [x for x in prompts if x.pset == "S8"]:
        lg = _run_logits(ctx, p, [RowSpec("base", "base", vec=z)])[0].float()
        ld = float(lg[no_id] - lg[yes_id])
        slog(f"S8[{p.idx}] base LD={ld:.6f}")


def _bundle_act_map(fo: FwdOut, bundles: Dict[int, FeatureBundle], b: int) -> Dict[int, Dict[int, float]]:
    """layer -> {feature_id: activation} from last-position bundle acts."""
    out: Dict[int, Dict[int, float]] = {}
    for l, acts in fo.acts_last.items():
        bun = bundles.get(l)
        if bun is None or not bun.feature_ids:
            continue
        row = acts[b].cpu().tolist()
        out[l] = {int(fid): float(row[j]) for j, fid in enumerate(bun.feature_ids)}
    return out


def row_log_record(
    p: Prompt,
    spec: RowSpec,
    fo: FwdOut,
    b: int,
    yes_id: int,
    no_id: int,
    tokenizer: Any,
    batch_size: int,
    *,
    bundles: Optional[Dict[int, FeatureBundle]] = None,
    controls: Optional[Dict[str, Any]] = None,
    gates: Optional[Dict[int, List[int]]] = None,
) -> Dict[str, Any]:
    lg = fo.logits[b].float()
    ld = float(lg[no_id] - lg[yes_id])
    top = torch.topk(lg, 5)
    tops = [{"id": int(i), "tok": tokenizer.decode([int(i)]), "logit": _py(float(v))}
            for v, i in zip(top.values, top.indices)]
    rec: Dict[str, Any] = {
        "set": p.pset,
        "prompt_idx": p.idx,
        "row_kind": spec.row_kind,
        "condition": spec.condition,
        "batch_size": batch_size,
        "ld": _py(ld),
        "logit_yes": _py(float(lg[yes_id])),
        "logit_no": _py(float(lg[no_id])),
        "top5": tops,
        "rms_final": _py(float(fo.rms_final[b])),
    }
    if spec.concept is not None:
        rec["concept"] = spec.concept
    if spec.seed is not None:
        rec["seed"] = spec.seed
    for l in fo.gate_act:
        rec[f"gate_act_L{l}"] = _py(fo.gate_act[l][b].cpu().tolist())
    # For base/steer: also log gate + M_dla feature acts (for Eff_G / Eff_Mdla).
    if (
        bundles is not None
        and controls is not None
        and gates is not None
        and (spec.row_kind == "base" or spec.condition == "steer")
    ):
        amap = _bundle_act_map(fo, bundles, b)
        feat_log: Dict[str, Dict[str, float]] = {}
        for l in amap:
            want = set(_gate_ids(gates, l)) | set(int(f) for f in controls["M_dla"][str(l)])
            feat_log[f"L{l}"] = {str(fid): amap[l][fid] for fid in want if fid in amap[l]}
        rec["feat_act"] = feat_log
    return rec


def build_group_rows(
    z: torch.Tensor,
    group_kind: str,
    conds: Sequence[str],
    *,
    concept: Optional[str] = None,
    seed: Optional[int] = None,
    vec: Optional[torch.Tensor] = None,
) -> List[RowSpec]:
    rows: List[RowSpec] = [RowSpec("base", "base", vec=z)]
    for c in conds:
        rows.append(
            RowSpec(group_kind, c, concept=concept, seed=seed, vec=vec if c != "shamG" else vec)
        )
    return rows


def assert_batch_integrity(rows: List[RowSpec], batch_size: int, group_size: int) -> None:
    assert rows[0].row_kind == "base" and rows[0].condition == "base"
    assert len(rows) <= batch_size
    rest = len(rows) - 1
    assert rest % group_size == 0, (len(rows), group_size)


def run_wo_004(
    model_dir,
    revision_sha,
    config_path,
    results_dir,
    ledger_path,
    transcoder_root,
    wo001_vectors_path,
    wo003_logits_path,
    git_hash_str,
    gpu_type="A100-80GB",
    n_cores=8.0,
    mem_gib=64.0,
    volume_commit=None,
) -> dict:
    t_run0 = time.perf_counter()
    results_path = Path(results_dir)
    raw = results_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    slog = Logger(raw / "sanity_log.txt")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    summary: Dict[str, Any] = {
        "wo_id": cfg.get("wo_id", "WO-004"),
        "git_hash": git_hash_str,
        "model_revision": revision_sha,
        "status": "RUNNING",
        "errors": [],
        "batch_size": int(cfg["batch_size"]),
    }

    def write_summary() -> None:
        (results_path / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    f_rows = (raw / "rows.jsonl").open("w")
    f_ctrl = (raw / "controls.jsonl").open("w")
    f_decomp = (raw / "decomposition.jsonl").open("w")

    with gpu_ledger(
        ledger_path,
        wo_id="WO-004",
        function="wo_004_run",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            sanity_provenance(slog, git_hash_str, revision_sha, str(model_dir))
            transcoder_root = Path(transcoder_root)
            alpha = float(cfg["alpha"])
            bs = int(cfg["batch_size"])
            gates = {int(k): [int(g) for g in v] for k, v in cfg["gates"].items()}
            yes_id_cfg, no_id_cfg = int(cfg["yes_id"]), int(cfg["no_id"])
            ld003 = load_wo003_ld(Path(wo003_logits_path))

            concepts, c_vecs, c_norms = build_concept_vecs(Path(wo001_vectors_path), alpha)
            assert len(concepts) == int(cfg["n_concepts"]) == 50
            rand_seeds = [int(s) for s in cfg["random_seeds"]]
            assert len(rand_seeds) == 20

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            prompts, pinfo = build_prompts(tokenizer, cfg)
            summary["dataset_revisions"] = pinfo.get("dataset_revisions", {})
            summary["prompt_counts"] = {s: sum(1 for p in prompts if p.pset == s) for s in SET_ORDER}
            with (raw / "prompts.jsonl").open("w") as fp:
                for p in prompts:
                    fp.write(
                        json.dumps(
                            {
                                "set": p.pset,
                                "prompt_idx": p.idx,
                                "start_pos": p.start_pos,
                                "n_tokens": p.n_tokens,
                                "text": p.text,
                                "ids": p.ids[0].tolist(),
                                "meta": p.meta,
                            }
                        )
                        + "\n"
                    )

            model = load_model(model_dir, revision=revision_sha, device=0)
            device = next(model.parameters()).device
            d_model = get_d_model(model)
            layers = get_layers(model)
            yes_id, no_id = yes_no_token_ids(tokenizer)
            assert yes_id == yes_id_cfg and no_id == no_id_cfg
            summary["env"] = collect_env()
            slog(f"yes_id={yes_id} no_id={no_id}; LD = logit(No)-logit(Yes)")

            dla_cache = build_or_load_dla_cache(
                Path(model_dir), transcoder_root, cfg, raw, yes_id, no_id, slog
            )
            du_n = dla_cache["du_info"]["delta_u_n"]

            tw: Dict[int, Any] = {}
            tap_layers = [int(l) for l in cfg["transcoder_layers"]]
            for l in tap_layers:
                folder = _tl_folder(transcoder_root, cfg, l)
                load_transcoder_config(folder)
                # Encode on GPU; decode/skip on CPU to leave headroom next to the 27B model.
                tw[l] = load_encode_weights(folder, gates[l], device, load_decode=False)
                _attach_decode(tw[l], folder, device="cpu")
            decode_tw = tw

            zero_vec = torch.zeros(d_model, dtype=torch.bfloat16)
            rand_vecs = build_random_vecs(concepts, c_norms, rand_seeds, alpha, d_model)
            steer_layer = int(cfg["layer_steer"])
            supports_ltk = True

            set_to_index = {s: i for i, s in enumerate(SET_ORDER)}
            controls_by: Dict[Tuple[str, int], Dict[str, Any]] = {}
            for p in prompts:
                ctrl = run_controls_for_prompt(
                    model,
                    layers,
                    device,
                    p,
                    dla_cache,
                    gates,
                    tw,
                    cfg,
                    set_to_index[p.pset],
                )
                controls_by[(p.pset, p.idx)] = ctrl
                f_ctrl.write(json.dumps(_py(ctrl)) + "\n")
            f_ctrl.flush()

            # Bundles per prompt (union gate + M_dla + M_norm)
            bundles_by_prompt: Dict[Tuple[str, int], Dict[int, FeatureBundle]] = {}
            for p in prompts:
                ctrl = controls_by[(p.pset, p.idx)]
                bb: Dict[int, FeatureBundle] = {}
                for l in tap_layers:
                    fids = union_bundle_ids(ctrl, gates, l)
                    bb[l] = load_feature_bundle(_tl_folder(transcoder_root, cfg, l), fids, device)
                bundles_by_prompt[(p.pset, p.idx)] = bb

            s1p = next(p for p in prompts if p.pset == "S1")
            ctrl_ref = controls_by[(s1p.pset, s1p.idx)]
            ctx: Dict[str, Any] = {
                "model": model,
                "layers": layers,
                "device": device,
                "tokenizer": tokenizer,
                "tw": tw,
                "gates": gates,
                "ctrl": ctrl_ref,
                "bundles": bundles_by_prompt[(s1p.pset, s1p.idx)],
                "steer_layer": steer_layer,
                "d_model": d_model,
                "zero_vec": zero_vec,
                "c_vecs": c_vecs,
                "supports_ltk": supports_ltk,
            }

            sanity_identity(ctx, prompts, concepts, slog)
            sanity_delta(ctx, s1p, concepts[0], slog, float(cfg["delta_check_rtol"]))
            sanity_readout(ctx, prompts, slog, concepts)
            sanity_s8_base_ld(ctx, prompts, slog)

            slog("=== SANITY 8: controls (first 3 S1) ===")
            for p in [x for x in prompts if x.pset == "S1"][:3]:
                c = controls_by[(p.pset, p.idx)]
                for li in tap_layers:
                    lk = f"L{li}"
                    detail = c.get(lk, {})
                    slog(
                        f"{p.pset}[{p.idx}] {lk}: K_dla={detail.get('K_dla')} "
                        f"frac={detail.get('K_frac_dla'):.4f} "
                        f"K_norm={detail.get('K_norm')} "
                        f"gate={detail.get('gate_detail')} "
                        f"M_dla={detail.get('M_dla_detail')} "
                        f"M_norm_ids={[d['id'] for d in detail.get('M_norm_detail', [])]}"
                    )

            rep_max = 0.0
            order = interleave_prompts(prompts)
            n_total = len(order)
            n_check = max(1, int(math.ceil(float(cfg["projection_check_fraction"]) * n_total)))
            limit_s = float(cfg["projected_gpu_limit_hours"]) * 3600.0
            t_main0 = time.perf_counter()
            n_rows = 0
            max_rep = float(cfg["replication_max_abs_ld"])
            decomp_logged: set = set()  # (set, prompt_idx, row_kind, concept|seed, layer)

            concept_batches = pack_groups(concepts, len(CONCEPT_CONDS), bs)
            random_batches = pack_groups(rand_seeds, len(RANDOM_CONDS), bs)
            slog(
                f"batch_size={bs}; concept batches={len(concept_batches)} "
                f"(≤{(bs-1)//len(CONCEPT_CONDS)} groups/fwd); "
                f"random batches={len(random_batches)}"
            )
            slog("SANITY5: group integrity asserted every batch (base first; groups unsplit)")

            def _write_decomp(fo: FwdOut, rows_local: List[RowSpec], prompt: Prompt) -> None:
                for b, spec in enumerate(rows_local):
                    if not (spec.row_kind == "base" or spec.condition == "steer"):
                        continue
                    rms = float(fo.rms_final[b])
                    for l in tap_layers:
                        key = (
                            prompt.pset,
                            prompt.idx,
                            spec.row_kind,
                            spec.concept,
                            spec.seed,
                            l,
                        )
                        if key in decomp_logged:
                            continue
                        decomp_logged.add(key)
                        dec = decompose_row(
                            fo.x_pre_last[l][b],
                            fo.post_last[l][b],
                            decode_tw[l],
                            du_n,
                            rms,
                        )
                        f_decomp.write(
                            json.dumps(
                                _py(
                                    {
                                        "set": prompt.pset,
                                        "prompt_idx": prompt.idx,
                                        "row_kind": spec.row_kind,
                                        "condition": spec.condition,
                                        "concept": spec.concept,
                                        "seed": spec.seed,
                                        "layer": l,
                                        **dec,
                                    }
                                )
                            )
                            + "\n"
                        )

            for pi, p in enumerate(order):
                ctrl = controls_by[(p.pset, p.idx)]
                bb = bundles_by_prompt[(p.pset, p.idx)]
                z = zero_vec
                base_logged = False

                for cb in concept_batches:
                    rows: List[RowSpec] = [RowSpec("base", "base", vec=z)]
                    for c in cb:
                        rows.extend(
                            build_group_rows(z, "concept", CONCEPT_CONDS, concept=c, vec=c_vecs[c])[1:]
                        )
                    assert_batch_integrity(rows, bs, len(CONCEPT_CONDS))
                    fo, supports_ltk = forward_batch(
                        model,
                        layers,
                        device,
                        p.ids,
                        p.start_pos,
                        rows,
                        bb,
                        tw,
                        gates,
                        ctrl,
                        steer_layer,
                        d_model,
                        supports_ltk=supports_ltk,
                    )
                    for b, spec in enumerate(rows):
                        if spec.row_kind == "base" and base_logged:
                            continue
                        rec = row_log_record(
                            p,
                            spec,
                            fo,
                            b,
                            yes_id,
                            no_id,
                            tokenizer,
                            bs,
                            bundles=bb,
                            controls=ctrl,
                            gates=gates,
                        )
                        f_rows.write(json.dumps(rec) + "\n")
                        n_rows += 1
                        if spec.row_kind == "base":
                            base_logged = True
                            key_b = (p.pset, p.idx, "baseline")
                            if key_b in ld003:
                                d_b = abs(float(rec["ld"]) - ld003[key_b])
                                rep_max = max(rep_max, d_b)
                                if d_b > max_rep:
                                    raise RunStop(
                                        f"SANITY4: base LD diff {d_b} > {max_rep} for {key_b}"
                                    )
                        if spec.condition == "steer" and spec.row_kind == "concept":
                            d = sanity_replication(
                                ld003, p.pset, p.idx, float(rec["ld"]), spec.concept, alpha, max_rep
                            )
                            if d is not None:
                                rep_max = max(rep_max, d)
                    _write_decomp(fo, rows, p)

                for rb in random_batches:
                    rows = [RowSpec("base", "base", vec=z)]
                    for s in rb:
                        rows.extend(
                            build_group_rows(
                                z, "random", RANDOM_CONDS, seed=int(s), vec=rand_vecs[int(s)]
                            )[1:]
                        )
                    assert_batch_integrity(rows, bs, len(RANDOM_CONDS))
                    fo, supports_ltk = forward_batch(
                        model,
                        layers,
                        device,
                        p.ids,
                        p.start_pos,
                        rows,
                        bb,
                        tw,
                        gates,
                        ctrl,
                        steer_layer,
                        d_model,
                        supports_ltk=supports_ltk,
                    )
                    for b, spec in enumerate(rows):
                        if spec.row_kind == "base" and base_logged:
                            continue
                        rec = row_log_record(
                            p,
                            spec,
                            fo,
                            b,
                            yes_id,
                            no_id,
                            tokenizer,
                            bs,
                            bundles=bb,
                            controls=ctrl,
                            gates=gates,
                        )
                        f_rows.write(json.dumps(rec) + "\n")
                        n_rows += 1
                        if spec.row_kind == "base":
                            base_logged = True
                    _write_decomp(fo, rows, p)

                if (pi + 1) == n_check:
                    per = (time.perf_counter() - t_main0) / (pi + 1)
                    proj = (t_main0 - t_run0) + per * n_total
                    summary["projection"] = {
                        "prompts_timed": pi + 1,
                        "sec_per_prompt": per,
                        "projected_total_hours": proj / 3600.0,
                        "limit_hours": limit_s / 3600.0,
                    }
                    slog(
                        f"=== SANITY 9: projected time after {pi+1}/{n_total}: "
                        f"{proj/3600:.2f} h (limit {limit_s/3600:.2f} h) ==="
                    )
                    if proj > limit_s:
                        raise RunStop(f"SANITY9: projected {proj/3600:.2f}h > limit")

                if (pi + 1) % int(cfg.get("checkpoint_every_prompts", 20)) == 0:
                    f_rows.flush()
                    f_decomp.flush()
                    write_summary()
                    if volume_commit:
                        try:
                            volume_commit()
                        except Exception as ve:  # noqa: BLE001
                            slog(f"volume_commit: {ve}")

            f_rows.flush()
            f_ctrl.flush()
            f_decomp.flush()
            summary["replication_max_abs_ld"] = rep_max
            summary["n_rows"] = n_rows
            summary["n_decomp"] = len(decomp_logged)
            summary["transcoder_revision"] = cfg.get("transcoder_revision")
            summary["total_seconds"] = time.perf_counter() - t_run0
            summary["status"] = "COMPLETE"
            expected_rows = EXPECTED_PROMPTS * (1 + 50 * len(CONCEPT_CONDS) + 20 * len(RANDOM_CONDS))
            summary["expected_n_rows"] = expected_rows
            if n_rows != expected_rows:
                slog(f"WARN n_rows={n_rows} != expected {expected_rows}")
            slog(f"DONE rows={n_rows} rep_max={rep_max:.4f} t={summary['total_seconds']/3600:.2f}h")
            write_summary()
            return summary

        except RunStop as e:
            slog(f"STOP: {e}")
            summary["status"] = f"STOPPED: {e}"
            summary["errors"].append(str(e))
            write_summary()
            return summary
        except Exception as e:  # noqa: BLE001
            slog(f"ABORT: {type(e).__name__}: {e}")
            slog(traceback.format_exc())
            summary["status"] = f"ABORTED: {type(e).__name__}: {e}"
            summary["errors"].append(str(e))
            write_summary()
            raise
        finally:
            f_rows.close()
            f_ctrl.close()
            f_decomp.close()
            slog.close()


if __name__ == "__main__":
    run_wo_004(
        model_dir=os.environ["SFO_MODEL_DIR"],
        revision_sha=os.environ["SFO_REVISION_SHA"],
        config_path=os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_004.yaml")),
        results_dir=os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-004")),
        ledger_path=os.environ.get("SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")),
        transcoder_root=os.environ["SFO_TRANSCODER_ROOT"],
        wo001_vectors_path=str(REPO_ROOT / "results/WO-001/raw/concept_vectors_L37.pt"),
        wo003_logits_path=str(REPO_ROOT / "results/WO-003/raw/logits_ld.jsonl"),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )