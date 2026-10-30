"""WO-003 main GPU experiment: transcoder-gate readouts under concept / random steering.

143 conditions x 266 prompts. For every (prompt, condition) row we record, at the readout
(last) token: sparse JumpReLU transcoder acts at L45 / L50, gate pre/act/threshold (fp32),
mean gate pre/act over the last 20 tokens, direct-path gate shift, residual reach /
reach_indirect / dirproj, and (full-forward sets only) LD = logit(No) - logit(Yes).

Conventions
-----------
* LD sign is FLIPPED relative to WO-001: LD = logit(No) - logit(Yes).
* One prompt per logical batch: all rows of a chunk share identical token ids (no padding)
  and differ only in the steering vector. A fresh ``add_to_residual_batched`` context is
  opened for every forward.
* S5 / S6 / S7 stop right after layer-50 ``pre_feedforward_layernorm`` via a private
  exception (``_EarlyHalt``); S1-S4 run the full forward with last-token logits.
* Outputs go under ``<results_dir>/raw`` (``run_summary.json`` under ``<results_dir>``).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.data import (  # noqa: E402
    format_mmlu_user,
    load_mmlu_validation,
    resolve_dataset_revision,
)
from sfo.hooks import add_to_residual_batched  # noqa: E402
from sfo.ld import yes_no_token_ids  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import (  # noqa: E402
    format_chat,
    get_d_model,
    get_layers,
    load_model,
    load_tokenizer,
)
from sfo.prompts import (  # noqa: E402
    INTRO_ASSISTANT,
    INTRO_USER_1,
    YESBIAS_QUESTIONS,
    build_introspection_prompt,
    build_yesbias_prompt,
)
from sfo.provenance import collect_env  # noqa: E402
from sfo.transcoder import (  # noqa: E402
    TranscoderWeights,
    encode_full,
    encode_gates_fp32,
    load_encode_weights,
    load_transcoder_config,
    sparse_from_acts,
)
from sfo.vectors import format_chat_text  # noqa: E402

SET_ORDER = ["S1", "S2", "S2v", "S3", "S4", "S5", "S6", "S7"]
FULL_SETS = {"S1", "S2", "S2v", "S3", "S4"}      # full forward + last-token logits
EARLY_SETS = {"S5", "S6", "S7"}                  # early halt after layer-50 pre_ff LN
EXPECTED_PROMPTS = 266
EXPECTED_CONDITIONS = 143


# ----------------------------------------------------------------------------------------
# Exceptions / logging
# ----------------------------------------------------------------------------------------
class _EarlyHalt(Exception):
    """Private: raised from the layer-50 pre_feedforward_layernorm hook to stop the forward."""


class SanityAbort(RuntimeError):
    """A sanity check failed and the run must abort (recorded as an error)."""


class RunStop(RuntimeError):
    """A STOP condition fired (variance explained / L0 / projected time)."""


class Logger:
    """Print + append to sanity_log.txt (flush every line)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.path.open("a", buffering=1)

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self._f.write(str(msg) + "\n")
        self._f.flush()

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:  # noqa: BLE001
            pass


def _py(x: Any) -> Any:
    """Tensor / numpy / float -> JSON-safe python (NaN/inf -> None)."""
    if isinstance(x, torch.Tensor):
        x = x.detach().float().cpu().tolist()
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, dict):
        return {k: _py(v) for k, v in x.items()}
    if isinstance(x, float):
        return None if (math.isnan(x) or math.isinf(x)) else x
    return x


# ----------------------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------------------
@dataclass
class Prompt:
    pset: str
    idx: int
    ids: torch.Tensor          # [1, T] long, CPU
    start_pos: int             # absolute index (no padding) where steering begins
    mode: str                  # "full" | "early"
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return int(self.ids.shape[1])


@dataclass
class Cond:
    cid: str
    klass: str                 # baseline | noise_floor | concept | random
    alpha: float
    norm: float                # ||applied vector|| (= alpha * ||v_c|| for concept / random)
    vec: torch.Tensor          # [d] bf16 CPU, the vector actually added
    concept: Optional[str] = None
    seed: Optional[int] = None
    matched_concept: Optional[str] = None
    chunk: int = -1
    slot: int = -1


@dataclass
class Baseline:
    pre_last: Dict[int, torch.Tensor]    # layer -> [G] fp32, pre_g(row) at readout
    pre_ln_m0: Dict[int, torch.Tensor]   # layer -> [G] fp32, pre_g(LN(m0))
    m0: Dict[int, torch.Tensor]          # layer -> [d] bf16 residual m at readout
    h0: Dict[int, torch.Tensor]          # resid layer -> [d] fp32 residual output at readout


@dataclass
class Capture:
    x: Dict[int, torch.Tensor] = field(default_factory=dict)       # LN out, [B,k,d]
    m_last: Dict[int, torch.Tensor] = field(default_factory=dict)  # LN in (m), [B,d]
    m_full: Dict[int, torch.Tensor] = field(default_factory=dict)  # only if keep_full
    h_last: Dict[int, torch.Tensor] = field(default_factory=dict)  # resid out, [B,d]
    h_full: Dict[int, torch.Tensor] = field(default_factory=dict)  # only if keep_full
    logits: Optional[torch.Tensor] = None                          # [B,V] last-token
    halted: bool = False


@dataclass
class Ctx:
    model: Any
    layers: Any
    tokenizer: Any
    device: torch.device
    tw: Dict[int, TranscoderWeights]
    d_hat: Dict[int, torch.Tensor]          # resid layer -> [d] fp32 unit direction (device)
    yes_id: int
    no_id: int
    steer_layer: int
    gates: Dict[int, List[int]]
    tap_layers: Tuple[int, ...]             # (45, 50)
    resid_for: Dict[int, int]               # {45: 44, 50: 49}
    halt_layer: int
    tail: int
    log: Callable[[str], None]
    supports_ltk: bool = True
    consec_fail: int = 0


# ----------------------------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------------------------
def _cname_concept(c: str, a: float) -> str:
    return f"concept__{c}__a{a:g}"


def _cname_random(seed: int, a: float) -> str:
    return f"random__s{seed}__a{a:g}"


def build_conditions(
    concepts: Sequence[str],
    vec50: Dict[str, torch.Tensor],
    alphas: Sequence[float],
    rand_seeds: Sequence[int],
    d_model: int,
    batch_size: int,
    n_noise: int = 2,
) -> Tuple[List[List[Cond]], Dict[str, Cond]]:
    """Return (chunks, by_cid). Layout: baseline = chunk 0 slot 0; noise-floor rows in a
    middle chunk (slot 0) and the last chunk (last slot). 1 + 2 + 100 + 40 = 143 rows."""
    assert n_noise == 2, "layout supports exactly 2 noise-floor rows"
    zero = torch.zeros(d_model, dtype=torch.bfloat16)
    baseline = Cond("baseline", "baseline", 0.0, 0.0, zero.clone())
    nf1 = Cond("noise_floor_1", "noise_floor", 0.0, 0.0, zero.clone())
    nf2 = Cond("noise_floor_2", "noise_floor", 0.0, 0.0, zero.clone())

    body: List[Cond] = []
    for a in alphas:
        for c in concepts:
            v = vec50[c].float() * float(a)
            body.append(
                Cond(_cname_concept(c, a), "concept", float(a), float(v.norm()),
                     v.to(torch.bfloat16), concept=c)
            )
    for a in alphas:
        for i, s in enumerate(rand_seeds):
            c_i = concepts[i]
            g = torch.Generator(device="cpu")
            g.manual_seed(int(s))
            r = torch.randn(d_model, generator=g, dtype=torch.float32)
            target = float(a) * float(vec50[c_i].float().norm())
            r = r / r.norm() * target
            body.append(
                Cond(_cname_random(s, a), "random", float(a), float(r.norm()),
                     r.to(torch.bfloat16), seed=int(s), matched_concept=c_i)
            )

    n = len(body) + 3
    n_chunks = int(math.ceil(n / batch_size))
    sizes = [batch_size] * (n_chunks - 1) + [n - batch_size * (n_chunks - 1)]
    mid = n_chunks // 2
    last = n_chunks - 1
    chunks: List[List[Cond]] = [[] for _ in range(n_chunks)]
    chunks[0].append(baseline)
    chunks[mid].insert(0 if mid != 0 else 1, nf1)
    nf2_chunk = last
    # reserve the last slot of the last chunk for nf2
    reserved = [0] * n_chunks
    reserved[nf2_chunk] += 1
    it = iter(body)
    for ci in range(n_chunks):
        cap = sizes[ci] - len(chunks[ci]) - reserved[ci]
        for _ in range(cap):
            nxt = next(it, None)
            if nxt is None:
                break
            chunks[ci].append(nxt)
    assert next(it, None) is None, "layout overflow"
    chunks[nf2_chunk].append(nf2)

    by_cid: Dict[str, Cond] = {}
    for ci, ch in enumerate(chunks):
        assert len(ch) == sizes[ci], (ci, len(ch), sizes[ci])
        for si, c in enumerate(ch):
            c.chunk, c.slot = ci, si
            assert c.cid not in by_cid
            by_cid[c.cid] = c
    assert chunks[0][0].cid == "baseline"
    return chunks, by_cid


def conditions_json(by_cid: Dict[str, Cond]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for cid, c in by_cid.items():
        d: Dict[str, Any] = {
            "class": c.klass,
            "alpha": c.alpha,
            "norm": c.norm,
            "batch_slot": {"chunk": c.chunk, "slot": c.slot},
        }
        if c.concept is not None:
            d["concept"] = c.concept
        if c.seed is not None:
            d["seed"] = c.seed
            d["matched_concept"] = c.matched_concept
        out[cid] = d
    return out


# ----------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------
def _chat_prompt(tokenizer: Any, messages: List[dict]) -> Tuple[str, torch.Tensor]:
    return format_chat_text(tokenizer, messages), format_chat(tokenizer, messages).cpu()


def _load_boolq(cfg: dict) -> Tuple[List[dict], str]:
    from datasets import load_dataset

    b = cfg["boolq"]
    rev = resolve_dataset_revision(b["repo"])
    ds = load_dataset(b["repo"], split=b["split"], revision=rev)
    ds = ds.add_column("orig_idx", list(range(len(ds)))).shuffle(seed=int(b["shuffle_seed"]))
    ds = ds.select(range(int(b["n"])))
    items = [{"orig_idx": int(r["orig_idx"]), "question": r["question"]} for r in ds]
    return items, rev


def _load_jbb(cfg: dict) -> Tuple[List[dict], List[dict], str, List[int], List[int]]:
    """Dev = first n_dev entries of default_rng(0).permutation(100). Only dev rows are
    selected from the dataset; held-out indices are returned but never read."""
    from datasets import load_dataset

    j = cfg["jbb"]
    n_total, n_dev = int(j["n_total"]), int(j["n_dev"])
    perm = np.random.default_rng(int(j["perm_seed"])).permutation(n_total)
    dev = [int(i) for i in perm[:n_dev]]
    held = [int(i) for i in perm[n_dev:]]
    assert len(dev) == 60 and len(held) == 40 and not (set(dev) & set(held))
    rev = resolve_dataset_revision(j["repo"])
    out = []
    for split in (j["harmful_split"], j["benign_split"]):
        ds = load_dataset(j["repo"], j["config"], split=split, revision=rev)
        assert len(ds) == n_total, f"{split}: {len(ds)} rows, expected {n_total}"
        sub = ds.select(dev)  # held-out rows are never selected / read
        out.append([{"orig_idx": dev[k], "goal": sub[k][j["field"]]} for k in range(len(dev))])
    return out[0], out[1], rev, dev, held


def build_prompts(tokenizer: Any, cfg: dict) -> Tuple[List[Prompt], Dict[str, Any]]:
    prompts: List[Prompt] = []
    info: Dict[str, Any] = {"dataset_revisions": {}}
    ge1 = int(cfg.get("start_pos_ge1", 1))
    ns = [int(n) for n in cfg["s1_trial_ns"]]

    # S1 INTRO (Macar start), S2 INTRO-ALL (pos >= 1)
    for i, n in enumerate(ns):
        text, ids, sp = build_introspection_prompt(tokenizer, n)
        prompts.append(Prompt("S1", i, ids.cpu(), int(sp), "full", text, {"trial_n": n}))
    for i, n in enumerate(ns):
        text, ids, _sp = build_introspection_prompt(tokenizer, n)
        prompts.append(Prompt("S2", i, ids.cpu(), ge1, "full", text, {"trial_n": n}))
    # S2v paraphrases (exact texts from config): three_turn or single_user
    s2v = list(cfg["s2v"])
    assert len(s2v) == 4, f"S2v needs 4 paraphrases, got {len(s2v)}"
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
        prompts.append(Prompt("S2v", i, ids, ge1, "full", text,
                              {"mode": mode, "paraphrase": meta_txt}))
    # S3 YESBIAS-FRAMED (Macar start)
    tn3 = int(cfg.get("s3_trial_n", 1))
    for i, q in enumerate(YESBIAS_QUESTIONS):
        text, ids, sp = build_yesbias_prompt(tokenizer, tn3, q)
        prompts.append(Prompt("S3", i, ids.cpu(), int(sp), "full", text,
                              {"question": q, "trial_n": tn3}))
    # S4 BOOLQ
    items, rev = _load_boolq(cfg)
    info["dataset_revisions"]["boolq"] = rev
    tmpl = cfg["boolq"]["user_template"]
    for i, it in enumerate(items):
        user = tmpl.format(q=it["question"])
        text, ids = _chat_prompt(tokenizer, [{"role": "user", "content": user}])
        prompts.append(Prompt("S4", i, ids, ge1, "full", text, {"orig_idx": it["orig_idx"]}))
    # S5 HARM / S6 BENIGN (JBB dev only)
    harm, benign, jrev, dev, held = _load_jbb(cfg)
    info["dataset_revisions"]["jbb"] = jrev
    info["jbb_dev_indices"] = dev
    info["jbb_heldout_indices"] = held
    for setname, rows in (("S5", harm), ("S6", benign)):
        for i, r in enumerate(rows):
            text, ids = _chat_prompt(tokenizer, [{"role": "user", "content": r["goal"]}])
            prompts.append(Prompt(setname, i, ids, ge1, "early", text,
                                  {"orig_idx": r["orig_idx"]}))
    # S7 MMLU
    items7, mrev = load_mmlu_validation(n=int(cfg["mmlu"]["n"]), seed=int(cfg["mmlu"]["seed"]))
    info["dataset_revisions"]["mmlu"] = mrev
    for i, it in enumerate(items7):
        text, ids = _chat_prompt(tokenizer, [{"role": "user", "content": format_mmlu_user(it)}])
        prompts.append(Prompt("S7", i, ids, ge1, "early", text,
                              {"mmlu_id": it["id"], "subject": it.get("subject")}))

    assert len(prompts) == EXPECTED_PROMPTS, f"{len(prompts)} prompts != {EXPECTED_PROMPTS}"
    tail_n = int(cfg["tail_tokens"])
    short = [(p.pset, p.idx, p.n_tokens) for p in prompts if p.n_tokens < tail_n]
    # Secondary readout uses last min(tail_tokens, T) tokens (forward_chunk already does this).
    # Some JBB Goals tokenize to <20 tokens under the chat template; do not drop them.
    info["short_prompts_lt_tail"] = short
    for p in prompts:
        assert 0 <= p.start_pos < p.n_tokens - 1, (p.pset, p.idx, p.start_pos, p.n_tokens)
        assert p.mode == ("full" if p.pset in FULL_SETS else "early")
    return prompts, info


def interleave_prompts(prompts: Sequence[Prompt]) -> List[Prompt]:
    """Round-robin over sets so the first 5% of prompts are representative for timing."""
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
# Forward with captures
# ----------------------------------------------------------------------------------------
def _mk_ln_hook(cap: Capture, l: int, keep: int, keep_full: bool, halt: bool, halt_layer: int):
    def hook(_mod: Any, args: Tuple[Any, ...], output: torch.Tensor) -> None:
        m = args[0]
        cap.x[l] = output[:, -keep:, :].detach().clone()
        cap.m_last[l] = m[:, -1, :].detach().clone()
        if keep_full:
            cap.m_full[l] = m.detach().clone()
        if halt and l == halt_layer:
            raise _EarlyHalt()

    return hook


def _mk_resid_hook(cap: Capture, l: int, keep_full: bool):
    def hook(_mod: Any, _args: Any, output: Any) -> None:
        h = output[0] if isinstance(output, tuple) else output
        cap.h_last[l] = h[:, -1, :].detach().clone()
        if keep_full:
            cap.h_full[l] = h.detach().clone()

    return hook


@torch.no_grad()
def forward_chunk(
    ctx: Ctx,
    ids1: torch.Tensor,
    vecs: torch.Tensor,
    start_pos: int,
    *,
    halt: bool,
    want_logits: bool,
    tail: Optional[int] = None,
    keep_full: bool = False,
) -> Capture:
    """One forward of B identical prompts (no padding) with per-row steering vectors.

    A fresh add_to_residual_batched context (and fresh capture hooks) is opened here.
    """
    dev = ctx.device
    B = int(vecs.shape[0])
    ids = ids1.to(dev)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.expand(B, -1).contiguous()
    T = int(ids.shape[1])
    keep = T if (keep_full or tail is None) else min(int(tail), T)
    starts = torch.full((B,), int(start_pos), dtype=torch.long, device=dev)
    v = vecs.to(device=dev, dtype=torch.bfloat16)

    cap = Capture()
    handles = []
    try:
        for l in ctx.tap_layers:
            handles.append(
                ctx.layers[l].pre_feedforward_layernorm.register_forward_hook(
                    _mk_ln_hook(cap, l, keep, keep_full, halt, ctx.halt_layer)
                )
            )
        for rl in sorted(set(ctx.resid_for.values())):
            handles.append(ctx.layers[rl].register_forward_hook(_mk_resid_hook(cap, rl, keep_full)))
        kw: Dict[str, Any] = dict(
            input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
        )
        if ctx.supports_ltk:
            kw["logits_to_keep"] = 1
        with add_to_residual_batched(ctx.model, ctx.steer_layer, v, starts):
            try:
                try:
                    out = ctx.model(**kw)
                except TypeError as e:
                    if "logits_to_keep" in str(e) and ctx.supports_ltk:
                        ctx.supports_ltk = False
                        kw.pop("logits_to_keep")
                        out = ctx.model(**kw)
                    else:
                        raise
                if want_logits:
                    cap.logits = out.logits[:, -1, :].detach()
            except _EarlyHalt:
                cap.halted = True
    finally:
        for h in handles:
            h.remove()
    return cap


# ----------------------------------------------------------------------------------------
# Per-chunk scalar computation
# ----------------------------------------------------------------------------------------
def _make_baseline(ctx: Ctx, cap: Capture, j: int) -> Baseline:
    pre_last, pre_ln_m0, m0, h0 = {}, {}, {}, {}
    for l in ctx.tap_layers:
        tw = ctx.tw[l]
        pre, _ = encode_gates_fp32(cap.x[l][j : j + 1], tw)
        pre_last[l] = pre[0, -1].clone()
        m0[l] = cap.m_last[l][j].clone()
        ln = ctx.layers[l].pre_feedforward_layernorm
        p0, _ = encode_gates_fp32(ln(m0[l].unsqueeze(0)), tw)
        pre_ln_m0[l] = p0[0].clone()
    for rl in set(ctx.resid_for.values()):
        h0[rl] = cap.h_last[rl][j].float().clone()
    return Baseline(pre_last, pre_ln_m0, m0, h0)


@torch.no_grad()
def run_chunk(
    ctx: Ctx, prompt: Prompt, conds: Sequence[Cond], base: Optional[Baseline]
) -> Tuple[List[Dict[str, Any]], Optional[Baseline]]:
    """Forward one chunk (identical ids, different vectors) and compute every scalar.

    Returns (rows, baseline). If ``base`` is None and the chunk contains the baseline row, the
    baseline for this prompt is created from it (and used for all rows of the chunk).
    """
    B = len(conds)
    vecs = torch.stack([c.vec for c in conds]).to(ctx.device)  # [B,d] bf16
    halt = prompt.mode == "early"
    cap = forward_chunk(
        ctx, prompt.ids, vecs, prompt.start_pos,
        halt=halt, want_logits=not halt, tail=ctx.tail,
    )
    if halt:
        assert cap.halted, "early-halt did not trigger"
    else:
        assert cap.logits is not None

    if base is None:
        for j, c in enumerate(conds):
            if c.klass == "baseline":
                base = _make_baseline(ctx, cap, j)
                break
    has_base = base is not None
    vf = vecs.float()
    vnorm = vf.norm(dim=-1)  # [B]
    nan = float("nan")

    rows: List[Dict[str, Any]] = [
        {
            "gate": {"set": prompt.pset, "prompt_idx": prompt.idx, "cid": c.cid, "B": B,
                     "alpha": c.alpha, "has_baseline": has_base},
            "resid": {"set": prompt.pset, "prompt_idx": prompt.idx, "cid": c.cid, "B": B},
            "ld": None,
            "sparse": {},
        }
        for c in conds
    ]

    # ---- gates + sparse per transcoder layer
    for l in ctx.tap_layers:
        tw = ctx.tw[l]
        G = len(ctx.gates[l])
        x = cap.x[l]  # [B,k,d] bf16
        pre, act = encode_gates_fp32(x, tw)  # [B,k,G]
        pre_last, act_last = pre[:, -1], act[:, -1]
        mean_pre, mean_act = pre.mean(dim=1), act.mean(dim=1)
        thr = tw.threshold_gates
        if has_base:
            dpre = pre_last - base.pre_last[l].unsqueeze(0)
            ln = ctx.layers[l].pre_feedforward_layernorm
            x_dir = ln(base.m0[l].unsqueeze(0) + vecs)          # LN(m0 + alpha v), bf16
            pre_dir, _ = encode_gates_fp32(x_dir, tw)
            direct = pre_dir - base.pre_ln_m0[l].unsqueeze(0)  # pre_g(LN(m0+av)) - pre_g(LN(m0))
        else:
            dpre = torch.full_like(pre_last, nan)
            direct = torch.full_like(pre_last, nan)
        w = tw.w_enc_gates  # [d,G] fp32
        cosv = (vf @ w) / (vnorm.unsqueeze(1) * w.norm(dim=0).unsqueeze(0) + 1e-30)
        cosv = torch.where(vnorm.unsqueeze(1) > 0, cosv, torch.full_like(cosv, nan))

        acts_full = encode_full(x[:, -1, :], tw)  # [B, width]
        pre_c, act_c, mp_c, ma_c = pre_last.cpu(), act_last.cpu(), mean_pre.cpu(), mean_act.cpu()
        dpre_c, dir_c, cos_c = dpre.cpu(), direct.cpu(), cosv.cpu()
        thr_c = thr.cpu()
        for b in range(B):
            rows[b]["gate"][f"L{l}"] = {
                "gate_ids": list(ctx.gates[l]),
                "pre": _py(pre_c[b]),
                "act": _py(act_c[b]),
                "thr": _py(thr_c),
                f"mean{ctx.tail}_pre": _py(mp_c[b]),
                f"mean{ctx.tail}_act": _py(ma_c[b]),
                "dpre": _py(dpre_c[b]),
                "direct_pre": _py(dir_c[b]),
                "cos_wv": _py(cos_c[b]),
            }
            idx, val = sparse_from_acts(acts_full[b])
            rows[b]["sparse"][l] = (idx.cpu(), val.cpu())
        del acts_full

    # ---- residual scalars
    for l in ctx.tap_layers:
        rl = ctx.resid_for[l]
        hs = cap.h_last[rl].float()  # [B,d]
        if has_base:
            h0 = base.h0[rl]
            n0 = h0.norm()
            diff = hs - h0.unsqueeze(0)
            reach = diff.norm(dim=-1) / n0
            reach_ind = (diff - vf).norm(dim=-1) / n0
            dproj = (diff @ ctx.d_hat[rl]) / n0
        else:
            reach = reach_ind = dproj = torch.full((B,), nan, device=hs.device)
        r_c, ri_c, dp_c = reach.cpu(), reach_ind.cpu(), dproj.cpu()
        for b in range(B):
            rows[b]["resid"][f"L{l}"] = {
                "resid_layer": rl,
                "reach": _py(float(r_c[b])),
                "reach_indirect": _py(float(ri_c[b])),
                "dirproj": _py(float(dp_c[b])),
            }

    # ---- LD (full forward only): LD = logit(No) - logit(Yes)
    if not halt:
        lg = cap.logits.float()
        ld = (lg[:, ctx.no_id] - lg[:, ctx.yes_id]).cpu()
        top1 = lg.argmax(dim=-1).cpu()
        ly, ln_ = lg[:, ctx.yes_id].cpu(), lg[:, ctx.no_id].cpu()
        for b in range(B):
            rows[b]["ld"] = {
                "set": prompt.pset, "prompt_idx": prompt.idx, "cid": conds[b].cid, "B": B,
                "ld": _py(float(ld[b])), "logit_yes": _py(float(ly[b])),
                "logit_no": _py(float(ln_[b])), "top1_id": int(top1[b]),
            }
    return rows, base


# ----------------------------------------------------------------------------------------
# Sanity-check helpers
# ----------------------------------------------------------------------------------------
def _tl_folder(root: Path, cfg: dict, layer: int) -> Path:
    return Path(root) / cfg["transcoder_subdir"].format(layer=layer)


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def _zeros(ctx: Ctx, n: int, d: int) -> torch.Tensor:
    return torch.zeros(n, d, dtype=torch.bfloat16, device=ctx.device)


def sanity1_provenance(slog: Logger, git_hash_str: str, revision_sha: str, model_dir: str) -> None:
    slog("=== SANITY 1: provenance (supplied by caller) ===")
    slog(f"git_hash(caller)={git_hash_str} model_revision={revision_sha} model_dir={model_dir}")
    if not git_hash_str or str(git_hash_str).strip().lower() in ("unknown", "none", ""):
        raise SanityAbort("SANITY1: git hash not supplied by caller")
    if not revision_sha or len(str(revision_sha)) < 7:
        raise SanityAbort("SANITY1: model revision sha not supplied by caller")
    slog("SANITY1 PASS")


def sanity2_configs(slog: Logger, cfg: dict, transcoder_root: Path) -> Dict[str, Any]:
    from safetensors import safe_open

    slog("=== SANITY 2: transcoder config.json (both layers) ===")
    out: Dict[str, Any] = {}
    for l in cfg["transcoder_layers"]:
        folder = _tl_folder(transcoder_root, cfg, int(l))
        raw = json.loads((folder / "config.json").read_text())
        slog(f"--- {folder.name}/config.json ---")
        slog(json.dumps(raw, indent=2, sort_keys=True))
        for k in ("width", "affine_connection", "hf_hook_point_in"):
            if k not in raw:
                raise SanityAbort(f"SANITY2: L{l} config.json missing key {k!r}")
        tc = load_transcoder_config(folder)  # asserts width / affine / folder / hook point
        assert tc.layer == int(l)
        with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
            keys = set(f.keys())
        need = {"w_enc", "b_enc", "threshold", "w_dec", "b_dec"}
        if not need <= keys:
            raise SanityAbort(f"SANITY2: L{l} params.safetensors missing {need - keys}")
        slog(f"L{l}: safetensors keys={sorted(keys)}")
        out[f"L{l}"] = {"config": raw, "safetensors_keys": sorted(keys)}
    slog("SANITY2 PASS")
    return out


def sanity3_ln_equals_mlp_input(ctx: Ctx, p: Prompt, slog: Logger) -> None:
    slog("=== SANITY 3: LN output == MLP input (bit-identical, 1 prompt) ===")
    mlp_in: Dict[int, torch.Tensor] = {}
    handles = []
    for l in ctx.tap_layers:
        def mk(l=l):
            def pre(_m: Any, args: Tuple[Any, ...]) -> None:
                mlp_in[l] = args[0].detach().clone()
            return pre
        handles.append(ctx.layers[l].mlp.register_forward_pre_hook(mk()))
    try:
        d = ctx.tw[ctx.tap_layers[0]].w_enc.shape[0]
        cap = forward_chunk(ctx, p.ids, _zeros(ctx, 1, d), p.start_pos,
                            halt=False, want_logits=False, keep_full=True)
    finally:
        for h in handles:
            h.remove()
    for l in ctx.tap_layers:
        ok = torch.equal(cap.x[l], mlp_in[l])
        slog(f"L{l}: LN output shape={tuple(cap.x[l].shape)} equal_to_mlp_input={ok}")
        if not ok:
            raise SanityAbort(f"SANITY3: L{l} LN output != mlp input (max abs "
                              f"{_max_abs(cap.x[l], mlp_in[l])})")
    slog("SANITY3 PASS")


def sanity4_early_halt(ctx: Ctx, prompts: Sequence[Prompt], vecs: Sequence[torch.Tensor],
                       slog: Logger) -> None:
    slog("=== SANITY 4: early-halt captures == full forward (2 prompts, unsteered + steered rows) ===")
    d = vecs[0].shape[-1]
    vv = torch.stack([torch.zeros(d, dtype=torch.bfloat16)] + list(vecs[:2])).to(ctx.device)
    for p in prompts:
        a = forward_chunk(ctx, p.ids, vv, p.start_pos, halt=False, want_logits=False, tail=ctx.tail)
        b = forward_chunk(ctx, p.ids, vv, p.start_pos, halt=True, want_logits=False, tail=ctx.tail)
        assert b.halted and not a.halted
        for l in ctx.tap_layers:
            for name, ta, tb in (("x", a.x[l], b.x[l]), ("m", a.m_last[l], b.m_last[l])):
                if not torch.equal(ta, tb):
                    raise SanityAbort(f"SANITY4: {p.pset}[{p.idx}] L{l} {name} differs "
                                      f"(max abs {_max_abs(ta, tb)})")
            sa = encode_full(a.x[l][:, -1], ctx.tw[l])
            sb = encode_full(b.x[l][:, -1], ctx.tw[l])
            if not torch.equal(sa, sb):
                raise SanityAbort(f"SANITY4: {p.pset}[{p.idx}] L{l} sparse acts differ")
        for rl in a.h_last:
            if not torch.equal(a.h_last[rl], b.h_last[rl]):
                raise SanityAbort(f"SANITY4: {p.pset}[{p.idx}] resid L{rl} differs")
        slog(f"{p.pset}[{p.idx}] (T={p.n_tokens}): LN out / m / sparse acts / resid44,49 identical")
    slog("SANITY4 PASS")


def sanity5_batch_and_steer(ctx: Ctx, p: Prompt, cvec: torch.Tensor, slog: Logger) -> Dict[str, Any]:
    slog("=== SANITY 5: alpha=0 identical within batch; cross-batch noise; alpha=4 effect; BOS ===")
    d = cvec.shape[-1]
    z = torch.zeros(d, dtype=torch.bfloat16)
    rows = [z, z, cvec.to(torch.bfloat16), z, z]
    vv = torch.stack(rows).to(ctx.device)
    cap = forward_chunk(ctx, p.ids, vv, p.start_pos, halt=False, want_logits=True, keep_full=True)
    zi, si = [0, 1, 3, 4], 2
    for l in ctx.tap_layers:
        for name, t in (("x_full", cap.x[l]), ("m_full", cap.m_full[l])):
            ref = t[0]
            for j in zi[1:]:
                if not torch.equal(t[j], ref):
                    raise SanityAbort(f"SANITY5: alpha=0 rows differ within batch ({name} L{l} row {j})")
    for rl in cap.h_full:
        for j in zi[1:]:
            if not torch.equal(cap.h_full[rl][j], cap.h_full[rl][0]):
                raise SanityAbort(f"SANITY5: alpha=0 resid rows differ within batch (L{rl})")
    if not torch.equal(cap.logits[zi[1]], cap.logits[0]):
        raise SanityAbort("SANITY5: alpha=0 logits differ within batch")
    slog("alpha=0 rows (4 of 5) bit-identical within the batch: x, m, resid, logits")

    res: Dict[str, Any] = {}
    for l in ctx.tap_layers:
        xs, x0 = cap.x[l][si], cap.x[l][0]
        if torch.equal(xs[0], x0[0]):
            slog(f"L{l}: BOS LN output identical between steered and alpha=0 rows")
        else:
            raise SanityAbort(f"SANITY5: steering changed BOS position at L{l} "
                              f"(max abs {_max_abs(xs[0], x0[0])})")
        pre_pos = max(0, p.start_pos)
        if pre_pos > 0 and not torch.equal(xs[:pre_pos], x0[:pre_pos]):
            raise SanityAbort(f"SANITY5: L{l} positions < start_pos changed under steering")
        i_s, v_s = sparse_from_acts(encode_full(xs[-1:], ctx.tw[l])[0])
        i_0, v_0 = sparse_from_acts(encode_full(x0[-1:], ctx.tw[l])[0])
        same = bool(torch.equal(i_s, i_0) and torch.equal(v_s, v_0))
        n_diff = int(len(set(i_s.tolist()) ^ set(i_0.tolist())))
        slog(f"L{l}: readout nnz steered={len(i_s)} alpha0={len(i_0)} sym_diff_support={n_diff} "
             f"identical={same} max|dx|={_max_abs(xs[-1], x0[-1]):.4f}")
        if l == 45 and same:
            raise SanityAbort("SANITY5: alpha=4 steering did not change L45 readout acts")
        res[f"L{l}_nnz_steered"] = int(len(i_s))
        res[f"L{l}_nnz_alpha0"] = int(len(i_0))
        res[f"L{l}_sym_diff"] = n_diff

    # cross-batch noise: alpha=0 row under different batch sizes
    ref_cap = forward_chunk(ctx, p.ids, _zeros(ctx, 24, d), p.start_pos,
                            halt=False, want_logits=True, tail=ctx.tail)
    noise: Dict[str, Any] = {}
    for B in (1, 5, 23):
        c2 = forward_chunk(ctx, p.ids, _zeros(ctx, B, d), p.start_pos,
                           halt=False, want_logits=True, tail=ctx.tail)
        e: Dict[str, Any] = {}
        for l in ctx.tap_layers:
            e[f"x_L{l}_maxabs"] = _max_abs(c2.x[l][0], ref_cap.x[l][0])
            pa, _ = encode_gates_fp32(c2.x[l][0, -1], ctx.tw[l])
            pb, _ = encode_gates_fp32(ref_cap.x[l][0, -1], ctx.tw[l])
            e[f"pre_g_L{l}_maxabs"] = _max_abs(pa, pb)
        for rl in ref_cap.h_last:
            e[f"h_L{rl}_maxabs"] = _max_abs(c2.h_last[rl][0], ref_cap.h_last[rl][0])
        lg2, lg1 = c2.logits[0].float(), ref_cap.logits[0].float()
        e["LD_absdiff"] = float(((lg2[ctx.no_id] - lg2[ctx.yes_id])
                                 - (lg1[ctx.no_id] - lg1[ctx.yes_id])).abs())
        noise[f"B{B}_vs_B24"] = e
        slog(f"cross-batch noise B={B} vs B=24 (alpha=0): {json.dumps(_py(e))}")
    res["cross_batch_noise"] = noise
    slog("SANITY5 PASS")
    return res


def _attach_decode(tw: TranscoderWeights, folder: Path, device: torch.device) -> None:
    from safetensors import safe_open

    with safe_open(str(folder / "params.safetensors"), framework="pt", device="cpu") as f:
        tw.w_dec = f.get_tensor("w_dec").to(device=device, dtype=torch.bfloat16)
        tw.b_dec = f.get_tensor("b_dec").to(device=device, dtype=torch.float32)
        if "affine_skip_connection" in set(f.keys()):
            tw.affine_skip = f.get_tensor("affine_skip_connection").to(
                device=device, dtype=torch.float32)


def _free_decode(tw: TranscoderWeights) -> None:
    tw.w_dec = None
    tw.b_dec = None
    tw.affine_skip = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_out_module(layers: Any, layer: int, hook_point: Optional[str]) -> Any:
    """'model.layers.{L}.mlp.output' -> layers[L].mlp (output hook). Defaults to mlp."""
    if not hook_point:
        return layers[layer].mlp
    parts = hook_point.split(".")
    assert parts[-1] == "output", f"unsupported hf_hook_point_out {hook_point!r}"
    mod = layers[layer]
    for name in parts[3:-1]:  # skip 'model','layers',L
        mod = getattr(mod, name)
    return mod


def sanity6_variance_explained(
    ctx: Ctx, prompts: Sequence[Prompt], transcoder_root: Path, cfg: dict, slog: Logger
) -> Dict[str, Any]:
    slog("=== SANITY 6: variance explained with decode (3 S1 + 3 S5 unsteered, pos != BOS) ===")
    sc = cfg["sanity"]
    d = ctx.tw[ctx.tap_layers[0]].w_enc.shape[0]
    result: Dict[str, Any] = {}
    stop_reasons: List[str] = []
    for l in ctx.tap_layers:
        tw = ctx.tw[l]
        folder = _tl_folder(transcoder_root, cfg, l)
        tcfg = load_transcoder_config(folder)
        mod = _resolve_out_module(ctx.layers, l, tcfg.raw.get("hf_hook_point_out"))
        slog(f"L{l}: hf_hook_point_out={tcfg.raw.get('hf_hook_point_out')!r} -> hooking {type(mod).__name__}")
        _attach_decode(tw, folder, ctx.device)
        try:
            width = tw.w_enc.shape[1]
            combos: Dict[str, Dict[str, Any]] = {}
            l0s: List[float] = []
            sum_y = torch.zeros(d, dtype=torch.float64, device=ctx.device)
            sum_y2, n_tok = 0.0, 0
            for p in prompts:
                holder: Dict[str, torch.Tensor] = {}

                def hk(_m: Any, _a: Any, out: Any, holder=holder) -> None:
                    holder["y"] = (out[0] if isinstance(out, tuple) else out).detach().clone()

                h = mod.register_forward_hook(hk)
                try:
                    cap = forward_chunk(ctx, p.ids, _zeros(ctx, 1, d), p.start_pos,
                                        halt=False, want_logits=False, keep_full=True)
                finally:
                    h.remove()
                x = cap.x[l][0, 1:]            # pos != BOS
                y = holder["y"][0, 1:].float()
                acts = encode_full(x, tw)      # [T-1, width] bf16
                l0s.extend((acts != 0).sum(dim=-1).float().cpu().tolist())
                yd = y.double()
                sum_y += yd.sum(0)
                sum_y2 += float((yd ** 2).sum())
                n_tok += int(y.shape[0])
                for wname, Wd in (("W", tw.w_dec), ("W.T", tw.w_dec.T)):
                    if tuple(Wd.shape) != (width, y.shape[-1]):
                        continue
                    base_pred = (acts @ Wd).float() + tw.b_dec.float()
                    skips: List[Tuple[str, Optional[torch.Tensor]]] = [("noskip", None)]
                    if tw.affine_skip is not None:
                        for sname, S in (("A", tw.affine_skip), ("A.T", tw.affine_skip.T)):
                            if tuple(S.shape) == (x.shape[-1], y.shape[-1]):
                                skips.append((sname, S))
                    for sname, S in skips:
                        pred = base_pred if S is None else base_pred + x.float() @ S
                        key = f"{wname}+{sname}"
                        e = combos.setdefault(key, {"sse": 0.0})
                        e["sse"] += float(((y - pred).double() ** 2).sum())
            sst = sum_y2 - float((sum_y ** 2).sum()) / max(n_tok, 1)
            ves = {k: 1.0 - v["sse"] / sst for k, v in combos.items()}
            mean_l0 = float(np.mean(l0s)) if l0s else float("nan")
            best_k = max(ves, key=lambda k: ves[k]) if ves else None
            best = ves[best_k] if best_k else float("-inf")
            slog(f"L{l}: n_tokens={n_tok} mean_L0={mean_l0:.1f} (min {min(l0s):.0f}, max {max(l0s):.0f})")
            for k, v in sorted(ves.items()):
                slog(f"L{l}:   VE[{k}] = {v:.4f}")
            slog(f"L{l}: BEST VE = {best:.4f} via {best_k}")
            result[f"L{l}"] = {"mean_L0": mean_l0, "ve": ves, "best_ve": best, "best_combo": best_k,
                               "n_tokens": n_tok}
            if not (best >= float(sc["ve_min"])):
                stop_reasons.append(f"L{l} best VE {best:.4f} < {sc['ve_min']}")
            if not (float(sc["l0_min"]) <= mean_l0 <= float(sc["l0_max"])):
                stop_reasons.append(f"L{l} mean L0 {mean_l0:.1f} outside [{sc['l0_min']},{sc['l0_max']}]")
        finally:
            _free_decode(tw)
    if stop_reasons:
        slog("SANITY6 STOP: " + "; ".join(stop_reasons))
        raise RunStop("SANITY6: " + "; ".join(stop_reasons))
    slog("SANITY6 PASS")
    return result


def sanity7_gate_precision(ctx: Ctx, prompts: Sequence[Prompt], cfg: dict, slog: Logger) -> Dict[str, Any]:
    slog("=== SANITY 7: gate fp32 vs bf16 (within 2% or both 0), 3 S1 prompts unsteered ===")
    tol = float(cfg["sanity"]["gate_rel_tol"])
    d = ctx.tw[ctx.tap_layers[0]].w_enc.shape[0]
    bad: List[str] = []
    res: Dict[str, Any] = {}
    for p in prompts:
        cap = forward_chunk(ctx, p.ids, _zeros(ctx, 1, d), p.start_pos,
                            halt=True, want_logits=False, tail=ctx.tail)
        for l in ctx.tap_layers:
            tw = ctx.tw[l]
            ids = torch.tensor(ctx.gates[l], device=ctx.device)
            x = cap.x[l][0, -1]
            pre32, act32 = encode_gates_fp32(x, tw)
            act16 = encode_full(x.unsqueeze(0), tw)[0][ids].float()
            pre16 = (x.to(torch.bfloat16) @ tw.w_enc[:, ids] + tw.b_enc[ids]).float()
            for gi, g in enumerate(ctx.gates[l]):
                a32, a16 = float(act32[gi]), float(act16[gi])
                p32, p16 = float(pre32[gi]), float(pre16[gi])
                both0 = (a32 == 0.0 and a16 == 0.0)
                rel = abs(a32 - a16) / max(abs(a32), 1e-30)
                ok = both0 or rel <= tol
                slog(f"{p.pset}[{p.idx}] L{l} g{g}: act fp32={a32:.5f} bf16={a16:.5f} rel={rel:.4f} "
                     f"| pre fp32={p32:.5f} bf16={p16:.5f} -> {'ok' if ok else 'FAIL'}")
                res[f"{p.pset}{p.idx}_L{l}_g{g}"] = {"act32": a32, "act16": a16, "pre32": p32,
                                                     "pre16": p16, "ok": ok}
                if not ok:
                    bad.append(f"{p.pset}[{p.idx}] L{l} g{g}")
    if bad:
        raise SanityAbort("SANITY7: fp32 vs bf16 gate mismatch > 2%: " + ", ".join(bad))
    slog("SANITY7 PASS")
    return res


def sanity8_token_views(ctx: Ctx, prompts: Sequence[Prompt], slog: Logger) -> None:
    slog("=== SANITY 8: final 5 tokens, readout token, top-5 next (2 prompts/set, unsteered) ===")
    d = ctx.tw[ctx.tap_layers[0]].w_enc.shape[0]
    tok = ctx.tokenizer
    for s in SET_ORDER:
        for p in [q for q in prompts if q.pset == s][:2]:
            cap = forward_chunk(ctx, p.ids, _zeros(ctx, 1, d), p.start_pos,
                                halt=False, want_logits=True, tail=ctx.tail)
            ids = p.ids[0].tolist()
            last5 = [(i, tok.decode([ids[i]])) for i in range(len(ids) - 5, len(ids))]
            lg = cap.logits[0].float()
            top = torch.topk(lg, 5)
            tops = [(tok.decode([int(i)]), round(float(v), 3)) for v, i in zip(top.values, top.indices)]
            slog(f"{p.pset}[{p.idx}] T={len(ids)} mode={p.mode} readout_pos={len(ids)-1} "
                 f"readout_tok={tok.decode([ids[-1]])!r}")
            slog(f"    last5={last5}")
            slog(f"    top5_next={tops}")


def sanity9_steer_marks(tokenizer: Any, prompts: Sequence[Prompt], slog: Logger) -> None:
    slog("=== SANITY 9: steering start marked in decoded prompt (1 per set) ===")
    for s in SET_ORDER:
        p = next(q for q in prompts if q.pset == s)
        ids = p.ids[0].tolist()
        pieces = [tokenizer.decode([i]) for i in ids]
        marked = "".join(pieces[: p.start_pos]) + "⟦STEER→⟧" + "".join(pieces[p.start_pos:])
        slog(f"{p.pset}[{p.idx}] start_pos={p.start_pos} T={len(ids)} "
             f"tok[start-1]={pieces[p.start_pos-1]!r} tok[start]={pieces[p.start_pos]!r}")
        slog("    " + repr(marked if len(marked) < 1800 else marked[:900] + " ... " + marked[-900:]))


def sanity10_gate_values(ctx: Ctx, p: Prompt, conds: Sequence[Cond], slog: Logger) -> None:
    slog("=== SANITY 10: gate values, S1 unsteered + alpha=4 for 3 concepts ===")
    rows, _ = run_chunk(ctx, p, list(conds), None)
    for c, r in zip(conds, rows):
        for l in ctx.tap_layers:
            g = r["gate"][f"L{l}"]
            slog(f"{c.cid:<34s} L{l} gates={g['gate_ids']} pre={_fmt(g['pre'])} act={_fmt(g['act'])} "
                 f"thr={_fmt(g['thr'])} dpre={_fmt(g['dpre'])} direct={_fmt(g['direct_pre'])} "
                 f"cos(w,v)={_fmt(g['cos_wv'])}")
        slog(f"{'':34s} reach45/50={r['resid']['L45']['reach']}/{r['resid']['L50']['reach']} "
             f"LD={None if r['ld'] is None else round(r['ld']['ld'], 4)}")


def _fmt(xs: Any) -> str:
    return "[" + ", ".join("None" if v is None else f"{v:.4f}" for v in xs) + "]"


# ----------------------------------------------------------------------------------------
# Output sinks
# ----------------------------------------------------------------------------------------
class Sinks:
    def __init__(self, raw: Path):
        self.raw = raw
        self.f_gate = (raw / "gate_scalars.jsonl").open("w")
        self.f_res = (raw / "residual_scalars.jsonl").open("w")
        self.f_ld = (raw / "logits_ld.jsonl").open("w")
        self.f_fail = (raw / "failed_rows.jsonl").open("w")
        self.sparse: Dict[int, Dict[Tuple[str, int, str], Tuple[torch.Tensor, torch.Tensor]]] = {
            45: {}, 50: {}}
        self.n_gate = self.n_res = self.n_ld = self.n_fail = 0

    def write_rows(self, rows: Sequence[Dict[str, Any]]) -> None:
        for r in rows:
            g, rs = r["gate"], r["resid"]
            key = (g["set"], g["prompt_idx"], g["cid"])
            self.f_gate.write(json.dumps(g) + "\n")
            self.f_res.write(json.dumps(rs) + "\n")
            self.n_gate += 1
            self.n_res += 1
            if r["ld"] is not None:
                self.f_ld.write(json.dumps(r["ld"]) + "\n")
                self.n_ld += 1
            for l, sp in r["sparse"].items():
                self.sparse[int(l)][key] = sp

    def write_failure(self, rec: Dict[str, Any]) -> None:
        self.f_fail.write(json.dumps(rec) + "\n")
        self.n_fail += 1

    def flush(self) -> None:
        for f in (self.f_gate, self.f_res, self.f_ld, self.f_fail):
            f.flush()

    def save_sparse(self) -> None:
        for l in (45, 50):
            tmp = self.raw / f"captures_sparse_L{l}.pt.tmp"
            torch.save(self.sparse[l], tmp)
            os.replace(tmp, self.raw / f"captures_sparse_L{l}.pt")

    def close(self) -> None:
        self.flush()
        for f in (self.f_gate, self.f_res, self.f_ld, self.f_fail):
            f.close()


def process_rows_safely(
    ctx: Ctx, prompt: Prompt, conds: Sequence[Cond], base: Optional[Baseline],
    sinks: Sinks, errors: List[str], max_consec: int,
) -> Optional[Baseline]:
    """run_chunk with logging; on any exception split the chunk (baseline stays first) and retry.
    Rows are never silently dropped: unrecoverable single rows are written to failed_rows.jsonl."""
    try:
        rows, base = run_chunk(ctx, prompt, conds, base)
        sinks.write_rows(rows)
        ctx.consec_fail = 0
        return base
    except Exception as e:  # noqa: BLE001
        msg = f"{prompt.pset}[{prompt.idx}] chunk of {len(conds)} ({conds[0].cid}..): {type(e).__name__}: {e}"
        ctx.log("EXCEPTION " + msg)
        ctx.log(traceback.format_exc())
        errors.append(msg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if len(conds) > 1:
            mid = len(conds) // 2
            base = process_rows_safely(ctx, prompt, conds[:mid], base, sinks, errors, max_consec)
            return process_rows_safely(ctx, prompt, conds[mid:], base, sinks, errors, max_consec)
        sinks.write_failure({"set": prompt.pset, "prompt_idx": prompt.idx, "cid": conds[0].cid,
                             "error": f"{type(e).__name__}: {e}"})
        ctx.consec_fail += 1
        if ctx.consec_fail > max_consec:
            raise RuntimeError(f"{ctx.consec_fail} consecutive single-row failures; aborting") from e
        return base


# ----------------------------------------------------------------------------------------
# Main entry
# ----------------------------------------------------------------------------------------
def _load_d_hat(path: str, layers: Sequence[int], device: torch.device) -> Dict[int, torch.Tensor]:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    dh = {int(k): v for k, v in pack["d_hat"].items()}
    out: Dict[int, torch.Tensor] = {}
    for l in layers:
        assert l in dh, f"d_hat missing layer {l}; have {sorted(dh)[:10]}..."
        v = dh[l].float().reshape(-1)
        out[l] = (v / v.norm()).to(device)
    return out


def run_wo_003(
    model_dir,
    revision_sha,
    config_path,
    results_dir,
    ledger_path,
    transcoder_root,
    wo001_vectors_path,
    wo001_snapshot_path,
    wo002b_directions_path,
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
        "wo_id": cfg.get("wo_id", "WO-003"),
        "git_hash": git_hash_str,
        "model_revision": revision_sha,
        "status": "RUNNING",
        "errors": [],
        "sanity": {},
    }

    def write_summary() -> None:
        (results_path / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    sinks: Optional[Sinks] = None
    with gpu_ledger(
        ledger_path,
        wo_id="WO-003",
        function="wo_003_run",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            # ---------------------------------------------------------------- SANITY 1, 2
            sanity1_provenance(slog, git_hash_str, revision_sha, str(model_dir))
            summary["sanity"]["s1_provenance"] = True
            transcoder_root = Path(transcoder_root)
            summary["sanity"]["s2_configs"] = sanity2_configs(slog, cfg, transcoder_root)
            if cfg.get("s2v_texts_provisional", False):
                slog("WARNING: configs/wo_003.yaml s2v_texts_provisional=true — S2v paraphrases are "
                     "NOT yet the frozen WO-003 texts.")
            summary["s2v_texts_provisional"] = bool(cfg.get("s2v_texts_provisional", False))

            # ---------------------------------------------------------------- setup
            alphas = [float(a) for a in cfg["alphas"]]
            rand_seeds = [int(s) for s in cfg["random_seeds"]]
            bs = int(cfg["batch_size"])
            gates = {int(k): [int(g) for g in v] for k, v in cfg["gates"].items()}
            tap_layers = tuple(int(l) for l in cfg["transcoder_layers"])
            resid_for = {int(k): int(v) for k, v in cfg["resid_layers"].items()}
            summary["config"] = {k: cfg[k] for k in
                                 ("layer_steer", "alphas", "batch_size", "gates", "resid_layers",
                                  "halt_layer", "tail_tokens", "random_seeds",
                                  "projected_gpu_limit_hours")}

            wo001 = json.loads(Path(wo001_snapshot_path).read_text())
            concepts = list(wo001["concepts"])
            assert len(concepts) == int(cfg["n_concepts"]) == 50
            pack50 = torch.load(wo001_vectors_path, map_location="cpu", weights_only=False)
            vec50 = {c: pack50["vectors"][i].float() for i, c in enumerate(pack50["concepts"])}
            for c in concepts:
                assert c in vec50, f"missing WO-001 vector for {c}"
            assert len(rand_seeds) == 20

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            d_model = int(next(iter(vec50.values())).shape[-1])

            chunks, by_cid = build_conditions(concepts, vec50, alphas, rand_seeds, d_model, bs,
                                              int(cfg["n_noise_floor"]))
            assert len(by_cid) == EXPECTED_CONDITIONS, len(by_cid)
            (raw / "conditions.json").write_text(json.dumps(conditions_json(by_cid), indent=2))
            slog(f"conditions: {len(by_cid)} in {len(chunks)} chunks of sizes {[len(c) for c in chunks]}; "
                 f"baseline@{by_cid['baseline'].chunk}/{by_cid['baseline'].slot} "
                 f"noise1@{by_cid['noise_floor_1'].chunk}/{by_cid['noise_floor_1'].slot} "
                 f"noise2@{by_cid['noise_floor_2'].chunk}/{by_cid['noise_floor_2'].slot}")

            prompts, pinfo = build_prompts(tokenizer, cfg)
            summary["dataset_revisions"] = pinfo["dataset_revisions"]
            summary["jbb_dev_indices"] = pinfo["jbb_dev_indices"]
            summary["jbb_heldout_indices_NOT_RUN"] = pinfo["jbb_heldout_indices"]
            summary["short_prompts_lt_tail"] = pinfo.get("short_prompts_lt_tail", [])
            if summary["short_prompts_lt_tail"]:
                slog(
                    f"NOTE: {len(summary['short_prompts_lt_tail'])} prompts have T < "
                    f"tail_tokens={cfg['tail_tokens']}; secondary readout uses last min(20,T): "
                    f"{summary['short_prompts_lt_tail'][:20]}..."
                )
            with (raw / "prompts.jsonl").open("w") as fp:
                for p in prompts:
                    fp.write(json.dumps({
                        "set": p.pset, "prompt_idx": p.idx, "mode": p.mode,
                        "start_pos": p.start_pos, "n_tokens": p.n_tokens, "text": p.text,
                        "ids": p.ids[0].tolist(), "meta": p.meta,
                    }) + "\n")
            counts = {s: sum(1 for p in prompts if p.pset == s) for s in SET_ORDER}
            summary["prompt_counts"] = counts
            slog(f"prompts: {len(prompts)} {counts}")
            expected_rows = len(prompts) * len(by_cid)

            # ---------------------------------------------------------------- model + transcoders
            model = load_model(model_dir, revision=revision_sha, device=0)
            device = next(model.parameters()).device
            assert get_d_model(model) == d_model
            layers = get_layers(model)
            yes_id, no_id = yes_no_token_ids(tokenizer)
            summary["env"] = collect_env()
            slog(f"yes_id={yes_id} no_id={no_id}; LD = logit(No) - logit(Yes)")

            tw = {l: load_encode_weights(_tl_folder(transcoder_root, cfg, l), gates[l], device)
                  for l in tap_layers}
            for l in tap_layers:
                assert tw[l].gate_ids == gates[l]
            d_hat = _load_d_hat(wo002b_directions_path, sorted(set(resid_for.values())), device)

            ctx = Ctx(
                model=model, layers=layers, tokenizer=tokenizer, device=device, tw=tw, d_hat=d_hat,
                yes_id=yes_id, no_id=no_id, steer_layer=int(cfg["layer_steer"]), gates=gates,
                tap_layers=tap_layers, resid_for=resid_for, halt_layer=int(cfg["halt_layer"]),
                tail=int(cfg["tail_tokens"]), log=slog,
            )

            P = {s: [p for p in prompts if p.pset == s] for s in SET_ORDER}
            c_vec = lambda c, a=4: by_cid[_cname_concept(concepts[c], a)].vec  # noqa: E731

            # ---------------------------------------------------------------- SANITY 3-10
            sanity3_ln_equals_mlp_input(ctx, P["S1"][0], slog)
            summary["sanity"]["s3_ln_equals_mlp_input"] = True
            sanity4_early_halt(ctx, [P["S1"][0], P["S5"][0]], [c_vec(0), c_vec(1)], slog)
            summary["sanity"]["s4_early_halt_equals_full"] = True
            summary["sanity"]["s5_batch_and_steer"] = sanity5_batch_and_steer(
                ctx, P["S1"][0], c_vec(0), slog)
            summary["sanity"]["s6_variance_explained"] = sanity6_variance_explained(
                ctx, P["S1"][:3] + P["S5"][:3], transcoder_root, cfg, slog)
            summary["sanity"]["s7_gate_precision"] = sanity7_gate_precision(ctx, P["S1"][:3], cfg, slog)
            sanity8_token_views(ctx, prompts, slog)
            sanity9_steer_marks(tokenizer, prompts, slog)
            sanity10_gate_values(
                ctx, P["S1"][0],
                [by_cid["baseline"]] + [by_cid[_cname_concept(concepts[i], 4)] for i in range(3)],
                slog)
            write_summary()

            # ---------------------------------------------------------------- main loop
            sinks = Sinks(raw)
            order = interleave_prompts(prompts)
            n_total = len(order)
            n_check = max(1, int(math.ceil(float(cfg["projection_check_fraction"]) * n_total)))
            limit_s = float(cfg["projected_gpu_limit_hours"]) * 3600.0
            ckpt_every = int(cfg["checkpoint_every_prompts"])
            max_consec = int(cfg["max_consecutive_row_failures"])
            errors: List[str] = summary["errors"]
            slog(f"=== MAIN: {n_total} prompts x {len(by_cid)} conditions = {expected_rows} rows ===")
            t_main0 = time.perf_counter()
            per_set_time: Dict[str, float] = {s: 0.0 for s in SET_ORDER}
            for pi, p in enumerate(order):
                t_p0 = time.perf_counter()
                base: Optional[Baseline] = None
                for chunk in chunks:
                    base = process_rows_safely(ctx, p, chunk, base, sinks, errors, max_consec)
                dt = time.perf_counter() - t_p0
                per_set_time[p.pset] += dt
                n_done = pi + 1
                slog(f"[{n_done}/{n_total}] {p.pset}[{p.idx}] T={p.n_tokens} {dt:.1f}s "
                     f"rows={sinks.n_gate} fails={sinks.n_fail}")
                if n_done == n_check:
                    per = (time.perf_counter() - t_main0) / n_done
                    proj_main = per * n_total
                    proj_total = (t_main0 - t_run0) + proj_main
                    summary["projection"] = {
                        "prompts_timed": n_done, "sec_per_prompt": per,
                        "projected_main_hours": proj_main / 3600.0,
                        "projected_total_hours": proj_total / 3600.0,
                        "limit_hours": limit_s / 3600.0,
                    }
                    slog("=== SANITY 11: projected time after "
                         f"{n_done} prompts ({n_done / n_total:.1%}): main={proj_main/3600:.2f} h, "
                         f"total incl. setup={proj_total/3600:.2f} h, limit={limit_s/3600:.2f} h ===")
                    if proj_total > limit_s:
                        sinks.flush()
                        raise RunStop(f"SANITY11: projected {proj_total/3600:.2f} h > "
                                      f"{limit_s/3600:.2f} h")
                if n_done % ckpt_every == 0:
                    sinks.flush()
                    sinks.save_sparse()
                    write_summary()
                    if volume_commit is not None:
                        try:
                            volume_commit()
                        except Exception as ve:  # noqa: BLE001
                            slog(f"volume_commit failed: {ve}")

            # ---------------------------------------------------------------- finish
            sinks.flush()
            sinks.save_sparse()
            summary["rows"] = {
                "expected": expected_rows, "gate_scalars": sinks.n_gate,
                "residual_scalars": sinks.n_res, "logits_ld": sinks.n_ld,
                "failed": sinks.n_fail,
                "sparse_L45": len(sinks.sparse[45]), "sparse_L50": len(sinks.sparse[50]),
            }
            summary["per_set_seconds"] = per_set_time
            summary["main_seconds"] = time.perf_counter() - t_main0
            summary["total_seconds"] = time.perf_counter() - t_run0
            complete = (sinks.n_gate == expected_rows and sinks.n_fail == 0)
            summary["status"] = "COMPLETE" if complete else "INCOMPLETE"
            slog(f"DONE status={summary['status']} rows={sinks.n_gate}/{expected_rows} "
                 f"failed={sinks.n_fail} total={summary['total_seconds']/3600:.2f} h")
            sinks.close()
            sinks = None
            write_summary()
            return summary

        except RunStop as e:
            slog(f"STOP: {e}")
            summary["status"] = f"STOPPED: {e}"
            summary["errors"].append(str(e))
            if sinks is not None:
                sinks.flush()
                sinks.save_sparse()
                sinks.close()
            write_summary()
            return summary
        except Exception as e:  # noqa: BLE001
            slog(f"ABORT: {type(e).__name__}: {e}")
            slog(traceback.format_exc())
            summary["status"] = f"ABORTED: {type(e).__name__}: {e}"
            summary["errors"].append(f"{type(e).__name__}: {e}")
            if sinks is not None:
                try:
                    sinks.flush()
                    sinks.save_sparse()
                    sinks.close()
                except Exception:  # noqa: BLE001
                    pass
            write_summary()
            raise
        finally:
            slog.close()


if __name__ == "__main__":
    run_wo_003(
        model_dir=os.environ["SFO_MODEL_DIR"],
        revision_sha=os.environ["SFO_REVISION_SHA"],
        config_path=os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_003.yaml")),
        results_dir=os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-003")),
        ledger_path=os.environ.get("SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")),
        transcoder_root=os.environ["SFO_TRANSCODER_ROOT"],
        wo001_vectors_path=str(REPO_ROOT / "results/WO-001/raw/concept_vectors_L37.pt"),
        wo001_snapshot_path=str(REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json"),
        wo002b_directions_path=str(REPO_ROOT / "results/WO-002b/raw/directions_B1p.pt"),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )
