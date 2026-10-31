"""WO-005: inverted-polarity robustness — logit pass + generations (GPU)."""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from sfo.hooks import add_to_residual_batched, _resolve_layers  # noqa: E402
from sfo.ld import yes_no_token_ids  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import format_chat, get_d_model, get_layers, load_model, load_tokenizer  # noqa: E402
from sfo.prompts import YESBIAS_QUESTIONS, build_yesbias_prompt, decode_new_tokens  # noqa: E402
from sfo.prompts_polarity import (  # noqa: E402
    PAIR_IDS,
    S3T_QUESTIONS,
    all_fi_wordings,
    build_framed_prompt,
    c_messages,
    fi_messages,
    late_start_pos,
    p0f_messages,
    p0i_messages,
    pair_turn3,
)
from sfo.provenance import collect_env  # noqa: E402
from sfo.vectors import compute_start_pos, format_chat_text, pad_batch  # noqa: E402
from wo_003_run import Logger, RunStop, SanityAbort, _chat_prompt, _load_boolq, _py  # noqa: E402

EXPECTED_STEERED = 122
EXPECTED_C = 132
ROWS_PER_STEERED = 1 + 1 + 100 + 40 + 50  # base + α0 + concept×2α + random×2α + late
EXPECTED_LOGIT = EXPECTED_STEERED * ROWS_PER_STEERED + EXPECTED_C  # 23424 + 132
EXPECTED_GENS = 12 * 150


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
        return int(self.ids.shape[1] if self.ids.ndim == 2 else self.ids.numel())


@dataclass
class RowSpec:
    row_kind: str  # base | alpha0 | concept | random | late
    condition: str
    alpha: Optional[float] = None
    concept: Optional[str] = None
    seed: Optional[int] = None
    vec: Optional[torch.Tensor] = None
    start_override: Optional[int] = None  # late: T-1


def distinct2(text: str) -> Optional[float]:
    w = text.split()
    if len(w) < 2:
        return None
    bg = list(zip(w[:-1], w[1:]))
    return len(set(bg)) / len(bg)


def max_run(text: str) -> int:
    w = [x.lower() for x in text.split()]
    best, cur, prev = 0, 0, None
    for x in w:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def is_degenerate(text: str) -> Tuple[Optional[float], bool]:
    d2 = distinct2(text)
    degen = (d2 is not None and d2 < 0.5) or max_run(text) >= 8
    return d2, bool(degen)


def top1_class(tok: str, yes_id: int, no_id: int, tid: int) -> str:
    if tid == yes_id:
        return "Yes"
    if tid == no_id:
        return "No"
    return "other"


@contextmanager
def _residual_checked(
    model: Any, layer_idx: int, vectors: torch.Tensor, starts: torch.Tensor, T: int
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
        st = starts.to(device=device, dtype=torch.long)
        abs_idx = torch.arange(state["abs_pos"], state["abs_pos"] + tlen, device=device)
        mask = abs_idx.unsqueeze(0) >= st.unsqueeze(1)
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


def build_concept_vecs(path: Path) -> Tuple[List[str], Dict[str, torch.Tensor], Dict[str, float]]:
    pack = torch.load(path, map_location="cpu", weights_only=False)
    concepts = list(pack["concepts"])
    raws: Dict[str, torch.Tensor] = {}
    norms: Dict[str, float] = {}
    for i, c in enumerate(concepts):
        v = pack["vectors"][i].float()
        norms[c] = float(v.norm())
        raws[c] = v
    return concepts, raws, norms


def scale_vec(raw: torch.Tensor, alpha: float) -> torch.Tensor:
    return (raw * float(alpha)).to(torch.bfloat16)


def build_random_vec(d_model: int, seed: int, target_norm: float) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    r = torch.randn(d_model, generator=g, dtype=torch.float32)
    r = r / r.norm() * float(target_norm)
    return r.to(torch.bfloat16)


def build_all_prompts(tokenizer: Any, cfg: dict, concepts: Sequence[str]) -> Tuple[List[Prompt], List[Prompt], Dict]:
    prompts: List[Prompt] = []
    info: Dict[str, Any] = {"dataset_revisions": {}}
    ns = [int(n) for n in cfg["trial_ns"]]
    ge1 = int(cfg.get("start_pos_ge1", 1))

    # F / I
    for pol, pset in (("F", "F"), ("I", "I")):
        idx = 0
        for pid in PAIR_IDS:
            for n in ns:
                msgs = fi_messages(pid, pol, n)
                text, ids, sp = build_framed_prompt(tokenizer, msgs, n)
                prompts.append(
                    Prompt(
                        pset,
                        idx,
                        ids.cpu() if hasattr(ids, "cpu") else ids,
                        int(sp),
                        text,
                        {"pair": pid, "polarity": pol, "n": n, "turn3": msgs[-1]["content"]},
                    )
                )
                idx += 1

    # P0F / P0I
    for pset, builder in (("P0F", p0f_messages), ("P0I", p0i_messages)):
        for i, n in enumerate(cfg["p0_trial_ns"]):
            msgs = builder(int(n))
            text, ids, sp = build_framed_prompt(tokenizer, msgs, int(n))
            prompts.append(
                Prompt(
                    pset,
                    i,
                    ids.cpu() if hasattr(ids, "cpu") else ids,
                    int(sp),
                    text,
                    {"pair": "P0", "polarity": "F" if pset == "P0F" else "I", "n": int(n), "turn3": msgs[-1]["content"]},
                )
            )

    # S3
    tn3 = int(cfg.get("s3_trial_n", 1))
    for i, q in enumerate(YESBIAS_QUESTIONS):
        text, ids, sp = build_yesbias_prompt(tokenizer, tn3, q)
        prompts.append(Prompt("S3", i, ids.cpu(), int(sp), text, {"question": q, "n": tn3}))

    # S3T
    for i, q in enumerate(S3T_QUESTIONS):
        text, ids, sp = build_yesbias_prompt(tokenizer, tn3, q)
        prompts.append(Prompt("S3T", i, ids.cpu(), int(sp), text, {"question": q, "n": tn3}))

    # S4
    items, rev = _load_boolq(cfg)
    info["dataset_revisions"]["boolq"] = rev
    tmpl = cfg["boolq"]["user_template"]
    for i, it in enumerate(items):
        user = tmpl.format(q=it["question"])
        text, ids = _chat_prompt(tokenizer, [{"role": "user", "content": user}])
        prompts.append(Prompt("S4", i, ids, ge1, text, {"orig_idx": it["orig_idx"]}))

    assert len(prompts) == EXPECTED_STEERED, f"steered prompts {len(prompts)} != {EXPECTED_STEERED}"

    # Set C
    c_prompts: List[Prompt] = []
    c_concepts = list(concepts)[: int(cfg["c_n_concepts"])]
    wordings = all_fi_wordings(1)
    ci = 0
    for w in wordings:
        for word in c_concepts:
            msgs = c_messages(w["turn3"], inject_word=word)
            # Trial n=1 still in wording after prefix — anchor on Trial 1
            text, ids, sp = build_framed_prompt(tokenizer, msgs, 1)
            c_prompts.append(
                Prompt(
                    "C",
                    ci,
                    ids.cpu() if hasattr(ids, "cpu") else ids,
                    int(sp),
                    text,
                    {
                        "pair": w["pair"],
                        "polarity": w["polarity"],
                        "n": 1,
                        "c_mode": "inject",
                        "word": word,
                        "turn3": msgs[-1]["content"],
                    },
                )
            )
            ci += 1
        msgs = c_messages(w["turn3"], inject_word=None)
        text, ids, sp = build_framed_prompt(tokenizer, msgs, 1)
        c_prompts.append(
            Prompt(
                "C",
                ci,
                ids.cpu() if hasattr(ids, "cpu") else ids,
                int(sp),
                text,
                {
                    "pair": w["pair"],
                    "polarity": w["polarity"],
                    "n": 1,
                    "c_mode": "noinject",
                    "word": None,
                    "turn3": msgs[-1]["content"],
                },
            )
        )
        ci += 1
    assert len(c_prompts) == EXPECTED_C, f"C prompts {len(c_prompts)} != {EXPECTED_C}"
    return prompts, c_prompts, info


def pack_row_batches(
    base: RowSpec,
    alpha0: RowSpec,
    others: Sequence[RowSpec],
    batch_size: int,
) -> List[List[RowSpec]]:
    """Each batch: [base, alpha0, ...complete others]; never split groups of size 1."""
    max_extra = batch_size - 2
    assert max_extra >= 1
    batches: List[List[RowSpec]] = []
    i = 0
    while i < len(others):
        chunk = list(others[i : i + max_extra])
        batches.append([base, alpha0] + chunk)
        i += len(chunk)
    return batches


@torch.no_grad()
def forward_logit_batch(
    model: Any,
    device: torch.device,
    ids1: torch.Tensor,
    default_start: int,
    rows: Sequence[RowSpec],
    steer_layer: int,
    d_model: int,
    supports_ltk: bool,
) -> Tuple[torch.Tensor, bool]:
    B = len(rows)
    assert rows[0].row_kind == "base"
    ids = ids1.to(device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    ids = ids.expand(B, -1).contiguous()
    T = int(ids.shape[1])
    zd = torch.zeros(d_model, dtype=torch.bfloat16, device=device)
    vecs = torch.stack(
        [(r.vec.to(device) if r.vec is not None else zd) for r in rows]
    )
    for b, r in enumerate(rows):
        if r.row_kind in ("base", "alpha0") or r.condition in ("base", "alpha0"):
            vecs[b] = 0
    starts = torch.tensor(
        [int(r.start_override) if r.start_override is not None else int(default_start) for r in rows],
        device=device,
        dtype=torch.long,
    )
    kw: Dict[str, Any] = dict(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False
    )
    if supports_ltk:
        kw["logits_to_keep"] = 1
    with _residual_checked(model, steer_layer, vecs, starts, T):
        try:
            out = model(**kw)
        except TypeError as e:
            if "logits_to_keep" in str(e) and supports_ltk:
                supports_ltk = False
                kw.pop("logits_to_keep")
                out = model(**kw)
            else:
                raise
    return out.logits[:, -1, :].detach(), supports_ltk


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    id_list: List[torch.Tensor],
    vectors: Optional[torch.Tensor],
    starts_unpadded: List[int],
    layer_idx: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
) -> List[str]:
    device = next(model.parameters()).device
    torch.manual_seed(int(seed))
    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
    starts_padded = torch.tensor(
        [pl + sp for pl, sp in zip(pad_lens, starts_unpadded)],
        device=device,
        dtype=torch.long,
    )
    prompt_lens = [ids.shape[-1] if ids.ndim == 2 else ids.numel() for ids in id_list]
    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if vectors is None:
        out = model.generate(**gen_kwargs)
    else:
        with add_to_residual_batched(model, layer_idx, vectors.to(device), starts_padded):
            out = model.generate(**gen_kwargs)
    responses = []
    for i in range(out.shape[0]):
        row = out[i, pad_lens[i] :]
        responses.append(decode_new_tokens(tokenizer, row, prompt_lens[i]))
    return responses


def row_record(
    p: Prompt,
    spec: RowSpec,
    logits: torch.Tensor,
    yes_id: int,
    no_id: int,
    tokenizer: Any,
    batch_size: int,
) -> Dict[str, Any]:
    lg = logits.float()
    ld = float(lg[no_id] - lg[yes_id])
    top = torch.topk(lg, 5)
    tops = [
        {"id": int(i), "tok": tokenizer.decode([int(i)]), "logit": _py(float(v))}
        for v, i in zip(top.values, top.indices)
    ]
    tid = int(top.indices[0])
    rec: Dict[str, Any] = {
        "set": p.pset,
        "prompt_idx": p.idx,
        "row_kind": spec.row_kind,
        "condition": spec.condition,
        "alpha": spec.alpha,
        "batch_size": batch_size,
        "ld": _py(ld),
        "logit_yes": _py(float(lg[yes_id])),
        "logit_no": _py(float(lg[no_id])),
        "top5": tops,
        "top1_class": top1_class(tops[0]["tok"], yes_id, no_id, tid),
        "start_pos": int(spec.start_override) if spec.start_override is not None else p.start_pos,
        "n_tokens": p.n_tokens,
        **{k: p.meta.get(k) for k in ("pair", "polarity", "n", "question", "c_mode", "word", "turn3") if k in p.meta},
    }
    if spec.concept is not None:
        rec["concept"] = spec.concept
    if spec.seed is not None:
        rec["seed"] = spec.seed
    return rec


def load_wo004_ld(path: Path) -> Dict[Tuple[str, int, str], float]:
    out: Dict[Tuple[str, int, str], float] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("condition") == "base" or r.get("row_kind") in ("base", "baseline"):
                out[(r["set"], int(r["prompt_idx"]), "baseline")] = float(r["ld"])
            elif r.get("condition") == "steer" and r.get("row_kind") == "concept":
                cid = f"concept__{r['concept']}__a4"
                out[(r["set"], int(r["prompt_idx"]), cid)] = float(r["ld"])
    return out


def load_wo004_prompt_ids(path: Path) -> Dict[Tuple[str, int], List[int]]:
    out: Dict[Tuple[str, int], List[int]] = {}
    if not path.exists():
        return out
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            out[(r["set"], int(r["prompt_idx"]))] = list(r["ids"])
    return out


def _gen_prompt_key(r: dict) -> Tuple[Any, ...]:
    return (r.get("set"), r.get("pair"), int(r.get("prompt_idx", -1)), int(r.get("n", -1)))


def _load_complete_gens(path: Path) -> Tuple[List[dict], set]:
    """Keep only fully complete prompts (150 rows); drop partial tails."""
    if not path.exists():
        return [], set()
    by: Dict[Tuple[Any, ...], List[dict]] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            by.setdefault(_gen_prompt_key(r), []).append(r)
    kept: List[dict] = []
    done = set()
    for key, lst in by.items():
        if len(lst) == 150:
            kept.extend(lst)
            done.add(key)
    return kept, done


def run_wo_005(
    model_dir,
    revision_sha,
    config_path,
    results_dir,
    ledger_path,
    wo001_vectors_path,
    wo004_rows_path,
    wo004_prompts_path,
    git_hash_str,
    gpu_type="A100-80GB",
    n_cores=8.0,
    mem_gib=64.0,
    volume_commit=None,
    resume: bool = False,
) -> dict:
    t0 = time.perf_counter()
    results_path = Path(results_dir)
    raw = results_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    slog = Logger(raw / "sanity_log.txt")
    cfg = yaml.safe_load(Path(config_path).read_text())
    summary: Dict[str, Any] = {
        "wo_id": "WO-005",
        "git_hash": git_hash_str,
        "model_revision": revision_sha,
        "status": "RUNNING",
        "errors": [],
        "batch_size": int(cfg["batch_size"]),
        "resume": bool(resume),
    }
    if resume and (results_path / "run_summary.json").exists():
        prev = json.loads((results_path / "run_summary.json").read_text())
        for k in ("dataset_revisions", "env", "projection", "replication_max_abs_ld", "n_logit_rows"):
            if k in prev:
                summary[k] = prev[k]

    def write_summary():
        (results_path / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    logit_path = raw / "logit_rows.jsonl"
    gens_path = raw / "gens.jsonl"
    skip_logit = False
    if resume and logit_path.exists():
        n_existing = sum(1 for line in logit_path.open() if line.strip())
        if n_existing == EXPECTED_LOGIT:
            skip_logit = True
            slog(f"=== RESUME: skip logit (n={n_existing}) ===")
        else:
            raise SanityAbort(
                f"RESUME: logit_rows.jsonl has {n_existing} rows, need {EXPECTED_LOGIT}"
            )

    kept_gens: List[dict] = []
    done_gen_keys: set = set()
    if resume:
        kept_gens, done_gen_keys = _load_complete_gens(gens_path)
        slog(
            f"=== RESUME: keeping {len(kept_gens)} gens "
            f"across {len(done_gen_keys)} complete prompts ==="
        )
        with gens_path.open("w") as gf:
            for r in kept_gens:
                gf.write(json.dumps(r) + "\n")

    f_logit = None if skip_logit else logit_path.open("w")
    f_gens = gens_path.open("a" if resume else "w")

    with gpu_ledger(
        ledger_path,
        wo_id="WO-005",
        function="wo_005_run_resume" if resume else "wo_005_run",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            if not git_hash_str or str(git_hash_str).lower() in ("unknown", ""):
                raise SanityAbort("SANITY1: git hash missing")
            slog(f"=== SANITY1 provenance git={git_hash_str} rev={revision_sha} resume={resume} ===")

            alphas = [float(a) for a in cfg["alphas"]]
            assert alphas == [2.0, 4.0]
            bs = int(cfg["batch_size"])
            concepts, raw_vecs, norms = build_concept_vecs(Path(wo001_vectors_path))
            assert len(concepts) == int(cfg["n_concepts"]) == 50
            rand_seeds = [int(s) for s in cfg["random_seeds"]]
            assert len(rand_seeds) == 20

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            prompts, c_prompts, pinfo = build_all_prompts(tokenizer, cfg, concepts)
            summary["dataset_revisions"] = pinfo.get("dataset_revisions", {})

            # SANITY2: P0 token ids vs WO-004
            wo004_ids = load_wo004_prompt_ids(Path(wo004_prompts_path))
            for p in prompts:
                if p.pset == "P0F":
                    key = ("S1", p.idx)
                elif p.pset == "P0I":
                    key = ("S8", p.idx)
                else:
                    continue
                if key not in wo004_ids:
                    continue
                got = p.ids[0].tolist() if p.ids.ndim == 2 else p.ids.tolist()
                if got != wo004_ids[key]:
                    raise SanityAbort(f"SANITY2 token ids mismatch {p.pset}[{p.idx}] vs WO-004 {key}")
            slog("SANITY2 token-id match for P0F/P0I n=1..3 vs WO-004 S1/S8: PASS")

            model = load_model(model_dir, revision=revision_sha, device=0)
            device = next(model.parameters()).device
            d_model = get_d_model(model)
            yes_id, no_id = yes_no_token_ids(tokenizer)
            assert yes_id == int(cfg["yes_id"]) and no_id == int(cfg["no_id"])
            summary["env"] = collect_env()
            steer_layer = int(cfg["layer_steer"])
            supports_ltk = True
            zd = torch.zeros(d_model, dtype=torch.bfloat16)

            # Precompute steered vectors
            c_vecs: Dict[Tuple[str, float], torch.Tensor] = {}
            for c in concepts:
                for a in alphas:
                    c_vecs[(c, a)] = scale_vec(raw_vecs[c], a)
            r_vecs: Dict[Tuple[int, float], torch.Tensor] = {}
            for i, s in enumerate(rand_seeds):
                for a in alphas:
                    r_vecs[(s, a)] = build_random_vec(d_model, s, float(a) * norms[concepts[i]])

            limit_s = float(cfg["projected_gpu_limit_hours"]) * 3600.0
            rep_max = float(summary.get("replication_max_abs_ld") or 0.0)
            n_logit = int(summary.get("n_logit_rows") or 0)

            if not skip_logit:
                # SANITY3: anchors for each wording n=1
                slog("=== SANITY3 anchors (n=1) ===")
                for w in all_fi_wordings(1):
                    msgs = fi_messages(w["pair"], w["polarity"], 1)
                    text, ids, sp = build_framed_prompt(tokenizer, msgs, 1)
                    T = int(ids.shape[1])
                    late = late_start_pos(T)
                    assert late == T - 1
                    slog(
                        f"{w['pair']}-{w['polarity']} start={sp} T={T} late={late} "
                        f"turn3={w['turn3'][:60]!r}..."
                    )
                slog("SANITY3 PASS")

                # SANITY4 top-5 base for F,I,S3T,C n=1
                slog("=== SANITY4 base top-5 ===")
                for p in prompts + c_prompts:
                    if p.pset not in ("F", "I", "S3T", "C"):
                        continue
                    if p.meta.get("n") not in (1, None) and p.pset != "C":
                        continue
                    if p.pset in ("F", "I") and int(p.meta.get("n", -1)) != 1:
                        continue
                    lg, supports_ltk = forward_logit_batch(
                        model, device, p.ids, p.start_pos,
                        [RowSpec("base", "base", vec=zd)],
                        steer_layer, d_model, supports_ltk,
                    )
                    top = torch.topk(lg[0].float(), 5)
                    tops = [
                        (tokenizer.decode([int(i)]), round(float(v), 3))
                        for v, i in zip(top.values, top.indices)
                    ]
                    slog(f"{p.pset}[{p.idx}] pair={p.meta.get('pair')} top5={tops}")

                # SANITY5 identity
                slog("=== SANITY5 identity ===")
                p0 = next(p for p in prompts if p.pset == "F")
                c0 = concepts[0]
                rows_id = [
                    RowSpec("base", "base", vec=zd),
                    RowSpec("alpha0", "alpha0", alpha=0.0, vec=zd),
                    RowSpec("concept", "steer", alpha=4.0, concept=c0, vec=c_vecs[(c0, 4.0)]),
                ]
                lg, supports_ltk = forward_logit_batch(
                    model, device, p0.ids, p0.start_pos, rows_id, steer_layer, d_model, supports_ltk
                )
                if not torch.equal(lg[0], lg[1]):
                    raise SanityAbort("SANITY5 alpha0 != base")
                ld0 = float(lg[0].float()[no_id] - lg[0].float()[yes_id])
                ld4 = float(lg[2].float()[no_id] - lg[2].float()[yes_id])
                if ld0 == ld4:
                    raise SanityAbort("SANITY5 alpha4 did not change LD")
                slog(f"SANITY5 PASS base_LD={ld0:.4f} a4_LD={ld4:.4f}")

                wo004_ld = load_wo004_ld(Path(wo004_rows_path))
                rep_max = 0.0
                max_rep = float(cfg["replication_max_abs_ld"])
                n_logit = 0
                n_check = max(
                    1, int(math.ceil(float(cfg["projection_check_fraction"]) * EXPECTED_STEERED))
                )
                t_logit0 = time.perf_counter()

                def _maybe_rep(p: Prompt, spec: RowSpec, rec: dict) -> None:
                    nonlocal rep_max
                    if p.pset not in ("P0F", "P0I"):
                        return
                    wo_set = "S1" if p.pset == "P0F" else "S8"
                    if spec.row_kind == "base":
                        key = (wo_set, p.idx, "baseline")
                        if key in wo004_ld:
                            d = abs(float(rec["ld"]) - wo004_ld[key])
                            rep_max = max(rep_max, d)
                            if d > max_rep:
                                raise RunStop(f"SANITY2 LD base diff {d} > {max_rep} {key}")
                    if (
                        spec.condition == "steer"
                        and spec.row_kind == "concept"
                        and spec.alpha == 4.0
                    ):
                        key = (wo_set, p.idx, f"concept__{spec.concept}__a4")
                        if key in wo004_ld:
                            d = abs(float(rec["ld"]) - wo004_ld[key])
                            rep_max = max(rep_max, d)
                            if d > max_rep:
                                raise RunStop(f"SANITY2 LD steer diff {d} > {max_rep} {key}")

                assert f_logit is not None
                # ---- Logit pass (steered sets)
                for pi, p in enumerate(prompts):
                    base = RowSpec("base", "base", vec=zd)
                    a0 = RowSpec("alpha0", "alpha0", alpha=0.0, vec=zd)
                    others: List[RowSpec] = []
                    for c in concepts:
                        for a in alphas:
                            others.append(
                                RowSpec("concept", "steer", alpha=a, concept=c, vec=c_vecs[(c, a)])
                            )
                    for i, s in enumerate(rand_seeds):
                        for a in alphas:
                            others.append(
                                RowSpec("random", "steer", alpha=a, seed=s, vec=r_vecs[(s, a)])
                            )
                    for c in concepts:
                        others.append(
                            RowSpec(
                                "late",
                                "late",
                                alpha=4.0,
                                concept=c,
                                vec=c_vecs[(c, 4.0)],
                                start_override=late_start_pos(p.n_tokens),
                            )
                        )
                    assert len(others) == 190
                    logged_base = False
                    for batch in pack_row_batches(base, a0, others, bs):
                        lg, supports_ltk = forward_logit_batch(
                            model, device, p.ids, p.start_pos, batch, steer_layer, d_model, supports_ltk
                        )
                        for b, spec in enumerate(batch):
                            if spec.row_kind in ("base", "alpha0") and logged_base:
                                continue
                            rec = row_record(p, spec, lg[b], yes_id, no_id, tokenizer, bs)
                            f_logit.write(json.dumps(rec) + "\n")
                            n_logit += 1
                            _maybe_rep(p, spec, rec)
                        logged_base = True

                    if (pi + 1) == n_check:
                        per = (time.perf_counter() - t_logit0) / (pi + 1)
                        remain = EXPECTED_STEERED - (pi + 1)
                        proj = (time.perf_counter() - t0) + per * remain
                        proj += per * 0.2 * EXPECTED_C
                        proj += 0.45 * 3600  # gen buffer; refined after gen 5%
                        summary["projection"] = {
                            "phase": "logit_5pct",
                            "sec_per_prompt": per,
                            "projected_total_hours": proj / 3600.0,
                        }
                        slog(
                            f"=== SANITY7 logit 5%: projected ~{proj/3600:.2f}h "
                            f"(limit {limit_s/3600:.2f}h) ==="
                        )
                        if proj > limit_s:
                            raise RunStop(f"SANITY7: projected {proj/3600:.2f}h > limit")

                    if (pi + 1) % int(cfg.get("checkpoint_every_prompts", 20)) == 0:
                        f_logit.flush()
                        write_summary()
                        if volume_commit:
                            try:
                                volume_commit()
                            except Exception as ve:  # noqa: BLE001
                                slog(f"volume_commit: {ve}")

                # Set C: unsteered only
                for p in c_prompts:
                    lg, supports_ltk = forward_logit_batch(
                        model, device, p.ids, p.start_pos,
                        [RowSpec("base", "base", vec=zd)],
                        steer_layer, d_model, supports_ltk,
                    )
                    rec = row_record(
                        p, RowSpec("base", "base", vec=zd), lg[0], yes_id, no_id, tokenizer, 1
                    )
                    f_logit.write(json.dumps(rec) + "\n")
                    n_logit += 1
                f_logit.flush()
                summary["n_logit_rows"] = n_logit
                summary["replication_max_abs_ld"] = rep_max
                slog(f"Logit pass done n_logit={n_logit} (expected {EXPECTED_LOGIT}) rep_max={rep_max}")
                if n_logit != EXPECTED_LOGIT:
                    slog(f"WARN n_logit={n_logit} != expected {EXPECTED_LOGIT}")
            else:
                slog(
                    f"RESUME: logit already complete n_logit={n_logit} "
                    f"rep_max={rep_max}"
                )

            # ---- Generation pass (F∪I, P1–P6, n=1)
            gen_prompts = [
                p for p in prompts
                if p.pset in ("F", "I") and int(p.meta.get("n", -1)) == 1
            ]
            assert len(gen_prompts) == 12, len(gen_prompts)
            seed_base = int(cfg["seed_base"])
            max_new = int(cfg["max_new_tokens"])
            temperature = float(cfg["temperature"])
            top_k = int(cfg["top_k"])
            top_p = float(cfg["top_p"])
            gbs = int(cfg["gen_batch_size"])
            n_gens = len(kept_gens)
            todo_prompts = [
                p
                for p in gen_prompts
                if (p.pset, p.meta["pair"], p.idx, int(p.meta["n"])) not in done_gen_keys
            ]
            slog(
                f"Gen pass: {len(todo_prompts)} prompts remaining "
                f"(skip {len(done_gen_keys)} complete)"
            )
            t_gen0 = time.perf_counter()
            n_gen_check = max(1, int(math.ceil(0.05 * max(1, len(todo_prompts)))))

            for gi, p in enumerate(todo_prompts):
                # base × 30
                for k in range(int(cfg["base_gen_repeats"])):
                    seed = seed_base + k
                    resps = generate_batch(
                        model, tokenizer, [p.ids], None, [p.start_pos],
                        steer_layer, max_new, temperature, top_k, top_p, seed,
                    )
                    d2, degen = is_degenerate(resps[0])
                    f_gens.write(json.dumps({
                        "set": p.pset, "pair": p.meta["pair"], "n": p.meta["n"],
                        "polarity": p.meta["polarity"], "prompt_idx": p.idx,
                        "condition": "base", "alpha": None, "sample_k": k,
                        "seed": seed, "prompt": p.meta.get("turn3"),
                        "response": resps[0], "distinct2": d2, "degenerate": degen,
                        "judge_raw": None, "claims": None,
                    }) + "\n")
                    n_gens += 1

                # concept α=2 / α=4 (batched; seed = seed_base+off+first_ci for batch)
                for a, seed_off in ((2.0, 1000), (4.0, 2000)):
                    for i0 in range(0, len(concepts), gbs):
                        chunk = list(enumerate(concepts))[i0 : i0 + gbs]
                        id_list = [p.ids for _ in chunk]
                        starts = [p.start_pos] * len(chunk)
                        vec = torch.stack([c_vecs[(c, a)] for _, c in chunk]).to(device)
                        seed = seed_base + seed_off + chunk[0][0]
                        resps = generate_batch(
                            model, tokenizer, id_list, vec, starts,
                            steer_layer, max_new, temperature, top_k, top_p, seed,
                        )
                        for j, (ci, c) in enumerate(chunk):
                            d2, degen = is_degenerate(resps[j])
                            f_gens.write(json.dumps({
                                "set": p.pset, "pair": p.meta["pair"], "n": p.meta["n"],
                                "polarity": p.meta["polarity"], "prompt_idx": p.idx,
                                "condition": "concept", "alpha": a, "concept": c,
                                "concept_index": ci,
                                "seed": seed_base + seed_off + ci,
                                "prompt": p.meta.get("turn3"), "response": resps[j],
                                "distinct2": d2, "degenerate": degen,
                                "judge_raw": None, "claims": None,
                            }) + "\n")
                            n_gens += 1

                # random α=4
                for i0 in range(0, len(rand_seeds), gbs):
                    chunk = rand_seeds[i0 : i0 + gbs]
                    id_list = [p.ids for _ in chunk]
                    starts = [p.start_pos] * len(chunk)
                    vec = torch.stack([r_vecs[(s, 4.0)] for s in chunk]).to(device)
                    seed = seed_base + 3000 + chunk[0]
                    resps = generate_batch(
                        model, tokenizer, id_list, vec, starts,
                        steer_layer, max_new, temperature, top_k, top_p, seed,
                    )
                    for j, s in enumerate(chunk):
                        d2, degen = is_degenerate(resps[j])
                        f_gens.write(json.dumps({
                            "set": p.pset, "pair": p.meta["pair"], "n": p.meta["n"],
                            "polarity": p.meta["polarity"], "prompt_idx": p.idx,
                            "condition": "random", "alpha": 4.0, "seed": s,
                            "gen_seed": seed_base + 3000 + s,
                            "prompt": p.meta.get("turn3"),
                            "response": resps[j], "distinct2": d2, "degenerate": degen,
                            "judge_raw": None, "claims": None,
                        }) + "\n")
                        n_gens += 1

                if (gi + 1) == n_gen_check:
                    per_g = (time.perf_counter() - t_gen0) / (gi + 1)
                    remain_g = len(todo_prompts) - (gi + 1)
                    # Include prior GPU time estimate from first-pass projection when resuming
                    prior_h = float(
                        (summary.get("projection") or {})
                        .get("phase_gen_5pct", {})
                        .get("projected_total_hours")
                        or 0.0
                    )
                    # Bound: this resume segment + already-spent first-pass estimate floor
                    proj = (time.perf_counter() - t0) + per_g * remain_g
                    if resume and prior_h > 0:
                        # Conservatively compare remaining-only wall to leftover budget
                        leftover = limit_s - prior_h * 3600.0
                        if leftover < 0:
                            leftover = limit_s  # first pass already near cap; still finish gens
                        if proj > leftover and proj > limit_s:
                            raise RunStop(
                                f"SANITY7 resume gen: projected segment {proj/3600:.2f}h "
                                f"> limit {limit_s/3600:.2f}h"
                            )
                    summary.setdefault("projection", {})["phase_gen_resume" if resume else "phase_gen_5pct"] = {
                        "sec_per_gen_prompt": per_g,
                        "projected_segment_hours": proj / 3600.0,
                        "n_todo": len(todo_prompts),
                    }
                    slog(
                        f"=== SANITY7 gen {'resume ' if resume else ''}5%: "
                        f"projected segment ~{proj/3600:.2f}h "
                        f"(limit {limit_s/3600:.2f}h) ==="
                    )
                    if not resume and proj > limit_s:
                        raise RunStop(f"SANITY7 gen: projected {proj/3600:.2f}h > limit")

                if (gi + 1) % 3 == 0:
                    f_gens.flush()
                    write_summary()
                    if volume_commit:
                        try:
                            volume_commit()
                        except Exception as ve:  # noqa: BLE001
                            slog(f"volume_commit: {ve}")

            f_gens.flush()
            summary["n_gens"] = n_gens
            summary["n_logit_rows"] = n_logit
            summary["replication_max_abs_ld"] = rep_max
            summary["total_seconds"] = time.perf_counter() - t0
            summary["status"] = "COMPLETE"
            if n_gens != EXPECTED_GENS:
                slog(f"WARN n_gens={n_gens} != {EXPECTED_GENS}")
            slog(
                f"DONE logit={n_logit} gens={n_gens} rep_max={rep_max:.4f} "
                f"t={summary['total_seconds']/3600:.2f}h resume={resume}"
            )
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
            if f_logit is not None:
                f_logit.close()
            f_gens.close()
            slog.close()


if __name__ == "__main__":
    run_wo_005(
        model_dir=os.environ["SFO_MODEL_DIR"],
        revision_sha=os.environ["SFO_REVISION_SHA"],
        config_path=os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_005.yaml")),
        results_dir=os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-005")),
        ledger_path=os.environ.get("SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")),
        wo001_vectors_path=str(REPO_ROOT / "results/WO-001/raw/concept_vectors_L37.pt"),
        wo004_rows_path=str(REPO_ROOT / "results/WO-004/raw/rows.jsonl"),
        wo004_prompts_path=str(REPO_ROOT / "results/WO-004/raw/prompts.jsonl"),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )
