"""WO-002b Part B: validate B1′ (shift-removal) at l0* with paired generation + steered-damage KL.

All generation goes through sfo.paired.paired_generate (never HF generate).
Arms: "none" (steering on, no blocker), "B1p" (B1′ with Part A directions at l0*),
"B1p-null" (B1′ with random unit directions, seed 0, same l0*).
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.ld import concept_match_string  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import format_chat, get_d_model, load_model, load_tokenizer  # noqa: E402
from sfo.paired import paired_forward_logits, paired_generate, unit  # noqa: E402
from sfo.prompts import (  # noqa: E402
    build_introspection_prompt,
    decode_new_tokens,
    format_chat_text,
    introspection_messages,
)
from sfo.provenance import collect_env  # noqa: E402
from sfo.vectors import compute_start_pos, pad_batch  # noqa: E402

ARMS = ("none", "B1p", "B1p-null")
N_INJ = 50 * 6 * 3
N_FID = 50 * 2 * 3
N_EXP = 50 * 2 * 3
EXPECTED_TRIALS = N_INJ + N_FID + N_EXP  # 1500
LAYER_LAST = 61


def _log(msg: str) -> None:
    print(msg, flush=True)


# ---------------------------------------------------------------- prompts
def build_fid_prompt(tokenizer, trial_n: int, prefill: str) -> Tuple[str, torch.Tensor, int]:
    messages = introspection_messages(trial_n)
    text = format_chat_text(tokenizer, messages) + prefill
    # tokenize without re-applying template
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    # ensure BOS at 0
    bos = tokenizer.bos_token_id
    if int(ids[0, 0].item()) != int(bos):
        ids = torch.cat([torch.tensor([[bos]]), ids], dim=1)
    assert int((ids[0] == bos).sum()) == 1
    # start_pos from the template portion only
    base_text = format_chat_text(tokenizer, messages)
    sp = compute_start_pos(tokenizer, base_text, trial_n)
    assert "<end_of_turn>" not in text.split(prefill)[-1]
    # prefill should not be followed by end_of_turn
    assert not text.rstrip().endswith("<end_of_turn>")
    return text, ids, sp


def build_exp_prompt(tokenizer, user: str) -> Tuple[str, torch.Tensor]:
    messages = [{"role": "user", "content": user}]
    text = format_chat_text(tokenizer, messages)
    ids = format_chat(tokenizer, messages)
    return text, ids


# ---------------------------------------------------------------- degeneracy
def distinct2(text: str) -> Optional[float]:
    """Unique whitespace bigrams / total bigrams (None if < 2 words)."""
    w = text.split()
    if len(w) < 2:
        return None
    bg = list(zip(w[:-1], w[1:]))
    return len(set(bg)) / len(bg)


def max_run(text: str) -> int:
    """Longest run of the same (case-insensitive) whitespace token."""
    w = [x.lower() for x in text.split()]
    best, cur = 0, 0
    prev = None
    for x in w:
        cur = cur + 1 if x == prev else 1
        prev = x
        best = max(best, cur)
    return best


def is_degenerate(text: str) -> Tuple[Optional[float], bool]:
    d2 = distinct2(text)
    degen = (d2 is not None and d2 < 0.5) or max_run(text) >= 8
    return d2, bool(degen)


# ---------------------------------------------------------------- generation
def _gen_batch(
    model,
    tokenizer,
    id_list: List[torch.Tensor],
    starts: List[int],
    steer_vecs: torch.Tensor,
    steer_layer: int,
    blocker_dirs: Optional[Dict[int, torch.Tensor]],
    l0: Optional[int],
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    seed: int,
    do_sample: bool,
) -> List[str]:
    """One paired_generate call over a homogeneous-arm batch. Returns decoded steered responses."""
    device = next(model.parameters()).device
    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
    st = torch.tensor([pl + s for pl, s in zip(pad_lens, starts)], device=device, dtype=torch.long)
    padded_len = int(input_ids.shape[1])
    steered_seqs, twin_seqs = paired_generate(
        model,
        input_ids,
        attn,
        st,
        steer_vecs.to(device=device, dtype=torch.bfloat16),
        steer_layer,
        blocker_dirs,
        l0,
        max_new_tokens=max_new,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
        do_sample=do_sample,
        max_layer=LAYER_LAST,
    )
    resps = []
    for s, t in zip(steered_seqs, twin_seqs):
        assert torch.equal(s, t), "twin integrity violated in paired_generate"
        resps.append(decode_new_tokens(tokenizer, s, padded_len))
    return resps


def run_part_b(
    *,
    model_dir: str,
    revision_sha: str,
    config_path: str,
    results_dir: str,
    ledger_path: str,
    wo001_vectors_path: str,
    wo001_snapshot_path: str,
    wo002_wikitext_path: str,
    git_hash_str: str,
    gpu_type: str = "A100-80GB",
    n_cores: float = 8.0,
    mem_gib: float = 64.0,
    volume_commit=None,
) -> Dict[str, Any]:
    t_run0 = time.perf_counter()
    results_path = Path(results_dir)
    raw = results_path / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    summary: Dict[str, Any] = {
        "git_hash": git_hash_str,
        "model_revision": revision_sha,
        "sanity": {},
        "errors": [],
    }

    with gpu_ledger(
        ledger_path,
        wo_id="WO-002b",
        function="wo_002b_partB",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            partA = json.loads((results_path / "partA_summary.json").read_text())
            if partA.get("stopped_before_partB"):
                _log(
                    "WARNING: Part A flagged stopped_before_partB "
                    f"({partA.get('stop_reason')}); continuing because Part B was invoked."
                )
            l0_star = int(partA["partA"]["l0_star"])
            directions = torch.load(
                raw / "directions_B1p.pt", map_location="cpu", weights_only=False
            )
            d_hat: Dict[int, torch.Tensor] = {int(k): v for k, v in directions["d_hat"].items()}
            layers = [int(l) for l in directions["layers"]]

            wo001 = json.loads(Path(wo001_snapshot_path).read_text())
            concepts_50 = list(wo001["concepts"])
            assert len(concepts_50) == 50
            pack50 = torch.load(wo001_vectors_path, map_location="cpu", weights_only=False)
            vec50 = {c: pack50["vectors"][i].float() for i, c in enumerate(pack50["concepts"])}
            for c in concepts_50:
                assert c in vec50, f"missing WO-001 vector for {c}"

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            model = load_model(model_dir, revision=revision_sha, device=0)
            device = next(model.parameters()).device
            d_model = get_d_model(model)
            summary["env"] = collect_env()
            summary["l0_star"] = l0_star

            alpha = float(cfg["alpha"])
            layer_steer = int(cfg["layer_steer"])
            pair_bs = int(cfg["pair_batch_size"])
            batch_size = int(cfg["batch_size"])
            temperature = float(cfg["temperature"])
            top_k = int(cfg["top_k"])
            top_p = float(cfg["top_p"])
            max_new = int(cfg["max_new_tokens"])
            seed_base = int(cfg["seed_base"])
            prefill = str(cfg["fid_prefill"])
            fid_new = int(cfg["fid_new_tokens"])
            exp_new = int(cfg["exp_new_tokens"])
            exp_user = str(cfg["exp_user"])
            _log(
                f"Config: l0*={l0_star} alpha={alpha} layer_steer={layer_steer} "
                f"pair_batch_size={pair_bs} seed_base={seed_base}"
            )

            # ---- arms: (blocker_dirs, l0)
            g0 = torch.Generator().manual_seed(int(cfg["b1_null_seed"]))
            null_dirs = {l: unit(torch.randn(d_model, generator=g0)) for l in layers}
            arm_cfg: Dict[str, Tuple[Optional[Dict[int, torch.Tensor]], Optional[int]]] = {
                "none": (None, None),
                "B1p": (d_hat, l0_star),
                "B1p-null": (null_dirs, l0_star),
            }

            # ---- sanity: B-fid prompt
            ftext, _fids, _fsp = build_fid_prompt(tokenizer, 1, prefill)
            _log(f"SANITY B-fid prompt:\n{ftext}")
            assert prefill in ftext
            assert "<end_of_turn>" not in ftext[ftext.find(prefill) :]
            _log("SANITY B-fid: prefill inside model turn, no end_of_turn after — PASS")

            # ---- sanity: α=0 identity (blocker vs none, bit-identical logits), cheap
            _, ids_t, sp_t = build_introspection_prompt(tokenizer, 1)
            ids_t = ids_t.to(device)
            nbi = 2
            ids_b = ids_t.expand(nbi, -1).contiguous()
            attn_b = torch.ones_like(ids_b)
            st_b = torch.full((nbi,), int(sp_t), dtype=torch.long, device=device)
            zero_v = torch.zeros(nbi, d_model, device=device, dtype=torch.bfloat16)
            a = paired_forward_logits(model, ids_b, attn_b, st_b, zero_v, layer_steer, None, None)
            b = paired_forward_logits(
                model, ids_b, attn_b, st_b, zero_v, layer_steer, d_hat, l0_star
            )
            ident_alpha0 = bool(torch.equal(a, b))
            del a, b
            torch.cuda.empty_cache()
            assert ident_alpha0, "α=0 identity failed (B1′ changed logits with zero steering)"
            _log("SANITY α=0 identity (B1′ vs none, bit-identical logits): PASS")
            summary["sanity"]["alpha0_identity"] = ident_alpha0

            # ---------------- resume check ----------------
            trials_path = raw / "partB_trials.jsonl"
            skip_gen = False
            if trials_path.exists():
                n_existing = sum(1 for _ in trials_path.open())
                if n_existing == EXPECTED_TRIALS:
                    _log(
                        f"Resuming: found {n_existing} trials on disk; "
                        "skipping generation, continuing with KL"
                    )
                    skip_gen = True
                else:
                    _log(f"Removing incomplete partB_trials.jsonl ({n_existing} lines)")
                    trials_path.unlink()
            batch_index = 0

            def write_rows(rows: List[dict]) -> None:
                with trials_path.open("a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")

            def run_batches(specs: List[dict], arm: str, sample: bool, max_tok: int) -> None:
                """Generate for homogeneous-arm specs in chunks of pair_batch_size."""
                nonlocal batch_index
                bdirs, bl0 = arm_cfg[arm]
                i = 0
                while i < len(specs):
                    chunk = specs[i : i + pair_bs]
                    id_list, starts, metas, vecs = [], [], [], []
                    for spec in chunk:
                        cond = spec["condition"]
                        if cond == "B-fid":
                            text, ids, sp = build_fid_prompt(tokenizer, int(spec["n"]), prefill)
                        elif cond == "B-exp":
                            text, ids = build_exp_prompt(tokenizer, exp_user)
                            sp = 1  # steer positions >= 1
                        else:  # B-inj
                            text, ids, sp = build_introspection_prompt(tokenizer, int(spec["n"]))
                        id_list.append(ids)
                        starts.append(sp)
                        metas.append({"prompt": text, "start_pos": sp})
                        vecs.append(alpha * vec50[spec["concept"]])
                    seed = seed_base + batch_index
                    steer_vecs = torch.stack([v.float() for v in vecs]).to(torch.bfloat16)
                    resps = _gen_batch(
                        model,
                        tokenizer,
                        id_list,
                        starts,
                        steer_vecs,
                        layer_steer,
                        bdirs,
                        bl0,
                        max_tok,
                        temperature,
                        top_k,
                        top_p,
                        seed,
                        do_sample=sample,
                    )
                    rows = []
                    for j, spec in enumerate(chunk):
                        d2, degen = is_degenerate(resps[j])
                        rows.append(
                            {
                                "condition": spec["condition"],
                                "arm": arm,
                                "concept": spec["concept"],
                                "n": spec["n"],
                                "seed": seed,
                                "batch_index": batch_index,
                                "start_pos": metas[j]["start_pos"],
                                "prompt": metas[j]["prompt"],
                                "response": resps[j],
                                "match": concept_match_string(spec["concept"], resps[j]),
                                "distinct2": d2,
                                "degenerate": degen,
                                "judge_detection_raw": None,
                                "judge_detection": None,
                                "judge_ident_raw": None,
                                "judge_ident": None,
                            }
                        )
                    write_rows(rows)
                    _log(
                        f"batch {batch_index}: {len(chunk)} {chunk[0]['condition']}/{arm} "
                        f"(seed {seed})"
                    )
                    batch_index += 1
                    i += len(chunk)

            # ---------------- generation ----------------
            t_gen = time.perf_counter()
            if not skip_gen:
                for arm in ARMS:
                    # B-inj: 50 concepts × n=1..6, sampled, 100 tokens
                    specs = [
                        {"condition": "B-inj", "concept": c, "n": n}
                        for c in concepts_50
                        for n in range(1, 7)
                    ]
                    run_batches(specs, arm, sample=True, max_tok=max_new)
                for arm in ARMS:
                    # B-fid: forced identification, n=1..2, greedy, 6 tokens
                    specs = [
                        {"condition": "B-fid", "concept": c, "n": n}
                        for c in concepts_50
                        for n in (1, 2)
                    ]
                    run_batches(specs, arm, sample=False, max_tok=fid_new)
                for arm in ARMS:
                    # B-exp: free-expression prompt, 2 samples, sampled, 64 tokens
                    specs = [
                        {"condition": "B-exp", "concept": c, "n": s}
                        for c in concepts_50
                        for s in (0, 1)
                    ]
                    run_batches(specs, arm, sample=True, max_tok=exp_new)
                _log(f"Generation done in {time.perf_counter() - t_gen:.1f}s")

            n_lines = sum(1 for _ in trials_path.open())
            _log(f"Wrote {n_lines} trials (expected {EXPECTED_TRIALS})")
            assert n_lines == EXPECTED_TRIALS, f"got {n_lines} != {EXPECTED_TRIALS}"
            summary["n_trials"] = n_lines

            # ---------------- steered-damage KL ----------------
            torch.cuda.empty_cache()
            t_kl = time.perf_counter()
            n_chunks = int(cfg["kl_n_chunks"])
            n_kl_concepts = int(cfg["kl_n_concepts"])
            chunks = [
                json.loads(l)
                for l in Path(wo002_wikitext_path).read_text().splitlines()
                if l.strip()
            ][:n_chunks]
            assert len(chunks) == n_chunks, f"only {len(chunks)} wikitext chunks"
            kl_concepts = concepts_50[:n_kl_concepts]
            chunk_tokens = int(cfg.get("wikitext_chunk_tokens", 64))
            bos = tokenizer.bos_token_id

            chunk_ids: List[torch.Tensor] = []
            for ch in chunks:
                ids = tokenizer(ch["text"], add_special_tokens=False, return_tensors="pt")[
                    "input_ids"
                ]
                content = ids[0, :chunk_tokens]
                row = torch.cat([torch.tensor([bos]), content], dim=0)
                chunk_ids.append(row.unsqueeze(0))

            pairs = [(ci, c) for ci in range(len(chunks)) for c in kl_concepts]
            kl_bs = max(1, min(batch_size, 4))  # logits [2B,T,V≈262k] twice — keep small
            kl_rows: List[dict] = []
            kl_means: Dict[str, float] = {}
            for arm in ARMS:
                bdirs, bl0 = arm_cfg[arm]
                vals_arm: List[float] = []
                for s in range(0, len(pairs), kl_bs):
                    chunk_pairs = pairs[s : s + kl_bs]
                    B = len(chunk_pairs)
                    input_ids, attn, pad_lens = pad_batch(
                        tokenizer, [chunk_ids[ci] for ci, _ in chunk_pairs], device
                    )
                    st = torch.tensor(
                        [pl + 1 for pl in pad_lens], device=device, dtype=torch.long
                    )
                    sv = torch.stack([alpha * vec50[c] for _, c in chunk_pairs]).to(
                        device=device, dtype=torch.bfloat16
                    )
                    logits = paired_forward_logits(
                        model, input_ids, attn, st, sv, layer_steer, bdirs, bl0,
                        max_layer=LAYER_LAST,
                    )
                    for b in range(B):
                        pl = pad_lens[b]
                        log_p = torch.log_softmax(logits[B + b, pl + 1 :].float(), dim=-1)  # twin
                        log_q = torch.log_softmax(logits[b, pl + 1 :].float(), dim=-1)  # steered
                        kl_tok = (log_p.exp() * (log_p - log_q)).sum(dim=-1)  # [T']
                        v = float(kl_tok.mean().item())
                        vals_arm.append(v)
                        kl_rows.append(
                            {
                                "arm": arm,
                                "chunk_id": chunks[chunk_pairs[b][0]]["id"],
                                "chunk_index": chunk_pairs[b][0],
                                "concept": chunk_pairs[b][1],
                                "n_positions": int(kl_tok.numel()),
                                "kl": v,
                            }
                        )
                        del log_p, log_q, kl_tok
                    del logits
                kl_means[arm] = float(sum(vals_arm) / len(vals_arm)) if vals_arm else float("nan")
                _log(f"KL steered {arm}: {kl_means[arm]:.6f} (n={len(vals_arm)})")
                torch.cuda.empty_cache()

            with (raw / "kl_raw.jsonl").open("w") as f:
                for r in kl_rows:
                    f.write(json.dumps(r) + "\n")

            kl_none = kl_means["none"]
            kl_ratio = kl_means["B1p"] / kl_none if kl_none > 0 else float("nan")
            kl_ratio_null = kl_means["B1p-null"] / kl_none if kl_none > 0 else float("nan")
            _log(f"kl_ratio (B1p/none) = {kl_ratio:.6f}; null/none = {kl_ratio_null:.6f}")
            _log(f"KL done in {time.perf_counter() - t_kl:.1f}s")

            summary["kl_means"] = kl_means
            summary["kl_ratio"] = None if math.isnan(kl_ratio) else kl_ratio
            summary["kl_ratio_null"] = None if math.isnan(kl_ratio_null) else kl_ratio_null
            summary["kl_config"] = {
                "n_chunks": n_chunks,
                "n_concepts": n_kl_concepts,
                "concepts": kl_concepts,
                "chunk_ids": [c["id"] for c in chunks],
                "chunk_tokens": chunk_tokens,
            }
            summary["elapsed_sec"] = time.perf_counter() - t_run0
            (results_path / "partB_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            return summary

        except Exception as e:  # noqa: BLE001
            _log(f"ERROR: {e}")
            _log(traceback.format_exc())
            summary["errors"].append(str(e))
            (results_path / "partB_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            raise


if __name__ == "__main__":
    run_part_b(
        model_dir=os.environ["SFO_MODEL_DIR"],
        revision_sha=os.environ["SFO_REVISION_SHA"],
        config_path=os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_002b.yaml")),
        results_dir=os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-002b")),
        ledger_path=os.environ.get("SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")),
        wo001_vectors_path=str(REPO_ROOT / "results/WO-001/raw/concept_vectors_L37.pt"),
        wo001_snapshot_path=str(REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json"),
        wo002_wikitext_path=os.environ.get(
            "SFO_WO002_WIKITEXT",
            str(REPO_ROOT / "results/WO-002/raw/wikitext_chunks.jsonl"),
        ),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )
