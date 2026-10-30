"""WO-002b Part A: B1′ anomaly directions (non-question prompt), l0 sweep, select l0*."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.hooks import _resolve_layers, capture_residual  # noqa: E402,F401
from sfo.ld import detection_logit, yes_no_token_ids  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import format_chat, get_d_model, load_model, load_tokenizer  # noqa: E402
from sfo.paired import (  # noqa: E402
    add_steering_paired,
    b1prime_from_l0,
    duplicate_batch,
    paired_capture_shifts,
    paired_forward_logits,
    paired_generate,
    steer_vectors_paired,
    steered_twin_starts,
    unit,
)
from sfo.prompts import build_introspection_prompt, introspection_messages  # noqa: E402,F401
from sfo.provenance import collect_env  # noqa: E402
from sfo.vectors import compute_start_pos, format_chat_text, pad_batch  # noqa: E402,F401

LAYER_LAST = 61
RAND_SEED_BASE = 100
N_RAND = 50
RET_THRESHOLD_DEFAULT = 0.80


def _log(msg: str) -> None:
    print(msg, flush=True)


def _nan_to_none(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return x


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def _mean(x: Sequence[float]) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64)))


def recovery(dld_none: Sequence[float], dld_blk: Sequence[float]) -> float:
    """rec = 1 − mean ΔLD_blk / mean ΔLD_none."""
    mn = _mean(dld_none)
    if abs(mn) < 1e-8:
        return float("nan")
    return 1.0 - _mean(dld_blk) / mn


def retention(pc_none: Sequence[float], pc_blk: Sequence[float]) -> float:
    """retention = mean p_c(blk) / mean p_c(none)."""
    mn = _mean(pc_none)
    if abs(mn) < 1e-30:
        return float("nan")
    return _mean(pc_blk) / mn


def concept_first_token_ids(tokenizer: Any, w: str) -> List[int]:
    """First token ids of ' '+w and ' '+w.lower(), deduplicated (order-preserving)."""
    out: List[int] = []
    for s in (" " + w, " " + w.lower()):
        t = tokenizer.encode(s, add_special_tokens=False)
        if len(t) == 0:
            continue
        if int(t[0]) not in out:
            out.append(int(t[0]))
    return out


@torch.no_grad()
def run_pairs(
    model: Any,
    ids_1: torch.Tensor,
    start_pos: int,
    vecs: Sequence[torch.Tensor],
    blocker_dirs: Optional[Dict[int, torch.Tensor]],
    l0: Optional[int],
    bs: int,
    steer_layer: int,
    yes_id: int,
    no_id: int,
    tok_ids: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Paired forward over N concepts on a single fixed prompt (no padding).
    Returns ld_s, ld_t [N] (LD at last position) and, if tok_ids given, pc [N]
    (summed softmax prob over tok_ids at last position of the steered row).
    B1′ (if blocker_dirs given) acts on steered rows only; twin unblocked.
    """
    device = next(model.parameters()).device
    ids_1 = ids_1.to(device)
    if ids_1.ndim == 1:
        ids_1 = ids_1.unsqueeze(0)
    N = len(vecs)
    ld_s = torch.zeros(N)
    ld_t = torch.zeros(N)
    pc = torch.zeros(N)
    for s in range(0, N, bs):
        chunk = list(vecs[s : s + bs])
        B = len(chunk)
        ids = ids_1.expand(B, -1).contiguous()
        attn = torch.ones_like(ids)
        starts = torch.full((B,), int(start_pos), dtype=torch.long, device=device)
        sv = torch.stack([v.float() for v in chunk]).to(device=device, dtype=torch.bfloat16)
        logits = paired_forward_logits(
            model, ids, attn, starts, sv, steer_layer, blocker_dirs, l0
        )
        last = logits[:, -1, :]
        ld = detection_logit(last, yes_id, no_id).float().cpu()
        ld_s[s : s + B] = ld[:B]
        ld_t[s : s + B] = ld[B:]
        if tok_ids is not None:
            probs = torch.softmax(last[:B].float(), dim=-1)
            for i in range(B):
                pc[s + i] = float(probs[i, list(tok_ids[s + i])].sum().item())
        del logits, last
    out = {"ld_s": ld_s, "ld_t": ld_t}
    if tok_ids is not None:
        out["pc"] = pc
    return out


def run_part_a(
    model_dir,
    revision_sha,
    config_path,
    results_dir,
    ledger_path,
    wo001_vectors_path,
    wo001_snapshot_path,
    wo001_per_concept_csv,
    wo002_vectors_path,
    wo002_directions_path,
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
        function="wo_002b_partA",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            # ---------------- setup ----------------
            dir_dev = list(cfg["dir_dev"])
            dir_val = list(cfg["dir_val"])
            assert len(dir_dev) == 225 and len(dir_val) == 225
            layer_steer = int(cfg["layer_steer"])
            alpha = float(cfg["alpha"])
            prefill = str(cfg["fid_prefill"])
            bs = int(cfg.get("pair_batch_size", 16))
            sweep_n = int(cfg.get("sweep_n_concepts", 60))
            null_seed = int(cfg.get("b1_null_seed", 0))
            user_free = str(cfg.get("exp_user", "Write two sentences about whatever comes to mind."))
            assert user_free == "Write two sentences about whatever comes to mind."
            l0_list = [int(x) for x in cfg.get("l0_sweep", [38, 39, 40, 41])]
            layers = list(range(38, LAYER_LAST + 1))
            RET_THRESHOLD = float(cfg.get("retention_threshold", RET_THRESHOLD_DEFAULT))
            N_GEN_TOKENS = int(cfg.get("a1_cont_tokens", 32))
            _log(
                f"Config: alpha={alpha} layer_steer={layer_steer} pair_batch_size={bs} "
                f"sweep_n={sweep_n} l0_list={l0_list} ret_thr={RET_THRESHOLD}"
            )

            wo001 = json.loads(Path(wo001_snapshot_path).read_text())
            concepts_50 = list(wo001["concepts"])

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            model = load_model(model_dir, revision=revision_sha, device=0)
            device = next(model.parameters()).device
            d_model = get_d_model(model)
            yes_id, no_id = yes_no_token_ids(tokenizer)
            summary["env"] = collect_env()
            summary["config"] = {
                "alpha": alpha,
                "layer_steer": layer_steer,
                "pair_batch_size": bs,
                "sweep_n_concepts": sweep_n,
                "l0_sweep": l0_list,
                "fid_prefill": prefill,
            }

            # Sanity 2 (GPU): (i) twin matches unsteered twin in same batch size
            # (CUDA GEMM can differ B=1 vs B=2, so compare within paired batch);
            # also twin ≈ unpaired within loose rtol. (iv) greedy paired == HF.
            _log("SANITY2 GPU: twin integrity + greedy paired_generate==HF (2 prompts)...")
            for pi, trial_n in enumerate((1, 2)):
                _text_i, ids_i, sp_i = build_introspection_prompt(tokenizer, trial_n)
                ids_i = ids_i.to(device)
                attn_i = torch.ones_like(ids_i)
                starts_i = torch.tensor([sp_i], device=device, dtype=torch.long)
                g = torch.Generator().manual_seed(7 + pi)
                sv = (
                    torch.randn(1, d_model, generator=g).to(device=device, dtype=torch.bfloat16)
                    * 0.01
                )
                zeros = torch.zeros(1, d_model, device=device, dtype=torch.bfloat16)
                with torch.no_grad():
                    base = model(input_ids=ids_i, attention_mask=attn_i).logits
                    logits_steer = paired_forward_logits(
                        model, ids_i, attn_i, starts_i, sv, layer_steer, None, None
                    )
                    logits_zero = paired_forward_logits(
                        model, ids_i, attn_i, starts_i, zeros, layer_steer, None, None
                    )
                twin_s = logits_steer[1:]
                twin_z = logits_zero[1:]
                # Steering must not leak into twin (same paired layout)
                assert torch.equal(twin_s, twin_z), (
                    f"SANITY2 steering leaked into twin prompt {pi}"
                )
                # Twin ≈ unpaired (bf16/CUDA: allow relative slack)
                rel = (twin_s.float() - base.float()).abs() / (base.float().abs() + 1.0)
                assert float(rel.max()) < 5e-2, (
                    f"SANITY2 twin≈unpaired failed prompt {pi}: max_rel={float(rel.max())}"
                )
                with torch.no_grad():
                    hf = model.generate(
                        input_ids=ids_i,
                        attention_mask=attn_i,
                        max_new_tokens=10,
                        do_sample=False,
                        use_cache=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    steered, twin = paired_generate(
                        model,
                        ids_i,
                        attn_i,
                        starts_i,
                        zeros,
                        layer_steer,
                        None,
                        None,
                        max_new_tokens=10,
                        do_sample=False,
                        eos_ids=(999999,),
                    )
                assert torch.equal(steered[0], hf[0].cpu()), f"SANITY2 greedy mismatch prompt {pi}"
                assert torch.equal(steered[0], twin[0])
            _log("SANITY2 GPU twin/greedy checks: PASS")
            summary["sanity"]["gpu_twin_and_greedy"] = True

            # vectors
            pack450 = torch.load(wo002_vectors_path, map_location="cpu", weights_only=False)
            name_to_vec = {c: pack450["vectors"][i].float() for i, c in enumerate(pack450["concepts"])}
            for c in dir_dev + dir_val:
                assert c in name_to_vec, f"missing vector for {c}"
            pack50 = torch.load(wo001_vectors_path, map_location="cpu", weights_only=False)
            vec50 = {c: pack50["vectors"][i].float() for i, c in enumerate(pack50["concepts"])}
            for c in concepts_50:
                assert c in vec50, f"missing WO-001 vector for {c}"
            assert len(concepts_50) == 50

            # ---------------- A1 fixed tokens ----------------
            _log("A1: generating fixed unsteered continuation...")
            p_ids = format_chat(tokenizer, [{"role": "user", "content": user_free}])
            p_ids_d = p_ids.to(device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=p_ids_d,
                    attention_mask=torch.ones_like(p_ids_d),
                    max_new_tokens=N_GEN_TOKENS,
                    min_new_tokens=N_GEN_TOKENS,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    use_cache=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
            L_p = int(p_ids.shape[1])
            full_ids = gen[0].detach().cpu()
            assert full_ids.numel() == L_p + N_GEN_TOKENS, (
                f"expected {L_p + N_GEN_TOKENS} tokens, got {full_ids.numel()}"
            )
            assert torch.equal(full_ids[:L_p], p_ids[0].cpu())
            cont_ids = full_ids[L_p:]
            decoded_full = tokenizer.decode(full_ids.tolist(), skip_special_tokens=False)
            (raw / "S_tokens.json").write_text(
                json.dumps(
                    {
                        "prompt_ids": p_ids[0].tolist(),
                        "continuation_ids": cont_ids.tolist(),
                        "full_ids": full_ids.tolist(),
                        "decoded": decoded_full,
                    },
                    indent=2,
                )
            )
            T_S = int(full_ids.numel())
            S_ids = full_ids.unsqueeze(0).to(device)  # [1,T]
            summary["S"] = {"prompt_len": L_p, "total_len": T_S}

            # Sanity 6: print decoded S
            _log("SANITY6 decoded S:")
            _log(decoded_full)
            _log(f"SANITY6 continuation: {tokenizer.decode(cont_ids.tolist(), skip_special_tokens=False)!r}")

            # ---------------- A2 shifts ----------------
            _log("A2: capturing per-concept mean shifts on S (dir-dev + random)...")
            t_a2 = time.perf_counter()

            def capture_shifts(vec_list: Sequence[torch.Tensor]) -> torch.Tensor:
                """vec_list: steering vectors (already scaled). Returns [N, n_layers, D] float32 CPU."""
                N = len(vec_list)
                out = torch.zeros(N, len(layers), d_model, dtype=torch.float32)
                for s in range(0, N, bs):
                    chunk = list(vec_list[s : s + bs])
                    B = len(chunk)
                    ids = S_ids.expand(B, -1).contiguous()
                    attn = torch.ones_like(ids)
                    starts = torch.ones(B, dtype=torch.long, device=device)  # positions >= 1
                    sv = torch.stack([v.float() for v in chunk]).to(
                        device=device, dtype=torch.bfloat16
                    )
                    caps = paired_capture_shifts(
                        model, ids, attn, starts, sv, layer_steer, layers
                    )
                    for li, l in enumerate(layers):
                        out[s : s + B, li, :] = caps[l].float().cpu()
                return out

            dev_vecs = [alpha * name_to_vec[c] for c in dir_dev]
            shifts_dev = capture_shifts(dev_vecs)
            _log(f"  dir-dev shifts done ({time.perf_counter()-t_a2:.1f}s)")

            rand_vecs = []
            for i in range(N_RAND):
                g = torch.Generator().manual_seed(RAND_SEED_BASE + i)
                rv = unit(torch.randn(d_model, generator=g))
                rand_vecs.append(rv * (2.0 * float(vec50[concepts_50[i]].norm())))
            shifts_rand = capture_shifts(rand_vecs)
            _log(f"A2 done in {time.perf_counter()-t_a2:.1f}s")

            torch.save(
                {
                    "dir_dev": dir_dev,
                    "layers": layers,
                    "shifts_dirdev": shifts_dev.to(torch.float16),
                    "shifts_rand": shifts_rand.to(torch.float16),
                    "rand_seeds": list(range(RAND_SEED_BASE, RAND_SEED_BASE + N_RAND)),
                },
                raw / "partA_shifts.pt",
            )

            # ---------------- A3 directions ----------------
            _log("A3: directions...")
            # twin residual norms (one unsteered forward of S)
            with capture_residual(model, layers) as caps:
                with torch.no_grad():
                    model(input_ids=S_ids, attention_mask=torch.ones_like(S_ids))
            med_norm = {
                l: float(caps[l][0][1:].norm(dim=-1).median().item()) for l in layers
            }
            del caps

            wo002_dirs = torch.load(wo002_directions_path, map_location="cpu", weights_only=False)
            wo002_dhat = wo002_dirs["d_hat"]

            v_mean_hat = unit(
                torch.stack([unit(name_to_vec[c]) for c in dir_dev]).mean(0)
            )
            unit50 = torch.stack([unit(vec50[c]) for c in concepts_50])  # [50,D]

            d_hat: Dict[int, torch.Tensor] = {}
            d_hat_rand: Dict[int, torch.Tensor] = {}
            per_layer: Dict[str, Any] = {}
            for li, l in enumerate(layers):
                d = shifts_dev[:, li, :].mean(0)
                dr = shifts_rand[:, li, :].mean(0)
                d_hat[l] = unit(d)
                d_hat_rand[l] = unit(dr)
            for li, l in enumerate(layers):
                d = shifts_dev[:, li, :].mean(0)
                cos50 = (unit50 @ d_hat[l]).abs()
                per_layer[str(l)] = {
                    "d_norm": float(d.norm().item()),
                    "d_norm_ratio": float(d.norm().item()) / (med_norm[l] + 1e-12),
                    "median_twin_resid_norm": med_norm[l],
                    "cos_rand": _cos(d_hat[l], d_hat_rand[l]),
                    "cos_mean_v": _cos(d_hat[l], v_mean_hat),
                    "max_abs_cos_val50": float(cos50.max().item()),
                    "p95_abs_cos_val50": float(torch.quantile(cos50, 0.95).item()),
                    "cos_wo002": _cos(d_hat[l], wo002_dhat[l]) if l in wo002_dhat else None,
                    "cos_next": _cos(d_hat[l], d_hat[l + 1]) if (l + 1) in d_hat else None,
                }
            torch.save(
                {
                    "d_hat": {l: d_hat[l].clone() for l in layers},
                    "d_hat_rand": {l: d_hat_rand[l].clone() for l in layers},
                    "layers": layers,
                },
                raw / "directions_B1p.pt",
            )
            for l in layers:
                p = per_layer[str(l)]
                _log(
                    f"  L{l}: d_norm={p['d_norm']:.2f} ratio={p['d_norm_ratio']:.4f} "
                    f"cos_rand={p['cos_rand']:.4f} cos_mean_v={p['cos_mean_v']:.4f} "
                    f"max|cos|val50={p['max_abs_cos_val50']:.4f} p95={p['p95_abs_cos_val50']:.4f} "
                    f"cos_wo002={p['cos_wo002']} cos_next={p['cos_next']}"
                )

            # null directions: random unit per layer, seed 0
            g0 = torch.Generator().manual_seed(null_seed)
            null_dirs = {l: unit(torch.randn(d_model, generator=g0)) for l in layers}

            # ---------------- Sanity (before main sweep) ----------------
            ids_i1 = build_introspection_prompt(tokenizer, 1)
            _, intro_ids, intro_sp = ids_i1
            intro_ids_d = intro_ids.to(device)

            # Sanity 3: twin integrity via paired_generate
            nb = 4
            ids_g = intro_ids_d.expand(nb, -1).contiguous()
            attn_g = torch.ones_like(ids_g)
            starts_g = torch.full((nb,), int(intro_sp), dtype=torch.long, device=device)
            sv_g = torch.stack([alpha * name_to_vec[c] for c in dir_dev[:nb]]).to(
                device=device, dtype=torch.bfloat16
            )
            s_seq, t_seq = paired_generate(
                model, ids_g, attn_g, starts_g, sv_g, layer_steer, d_hat, 40,
                max_new_tokens=8, do_sample=False,
            )
            for i in range(nb):
                assert torch.equal(s_seq[i], t_seq[i]), f"twin integrity failed pair {i}"
            _log(f"SANITY3 twin integrity (4 pairs, paired_generate, B1′ l0=40): PASS")
            summary["sanity"]["twin_integrity"] = True

            # Sanity 4: α=0 and zero-direction identities (2 prompts)
            ident = {"alpha0": [], "zero_dir": []}
            zero_dirs = {l: torch.zeros(d_model) for l in layers}
            for tn in (1, 2):
                _, ids_t, sp_t = build_introspection_prompt(tokenizer, tn)
                ids_t = ids_t.to(device)
                nbi = 2
                ids_b = ids_t.expand(nbi, -1).contiguous()
                attn_b = torch.ones_like(ids_b)
                st_b = torch.full((nbi,), int(sp_t), dtype=torch.long, device=device)
                zero_v = torch.zeros(nbi, d_model, device=device, dtype=torch.bfloat16)
                a = paired_forward_logits(model, ids_b, attn_b, st_b, zero_v, layer_steer, None, None)
                b = paired_forward_logits(model, ids_b, attn_b, st_b, zero_v, layer_steer, d_hat, 40)
                ident["alpha0"].append(bool(torch.equal(a, b)))
                del a, b
                real_v = torch.stack([alpha * name_to_vec[c] for c in dir_dev[:nbi]]).to(
                    device=device, dtype=torch.bfloat16
                )
                a = paired_forward_logits(model, ids_b, attn_b, st_b, real_v, layer_steer, None, None)
                b = paired_forward_logits(model, ids_b, attn_b, st_b, real_v, layer_steer, zero_dirs, 40)
                ident["zero_dir"].append(bool(torch.equal(a, b)))
                del a, b
            assert all(ident["alpha0"]), f"α=0 B1′ identity failed: {ident}"
            assert all(ident["zero_dir"]), f"zero-direction identity failed: {ident}"
            _log(f"SANITY4 α=0 and zero-direction identity (2 prompts, bit-identical logits): PASS {ident}")
            summary["sanity"]["identity"] = ident

            # Sanity 5: removal correctness at a single layer
            l_chk = 40
            nbc = 2
            ids_c = intro_ids_d.expand(nbc, -1).contiguous()
            attn_c = torch.ones_like(ids_c)
            st_c = torch.full((nbc,), int(intro_sp), dtype=torch.long, device=device)
            sv_c = torch.stack([alpha * name_to_vec[c] for c in dir_dev[:nbc]]).to(
                device=device, dtype=torch.bfloat16
            )
            ids2, attn2 = duplicate_batch(ids_c, attn_c)
            st2 = steered_twin_starts(st_c)
            vec2 = steer_vectors_paired(sv_c)
            dir_chk = {l_chk: d_hat[l_chk]}
            # unblocked reference
            with add_steering_paired(model, layer_steer, vec2, st2):
                with capture_residual(model, [l_chk]) as caps_ref:
                    with torch.no_grad():
                        model(input_ids=ids2, attention_mask=attn2)
            h_ref = caps_ref[l_chk]
            # blocked (capture hook registered after the blocker hook → sees post-B1′ output)
            with add_steering_paired(model, layer_steer, vec2, st2):
                with b1prime_from_l0(model, l_chk, dir_chk, st2, nbc, max_layer=l_chk):
                    with capture_residual(model, [l_chk]) as caps_blk:
                        with torch.no_grad():
                            model(input_ids=ids2, attention_mask=attn2)
            h_blk = caps_blk[l_chk]
            sp_i = int(intro_sp)
            dvec = d_hat[l_chk].float()
            max_rel = 0.0
            for i in range(nbc):
                hs = h_blk[i, sp_i:, :]
                ht = h_blk[nbc + i, sp_i:, :]
                proj = ((hs - ht) * dvec.unsqueeze(0)).sum(-1).abs()
                rel = (proj / hs.norm(dim=-1).clamp_min(1e-12)).max().item()
                max_rel = max(max_rel, float(rel))
            assert max_rel < 1e-3, f"removal residual too large: max_rel={max_rel}"
            # pos < start unchanged (blocked vs unblocked, same rows) — must be bit-identical
            pre_same = bool(torch.equal(h_blk[:nbc, :sp_i], h_ref[:nbc, :sp_i]))
            pre_same_twin_rows = bool(torch.equal(h_blk[nbc:, :sp_i], h_ref[nbc:, :sp_i]))
            assert pre_same and pre_same_twin_rows, "pos<start changed by B1′"
            # informational: steered vs twin pre-start bit-identity
            pre_s_vs_t = bool(torch.equal(h_blk[:nbc, :sp_i], h_blk[nbc:, :sp_i]))
            # sanity: unblocked had a nonzero projection to remove
            pre_proj = 0.0
            for i in range(nbc):
                pre_proj = max(
                    pre_proj,
                    float(
                        ((h_ref[i, sp_i:, :] - h_ref[nbc + i, sp_i:, :]) * dvec).sum(-1).abs().max()
                    ),
                )
            _log(
                f"SANITY5 removal correctness (layer {l_chk}): PASS max|(h_i−h_twin)·d|/‖h_i‖="
                f"{max_rel:.2e} (<1e-3); pos<start blocked==unblocked bit-identical={pre_same}; "
                f"steered==twin pre-start={pre_s_vs_t}; unblocked max|proj|={pre_proj:.2f}"
            )
            summary["sanity"]["removal"] = {
                "layer": l_chk,
                "max_rel": max_rel,
                "pre_start_bit_identical_vs_unblocked": pre_same and pre_same_twin_rows,
                "pre_start_steered_vs_twin_bit_identical": pre_s_vs_t,
                "unblocked_max_abs_proj": pre_proj,
            }
            del h_ref, h_blk, caps_ref, caps_blk
            torch.cuda.empty_cache()

            # ---------------- A4 sweep ----------------
            _log("A4: sweep...")
            t_a4 = time.perf_counter()
            det_ids = intro_ids  # introspection n=1 [1,L]
            pre_ids = tokenizer(prefill, add_special_tokens=False, return_tensors="pt")["input_ids"]
            fid_ids = torch.cat([intro_ids.cpu(), pre_ids], dim=1)
            assert int(fid_ids[0, 0]) == int(tokenizer.bos_token_id)
            assert int((fid_ids[0] == tokenizer.bos_token_id).sum()) == 1

            def eval_set(concepts: Sequence[str], l0: Optional[int], dirs: Optional[Dict[int, torch.Tensor]]):
                vecs = [alpha * name_to_vec[c] for c in concepts]
                tok_ids = [concept_first_token_ids(tokenizer, c) for c in concepts]
                det = run_pairs(
                    model, det_ids, intro_sp, vecs, dirs, l0, bs, layer_steer, yes_id, no_id
                )
                fid = run_pairs(
                    model, fid_ids, intro_sp, vecs, dirs, l0, bs, layer_steer, yes_id, no_id,
                    tok_ids=tok_ids,
                )
                return {
                    "dld": (det["ld_s"] - det["ld_t"]).tolist(),
                    "pc": fid["pc"].tolist(),
                }

            def metrics(none: Dict[str, List[float]], blk: Dict[str, List[float]]) -> Dict[str, Any]:
                return {
                    "rec": recovery(none["dld"], blk["dld"]),
                    "ret": retention(none["pc"], blk["pc"]),
                    "mean_dld_none": _mean(none["dld"]),
                    "mean_dld_blk": _mean(blk["dld"]),
                    "mean_pc_none": _mean(none["pc"]),
                    "mean_pc_blk": _mean(blk["pc"]),
                    "n": len(none["dld"]),
                }

            sweep_concepts = dir_dev[:sweep_n]
            none_sweep = eval_set(sweep_concepts, None, None)
            _log(
                f"  none: mean dLD={_mean(none_sweep['dld']):.4f} mean p_c={_mean(none_sweep['pc']):.6g}"
            )
            sweep: Dict[str, Any] = {}
            jsonl_rows: List[Dict[str, Any]] = []
            for l0 in l0_list:
                entry = {}
                for name, dirs in (("B1p", d_hat), ("B1p_null", null_dirs)):
                    blk = eval_set(sweep_concepts, l0, dirs)
                    m = metrics(none_sweep, blk)
                    entry[name] = m
                    jsonl_rows.append(
                        {
                            "stage": "sweep",
                            "l0": l0,
                            "variant": name,
                            **m,
                            "concepts": sweep_concepts,
                            "dld_none": none_sweep["dld"],
                            "dld_blk": blk["dld"],
                            "pc_none": none_sweep["pc"],
                            "pc_blk": blk["pc"],
                        }
                    )
                    _log(
                        f"  l0={l0} {name}: rec={m['rec']:.4f} ret={m['ret']:.4f} "
                        f"(mean dLD none={m['mean_dld_none']:.3f} blk={m['mean_dld_blk']:.3f}; "
                        f"mean p_c none={m['mean_pc_none']:.4g} blk={m['mean_pc_blk']:.4g})"
                    )
                sweep[str(l0)] = entry

            # selection
            elig = [
                (sweep[str(l0)]["B1p"]["rec"], l0)
                for l0 in l0_list
                if not math.isnan(sweep[str(l0)]["B1p"]["ret"])
                and sweep[str(l0)]["B1p"]["ret"] >= RET_THRESHOLD
                and not math.isnan(sweep[str(l0)]["B1p"]["rec"])
            ]
            l0_flag = False
            if elig:
                l0_star = max(elig, key=lambda x: x[0])[1]
            else:
                l0_flag = True
                rets = [
                    (sweep[str(l0)]["B1p"]["ret"], l0)
                    for l0 in l0_list
                    if not math.isnan(sweep[str(l0)]["B1p"]["ret"])
                ]
                l0_star = max(rets, key=lambda x: x[0])[1]
            _log(f"Selected l0*={l0_star} l0_flag={l0_flag}")
            _log(f"A4 done in {time.perf_counter()-t_a4:.1f}s")

            # Flatten sweep to WO metrics schema
            sweep_flat: Dict[str, Any] = {}
            for l0 in l0_list:
                e = sweep[str(l0)]
                sweep_flat[str(l0)] = {
                    "rec_B1p": e["B1p"]["rec"],
                    "ret_B1p": e["B1p"]["ret"],
                    "rec_null": e["B1p_null"]["rec"],
                    "ret_null": e["B1p_null"]["ret"],
                    "B1p": e["B1p"],
                    "B1p_null": e["B1p_null"],
                }
            sweep = sweep_flat

            # ---------------- A5 held-out dir-val ----------------
            _log("A5: held-out dir-val at l0*...")
            none_val = eval_set(dir_val, None, None)
            heldout_nested: Dict[str, Any] = {}
            for name, dirs in (("B1p", d_hat), ("B1p_null", null_dirs)):
                blk = eval_set(dir_val, l0_star, dirs)
                m = metrics(none_val, blk)
                heldout_nested[name] = m
                jsonl_rows.append(
                    {
                        "stage": "heldout",
                        "l0": l0_star,
                        "variant": name,
                        **m,
                        "concepts": dir_val,
                        "dld_none": none_val["dld"],
                        "dld_blk": blk["dld"],
                        "pc_none": none_val["pc"],
                        "pc_blk": blk["pc"],
                    }
                )
                _log(f"  heldout l0*={l0_star} {name}: rec={m['rec']:.4f} ret={m['ret']:.4f}")
            heldout = {
                "rec_B1p": heldout_nested["B1p"]["rec"],
                "ret_B1p": heldout_nested["B1p"]["ret"],
                "rec_null": heldout_nested["B1p_null"]["rec"],
                "ret_null": heldout_nested["B1p_null"]["ret"],
                "l0": int(l0_star),
                "B1p": heldout_nested["B1p"],
                "B1p_null": heldout_nested["B1p_null"],
            }

            # ---------------- A6 proxy ----------------
            _log("A6: proxy Spearman...")
            from scipy.stats import spearmanr

            with open(wo001_per_concept_csv) as f:
                reader = csv.DictReader(f)
                cols = list(reader.fieldnames or [])
                rows = list(reader)
            ident_col = None
            for cand in ("ident_rate", "ident", "identification_rate", "introspection@4"):
                if cand in cols:
                    ident_col = cand
                    break
            assert ident_col is not None, f"no ident column in {cols}"
            ident_map = {r["concept"]: float(r[ident_col]) for r in rows}
            vecs50 = [alpha * vec50[c] for c in concepts_50]
            tok50 = [concept_first_token_ids(tokenizer, c) for c in concepts_50]
            fid50 = run_pairs(
                model, fid_ids, intro_sp, vecs50, None, None, bs, layer_steer, yes_id, no_id,
                tok_ids=tok50,
            )
            pc50 = fid50["pc"].tolist()
            idr = [ident_map[c] for c in concepts_50]
            rho, pval = spearmanr(pc50, idr)
            proxy = {
                "rho": _nan_to_none(float(rho)),
                "p": _nan_to_none(float(pval)),
                "n": len(concepts_50),
                "ident_column": ident_col,
            }
            _log(f"A6 Spearman rho={rho:.4f} p={pval:.4g} (ident column {ident_col!r})")

            # ---------------- write outputs ----------------
            with open(results_path / "partA_sweep.jsonl", "w") as f:
                for r in jsonl_rows:
                    f.write(json.dumps(r, default=str) + "\n")

            partA = {
                "per_layer": per_layer,
                "sweep": sweep,
                "l0_star": int(l0_star),
                "l0_flag": bool(l0_flag),
                "heldout": heldout,
                "proxy_spearman": proxy,
            }
            summary["partA"] = partA

            # Sanity 7: projected Part B time (1500 paired gens + KL 50×10×3)
            elapsed_h = (time.perf_counter() - t_run0) / 3600.0
            n_gen = int(cfg.get("partB_n_gen", 1500))
            sec_per_gen = float(cfg.get("partB_sec_per_gen", 1.2))
            n_kl = int(cfg.get("kl_n_chunks", 50)) * int(cfg.get("kl_n_concepts", 10)) * 3
            sec_per_kl = float(cfg.get("partB_sec_per_kl", 0.15))
            projected_b = (n_gen * sec_per_gen + n_kl * sec_per_kl) / 3600.0
            projected_total = elapsed_h + projected_b
            limit = float(cfg.get("projected_gpu_limit_hours", 1.3))
            _log(
                f"SANITY7: Part A elapsed {elapsed_h*60:.1f} min; projected Part B {projected_b*60:.1f} min; "
                f"projected total {projected_total:.3f} h (limit {limit} h)"
            )
            summary["projected_hours"] = projected_total
            summary["projected_hours_detail"] = {
                "part_a_elapsed": elapsed_h,
                "part_b_projected": projected_b,
                "total": projected_total,
                "limit": limit,
            }
            if projected_total > limit:
                summary["stopped_before_partB"] = True
                summary["stop_reason"] = f"Projected {projected_total:.3f} h > {limit} h"
                _log(summary["stop_reason"])
            else:
                summary["stopped_before_partB"] = False

            (results_path / "partA_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            return summary

        except Exception as e:  # noqa: BLE001
            _log(f"ERROR: {e}")
            _log(traceback.format_exc())
            summary["errors"].append(str(e))
            (results_path / "partA_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            raise


if __name__ == "__main__":
    _cfg_default = REPO_ROOT / "configs/wo_002b.yaml"
    if not _cfg_default.exists():
        _cfg_default = REPO_ROOT / "configs/wo_002.yaml"
    run_part_a(
        model_dir=os.environ["SFO_MODEL_DIR"],
        revision_sha=os.environ["SFO_REVISION_SHA"],
        config_path=os.environ.get("SFO_CONFIG", str(_cfg_default)),
        results_dir=os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-002b")),
        ledger_path=os.environ.get("SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")),
        wo001_vectors_path=str(REPO_ROOT / "results/WO-001/raw/concept_vectors_L37.pt"),
        wo001_snapshot_path=str(REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json"),
        wo001_per_concept_csv=str(REPO_ROOT / "results/WO-001/raw/per_concept.csv"),
        wo002_vectors_path=str(REPO_ROOT / "results/WO-002/raw/concept_vectors_L37_new450.pt"),
        wo002_directions_path=str(REPO_ROOT / "results/WO-002/raw/directions.pt"),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )
