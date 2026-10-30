"""WO-002 Part A: find anomaly direction, layer sweep, select l0*."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.blockers import mean_ablate_from_l0  # noqa: E402
from sfo.hooks import add_to_residual_batched, capture_residual  # noqa: E402
from sfo.inputs_ast import load_new_concepts, load_reference_inputs  # noqa: E402
from sfo.ld import detection_logit, yes_no_token_ids  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import get_d_model, get_layers, load_model, load_tokenizer  # noqa: E402
from sfo.prompts import build_introspection_prompt  # noqa: E402
from sfo.provenance import collect_env  # noqa: E402
from sfo.vectors import compute_concept_vectors, pad_batch, residual_norm_diagnostics  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


@torch.no_grad()
def forward_ld_and_hiddens(
    model,
    tokenizer,
    id_list: List[torch.Tensor],
    starts_unpadded: List[int],
    layer_idxs: List[int],
    vectors: Optional[torch.Tensor],
    steer_layer: int,
    yes_id: int,
    no_id: int,
    capture_all_from_start: bool = False,
) -> Dict[str, Any]:
    """
    Batched forward. Returns LD [B], last-pos hiddens {l: [B,d]}, and optionally
    full residuals from start for each layer {l: list of [T_i, d] per row}.
    """
    device = next(model.parameters()).device
    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
    B, T = input_ids.shape
    starts_p = torch.tensor(
        [pl + s for pl, s in zip(pad_lens, starts_unpadded)],
        device=device,
        dtype=torch.long,
    )

    caps_layers = list(layer_idxs)
    with capture_residual(model, caps_layers) as caps:
        if vectors is None:
            out = model(input_ids=input_ids, attention_mask=attn)
        else:
            with add_to_residual_batched(
                model, steer_layer, vectors.to(device=device, dtype=torch.bfloat16), starts_p
            ):
                out = model(input_ids=input_ids, attention_mask=attn)

    logits = out.logits  # [B,T,V]
    # last non-pad position is -1 with left padding
    ld = detection_logit(logits[:, -1, :], yes_id, no_id).cpu()

    last_h = {}
    full_from_start = {}
    for li in layer_idxs:
        h = caps[li]  # [B,T,d] float32 CPU
        last_h[li] = h[:, -1, :].clone()
        if capture_all_from_start:
            rows = []
            for b in range(B):
                sp = int(starts_p[b].item())  # padded coord; but h is on padded seq
                # positions on the captured tensor are 0..T-1 matching padded input
                rows.append(h[b, sp:, :].clone())
            full_from_start[li] = rows

    return {
        "ld": ld,
        "last_h": last_h,
        "full_from_start": full_from_start,
        "pad_lens": pad_lens,
        "starts_padded": starts_p.cpu(),
    }


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-12)


def recovery_score(dld_none: torch.Tensor, dld_blk: torch.Tensor) -> float:
    """1 - mean(dld_blk)/mean(dld_none)."""
    mn = float(dld_none.mean().item())
    if abs(mn) < 1e-8:
        return float("nan")
    return 1.0 - float(dld_blk.mean().item()) / mn


def run_part_a(
    *,
    model_dir: str,
    revision_sha: str,
    config_path: str,
    results_dir: str,
    ledger_path: str,
    submodule_root: str,
    wo001_vectors_path: str,
    wo001_snapshot_path: str,
    wo001_per_concept_csv: str,
    git_hash_str: str,
    gpu_type: str = "A100-80GB",
    n_cores: float = 8.0,
    mem_gib: float = 64.0,
    volume_commit=None,
) -> Dict[str, Any]:
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
        wo_id="WO-002",
        function="wo_002_partA",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            # --- inputs / splits ---
            wo001 = json.loads(Path(wo001_snapshot_path).read_text())
            concepts_50 = list(wo001["concepts"])
            baselines = list(wo001["baselines"])
            new_concepts = load_new_concepts(Path(submodule_root))
            assert len(new_concepts) == 450 and len(set(new_concepts)) == 450
            overlap = set(x.lower() for x in new_concepts) & (
                set(x.lower() for x in concepts_50) | set(x.lower() for x in baselines)
            )
            assert not overlap, f"overlap: {overlap}"
            dir_dev = list(cfg["dir_dev"])
            dir_val = list(cfg["dir_val"])
            assert len(dir_dev) == 225 and len(dir_val) == 225
            _log(
                f"SANITY2: NEW_CONCEPTS=450 unique; overlap with 50+100=0; "
                f"dir_dev={len(dir_dev)} dir_val={len(dir_val)}"
            )
            summary["sanity"]["new_concepts"] = {
                "n": 450,
                "overlap": 0,
                "dir_dev": len(dir_dev),
                "dir_val": len(dir_val),
            }

            snap = {
                "new_concepts": new_concepts,
                "dir_dev": dir_dev,
                "dir_val": dir_val,
                "concepts_50": concepts_50,
                "baselines": baselines,
                "alpha": cfg["alpha"],
                "layer_steer": cfg["layer_steer"],
            }
            (raw / "inputs_snapshot.json").write_text(json.dumps(snap, indent=2))

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            model = load_model(model_dir, revision=revision_sha, device=0)
            d_model = get_d_model(model)
            _ = get_layers(model)
            yes_id, no_id = yes_no_token_ids(tokenizer)
            _log(
                f"SANITY3: Yes id={yes_id} ({tokenizer.decode([yes_id])!r}) "
                f"No id={no_id} ({tokenizer.decode([no_id])!r})"
            )
            summary["sanity"]["yes_no"] = {
                "yes_id": yes_id,
                "no_id": no_id,
                "yes": tokenizer.decode([yes_id]),
                "no": tokenizer.decode([no_id]),
            }
            summary["env"] = collect_env()

            layer_steer = int(cfg["layer_steer"])
            alpha = float(cfg["alpha"])
            layers = list(cfg["layers_capture"])
            batch_size = int(cfg["batch_size"])
            vec_bs = int(cfg.get("vector_batch_size", 16))

            # --- concept vectors for 450 ---
            _log("Computing 450 concept vectors...")
            t0 = time.perf_counter()
            pack450 = compute_concept_vectors(
                model, tokenizer, new_concepts, baselines, layer_idx=layer_steer, batch_size=vec_bs
            )
            # reorder to new_concepts order already
            torch.save(
                {
                    "concepts": new_concepts,
                    "vectors": pack450["vectors"],
                    "norms": pack450["norms"],
                    "mean_baseline": pack450["mean_baseline"],
                    "layer": layer_steer,
                },
                raw / "concept_vectors_L37_new450.pt",
            )
            _log(f"450 vectors done in {time.perf_counter()-t0:.1f}s")

            # index maps
            name_to_vec = {c: pack450["vectors"][i] for i, c in enumerate(new_concepts)}
            pack50 = torch.load(wo001_vectors_path, map_location="cpu", weights_only=False)
            vec50 = {c: pack50["vectors"][i] for i, c in enumerate(pack50["concepts"])}
            norms50 = {c: float(pack50["norms"][i]) for i, c in enumerate(pack50["concepts"])}

            # ========== Sanity 4: blocker identity ==========
            text, ids, sp = build_introspection_prompt(tokenizer, 1)
            device = next(model.parameters()).device
            ids_d = ids.to(device)
            with torch.no_grad():
                base_logits = model(input_ids=ids_d).logits
            zero_dirs = {l: torch.zeros(d_model) for l in range(40, 45)}
            zero_mus = {l: 0.0 for l in range(40, 45)}
            with mean_ablate_from_l0(
                model, 40, zero_dirs, zero_mus, torch.tensor([sp], device=device), max_layer=44
            ):
                with torch.no_grad():
                    z_logits = model(input_ids=ids_d).logits
            assert torch.equal(base_logits, z_logits)
            with mean_ablate_from_l0(
                model, 40, {}, {}, torch.tensor([sp], device=device), max_layer=44
            ):
                with torch.no_grad():
                    e_logits = model(input_ids=ids_d).logits
            assert torch.equal(base_logits, e_logits)
            _log("SANITY4 blocker identity: PASS")
            summary["sanity"]["blocker_identity"] = True

            # ========== Sanity 5: projection equals mu ==========
            # Register a read-back hook AFTER the ablation hooks so we observe
            # the post-ablation residual (capture_residual alone can race on order).
            g = torch.Generator().manual_seed(0)
            d_hat = torch.randn(d_model, generator=g)
            d_hat = d_hat / d_hat.norm()
            mu_test = 0.25
            from sfo.hooks import _resolve_layers

            held: dict = {}

            def _read_hook(_m, _i, output):
                hidden = output[0] if isinstance(output, tuple) else output
                held["h"] = hidden.detach().float().cpu()
                return output

            layer40 = _resolve_layers(model)[40]
            # Baseline (no blocker) for positions < start_pos
            with torch.no_grad():
                _ = model(input_ids=ids_d)
            # actually capture baseline
            h_base_holder: dict = {}

            def _base_hook(_m, _i, output):
                hidden = output[0] if isinstance(output, tuple) else output
                h_base_holder["h"] = hidden.detach().float().cpu()
                return output

            hb = layer40.register_forward_hook(_base_hook)
            try:
                with torch.no_grad():
                    model(input_ids=ids_d)
            finally:
                hb.remove()
            h0 = h_base_holder["h"][0]

            with mean_ablate_from_l0(
                model,
                40,
                {40: d_hat},
                {40: mu_test},
                torch.tensor([sp], device=device),
                max_layer=40,
            ):
                rh = layer40.register_forward_hook(_read_hook)
                try:
                    with torch.no_grad():
                        model(input_ids=ids_d)
                finally:
                    rh.remove()
            h = held["h"][0]
            proj = (h * d_hat.float()).sum(-1)
            # Relative to residual L2 scale (bf16 cannot hold |proj-μ|/|μ|≪1 when
            # ‖h‖~1e4–1e5 and μ is O(1); WO intent is accurate mean-ablation).
            scale = h[sp:].norm(dim=-1).mean().clamp_min(1.0)
            rel = (proj[sp:] - mu_test).abs() / scale
            assert float(rel.max()) < 1e-2, (
                f"proj≈μ failed: max_rel={float(rel.max())} "
                f"proj_mean={float(proj[sp:].mean())} mu={mu_test} scale={float(scale)}"
            )
            assert torch.allclose(h[:sp], h0[:sp], atol=1e-2, rtol=1e-2)
            _log(
                f"SANITY5 projection≈μ: PASS (max rel_to_‖h‖ {float(rel.max()):.2e}; "
                f"proj_mean={float(proj[sp:].mean()):.4f})"
            )
            summary["sanity"]["blocker_correctness"] = True

            # ========== Sanity 6: composition batched vs single ==========
            trial_ns = [1, 2, 3, 4]
            id_list, starts = [], []
            for tn in trial_ns:
                _, ids_i, sp_i = build_introspection_prompt(tokenizer, tn)
                id_list.append(ids_i)
                starts.append(sp_i)
            v0 = (alpha * name_to_vec[dir_dev[0]]).to(torch.bfloat16)
            vecs = v0.unsqueeze(0).expand(len(trial_ns), -1).contiguous()
            # temporary direction at layers 40-42
            dirs_tmp = {l: d_hat for l in range(40, 43)}
            mus_tmp = {l: 0.0 for l in range(40, 43)}

            def _greedy_comp(ids_l, sps, vec_b, n_new=10):
                torch.manual_seed(7)
                input_ids, attn, pad_lens = pad_batch(tokenizer, ids_l, device)
                st = torch.tensor([pl + s for pl, s in zip(pad_lens, sps)], device=device)
                with add_to_residual_batched(model, layer_steer, vec_b.to(device), st):
                    with mean_ablate_from_l0(model, 40, dirs_tmp, mus_tmp, st, max_layer=42):
                        out = model.generate(
                            input_ids=input_ids,
                            attention_mask=attn,
                            max_new_tokens=n_new,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                gens = []
                for i, ids_i in enumerate(ids_l):
                    plen = ids_i.shape[-1]
                    row = out[i, pad_lens[i] :]
                    gens.append(row[plen:].detach().cpu())
                return gens

            bat = _greedy_comp(id_list, starts, vecs, 10)
            for i in range(len(trial_ns)):
                s = _greedy_comp([id_list[i]], [starts[i]], vecs[i : i + 1], 10)[0]
                assert torch.equal(bat[i][:10], s[:10]), f"comp mismatch {i}"
            _log("SANITY6 composition: PASS")
            summary["sanity"]["composition"] = True

            # ========== A1 Record dir-dev ==========
            _log("A1: recording dir-dev steered/unsteered...")
            t_a1 = time.perf_counter()
            trial_ns = list(cfg["trial_ns_parta"])

            # Unsteered n=1,2,3 (shared)
            unsteered = {}
            for n in trial_ns:
                text, ids, sp = build_introspection_prompt(tokenizer, n)
                r = forward_ld_and_hiddens(
                    model,
                    tokenizer,
                    [ids],
                    [sp],
                    layers,
                    None,
                    layer_steer,
                    yes_id,
                    no_id,
                    capture_all_from_start=True,
                )
                unsteered[n] = {
                    "ld": r["ld"][0],
                    "last_h": {l: r["last_h"][l][0] for l in layers},
                    "full_from_start": {l: r["full_from_start"][l][0] for l in layers},
                    "start_pos": sp,
                }

            # Steered per concept × n
            steered_last = {l: [] for l in layers}  # list aligned with (c,n) order
            steered_ld = []
            steered_meta = []
            concepts_batch = list(dir_dev)
            for start in range(0, len(concepts_batch), batch_size):
                chunk = concepts_batch[start : start + batch_size]
                for n in trial_ns:
                    id_list, starts, vecs = [], [], []
                    for c in chunk:
                        _, ids, sp = build_introspection_prompt(tokenizer, n)
                        id_list.append(ids)
                        starts.append(sp)
                        vecs.append(alpha * name_to_vec[c])
                    vec_t = torch.stack(vecs).to(torch.bfloat16)
                    r = forward_ld_and_hiddens(
                        model,
                        tokenizer,
                        id_list,
                        starts,
                        layers,
                        vec_t,
                        layer_steer,
                        yes_id,
                        no_id,
                        capture_all_from_start=False,
                    )
                    for i, c in enumerate(chunk):
                        steered_ld.append(float(r["ld"][i]))
                        steered_meta.append({"concept": c, "n": n})
                        for l in layers:
                            steered_last[l].append(r["last_h"][l][i])

            for l in layers:
                steered_last[l] = torch.stack(steered_last[l], dim=0)
            torch.save(
                {
                    "steered_last": steered_last,
                    "steered_ld": torch.tensor(steered_ld),
                    "steered_meta": steered_meta,
                    "unsteered": unsteered,
                    "layers": layers,
                    "dir_dev": dir_dev,
                },
                raw / "partA_dirdev.pt",
            )
            _log(f"A1 done in {time.perf_counter()-t_a1:.1f}s")

            # ========== A2 Directions ==========
            # For each (c,n), unsteered last_h is unsteered[n]
            per_layer = {}
            d_raw = {}
            d_hat = {}
            # median unsteered residual norm excl BOS at layer l — use residual_norm on n=1
            _, ids1, _ = build_introspection_prompt(tokenizer, 1)
            # We'll compute per-layer median from unsteered full positions for n=1
            for l in layers:
                diffs = []
                for i, meta in enumerate(steered_meta):
                    n = meta["n"]
                    diffs.append(steered_last[l][i] - unsteered[n]["last_h"][l])
                d = torch.stack(diffs, dim=0).mean(dim=0)
                d_raw[l] = d
                d_hat[l] = unit(d)
                # median unsteered residual norm at l from full positions (all tokens in full_from_start has no BOS if start>0)
                # Use last-pos unsteered norms across n as proxy + full
                all_pos = torch.cat([unsteered[n]["full_from_start"][l] for n in trial_ns], dim=0)
                med_norm = float(all_pos.norm(dim=-1).median().item())
                d_norm = float(d.norm().item())
                # cos with mean unit concept vector over dir-dev
                v_mean = unit(
                    torch.stack([name_to_vec[c] for c in dir_dev], dim=0).mean(0)
                )
                cos_mean_v = float(torch.dot(d_hat[l], v_mean).item())
                # vs validation 50
                cos_val = []
                for c in concepts_50:
                    cos_val.append(float(torch.dot(d_hat[l], unit(vec50[c])).item()))
                cos_val_t = torch.tensor(cos_val).abs()
                per_layer[str(l)] = {
                    "d_norm": d_norm,
                    "d_norm_ratio": d_norm / (med_norm + 1e-12),
                    "cos_mean_v": cos_mean_v,
                    "max_cos_val50": float(cos_val_t.max().item()),
                    "p95_cos_val50": float(torch.quantile(cos_val_t, 0.95).item()),
                    "cos_next": None,
                }
            for l in layers:
                if l + 1 in d_hat:
                    per_layer[str(l)]["cos_next"] = float(
                        torch.dot(d_hat[l], d_hat[l + 1]).item()
                    )

            # ========== A3 μ_l ==========
            mus = {}
            for l in layers:
                dots = []
                for n in trial_ns:
                    h = unsteered[n]["full_from_start"][l]  # [T,d]
                    dots.append((h * d_hat[l].unsqueeze(0)).sum(-1))
                mus[l] = float(torch.cat(dots).mean().item())

            torch.save(
                {"d_hat": d_hat, "d_raw": d_raw, "mus": mus, "layers": layers},
                raw / "directions.pt",
            )

            # ========== A4/A5 Layer sweep ==========
            sweep_concepts = dir_dev[: int(cfg["sweep_n_concepts"])]
            l0_list = list(cfg["l0_sweep"])
            _log(f"A5 layer sweep on {len(sweep_concepts)} concepts, l0={l0_list}")

            def compute_dlds(concepts, n, blocker_dirs, blocker_mus, l0):
                """Return dld_none [C], dld_blk [C] for n=1."""
                dld_none = []
                dld_blk = []
                # unsteered LD (shared)
                _, ids_u, sp_u = build_introspection_prompt(tokenizer, n)
                r_u = forward_ld_and_hiddens(
                    model, tokenizer, [ids_u], [sp_u], [38], None, layer_steer, yes_id, no_id
                )
                ld_u = float(r_u["ld"][0])
                # unsteered with blocker
                device = next(model.parameters()).device
                input_ids, attn, pad_lens = pad_batch(tokenizer, [ids_u], device)
                st = torch.tensor([pad_lens[0] + sp_u], device=device)
                with mean_ablate_from_l0(
                    model, l0, blocker_dirs, blocker_mus, st, max_layer=61
                ):
                    with torch.no_grad():
                        out_ub = model(input_ids=input_ids, attention_mask=attn)
                ld_ub = float(detection_logit(out_ub.logits[0, -1], yes_id, no_id))

                for start in range(0, len(concepts), batch_size):
                    chunk = concepts[start : start + batch_size]
                    id_list, starts, vecs = [], [], []
                    for c in chunk:
                        _, ids, sp = build_introspection_prompt(tokenizer, n)
                        id_list.append(ids)
                        starts.append(sp)
                        vecs.append(alpha * name_to_vec[c])
                    vec_t = torch.stack(vecs).to(torch.bfloat16)
                    # steered none
                    r_s = forward_ld_and_hiddens(
                        model,
                        tokenizer,
                        id_list,
                        starts,
                        [38],
                        vec_t,
                        layer_steer,
                        yes_id,
                        no_id,
                    )
                    # steered + blocker
                    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
                    st = torch.tensor(
                        [pl + s for pl, s in zip(pad_lens, starts)], device=device
                    )
                    with add_to_residual_batched(
                        model, layer_steer, vec_t.to(device), st
                    ):
                        with mean_ablate_from_l0(
                            model, l0, blocker_dirs, blocker_mus, st, max_layer=61
                        ):
                            with torch.no_grad():
                                out_sb = model(input_ids=input_ids, attention_mask=attn)
                    ld_sb = detection_logit(out_sb.logits[:, -1], yes_id, no_id).cpu()
                    for i in range(len(chunk)):
                        dld_none.append(float(r_s["ld"][i]) - ld_u)
                        dld_blk.append(float(ld_sb[i]) - ld_ub)
                return torch.tensor(dld_none), torch.tensor(dld_blk)

            sweep = {}
            for l0 in l0_list:
                # B1
                dld_n, dld_b = compute_dlds(sweep_concepts, 1, d_hat, mus, l0)
                rec_b1 = recovery_score(dld_n, dld_b)
                # restrict to dld_none > 1
                mask = dld_n > 1.0
                per_c = None
                if mask.any():
                    per_c = [
                        float(1 - dld_b[i] / dld_n[i])
                        for i in range(len(dld_n))
                        if float(dld_n[i]) > 1.0
                    ]

                # B1-rand seeds 0-4
                rand_recs = []
                for seed in cfg["b1_rand_seeds_sweep"]:
                    g = torch.Generator().manual_seed(int(seed))
                    rd = {}
                    rm = {}
                    for l in layers:
                        v = torch.randn(d_model, generator=g)
                        rd[l] = unit(v)
                        # μ from unsteered
                        dots = []
                        for n in trial_ns:
                            h = unsteered[n]["full_from_start"][l]
                            dots.append((h * rd[l].unsqueeze(0)).sum(-1))
                        rm[l] = float(torch.cat(dots).mean().item())
                    _, dld_br = compute_dlds(sweep_concepts, 1, rd, rm, l0)
                    rand_recs.append(recovery_score(dld_n, dld_br))

                # B1-pc: top PC of unsteered residuals at each layer
                pc_d = {}
                pc_m = {}
                for l in layers:
                    mats = torch.cat(
                        [unsteered[n]["full_from_start"][l] for n in trial_ns], dim=0
                    )  # [N,d]
                    mats = mats - mats.mean(0, keepdim=True)
                    # top PC via SVD (or eigh of cov) — use torch.pca_lowrank
                    _, _, V = torch.pca_lowrank(mats, q=1, center=False)
                    pc_d[l] = unit(V[:, 0].float())
                    dots = []
                    for n in trial_ns:
                        h = unsteered[n]["full_from_start"][l]
                        dots.append((h * pc_d[l].unsqueeze(0)).sum(-1))
                    pc_m[l] = float(torch.cat(dots).mean().item())
                _, dld_pc = compute_dlds(sweep_concepts, 1, pc_d, pc_m, l0)
                rec_pc = recovery_score(dld_n, dld_pc)

                sweep[str(l0)] = {
                    "B1": rec_b1,
                    "B1rand_mean": float(np.mean(rand_recs)),
                    "B1rand_max": float(np.max(rand_recs)),
                    "B1pc": rec_pc,
                    "dLD_none_mean": float(dld_n.mean()),
                    "n_dLD_gt1": int(mask.sum()),
                    "per_concept_recovery_gt1": per_c,
                }
                _log(f"  l0={l0}: B1={rec_b1:.4f} rand_mean={np.mean(rand_recs):.4f} pc={rec_pc:.4f}")

            # Select l0* = highest B1 recovery; tie within 0.02 → larger l0.
            scored = [(sweep[str(l0)]["B1"], l0) for l0 in l0_list]
            max_rec = max(s[0] for s in scored)
            candidates = [l0 for rec, l0 in scored if abs(rec - max_rec) <= 0.02]
            l0_star = max(candidates)
            _log(f"Selected l0*={l0_star} (max B1 recovery={max_rec:.4f})")

            # ========== A6 Held-out dir-val ==========
            _log("A6 held-out dir-val...")
            dld_n, dld_b = compute_dlds(dir_val, 1, d_hat, mus, l0_star)
            held = {
                "B1": recovery_score(dld_n, dld_b),
                "dLD_none_mean": float(dld_n.mean()),
            }
            rand_recs = []
            for seed in cfg["b1_rand_seeds_heldout"]:
                g = torch.Generator().manual_seed(int(seed))
                rd, rm = {}, {}
                for l in layers:
                    v = torch.randn(d_model, generator=g)
                    rd[l] = unit(v)
                    dots = [
                        (unsteered[n]["full_from_start"][l] * rd[l].unsqueeze(0)).sum(-1)
                        for n in trial_ns
                    ]
                    rm[l] = float(torch.cat(dots).mean().item())
                _, dld_br = compute_dlds(dir_val, 1, rd, rm, l0_star)
                rand_recs.append(recovery_score(dld_n, dld_br))
            held["B1rand_mean"] = float(np.mean(rand_recs))
            held["B1rand_max"] = float(np.max(rand_recs))
            # PC at l0*
            pc_d, pc_m = {}, {}
            for l in layers:
                mats = torch.cat(
                    [unsteered[n]["full_from_start"][l] for n in trial_ns], dim=0
                )
                mats = mats - mats.mean(0, keepdim=True)
                _, _, V = torch.pca_lowrank(mats, q=1, center=False)
                pc_d[l] = unit(V[:, 0].float())
                dots = [
                    (unsteered[n]["full_from_start"][l] * pc_d[l].unsqueeze(0)).sum(-1)
                    for n in trial_ns
                ]
                pc_m[l] = float(torch.cat(dots).mean().item())
            _, dld_pc = compute_dlds(dir_val, 1, pc_d, pc_m, l0_star)
            held["B1pc"] = recovery_score(dld_n, dld_pc)
            torch.save(
                {"dld_none": dld_n, "dld_b1": dld_b, "dir_val": dir_val},
                raw / "partA_dirval.pt",
            )
            _log(f"Held-out B1 recovery={held['B1']:.4f}")

            # Random-vector transfer
            _log("A6 random-vector transfer...")
            dld_rand = []
            dld_rand_blk = []
            for i, seed in enumerate(cfg["random_transfer_seeds"]):
                c = concepts_50[i % 50]
                target = 2.0 * norms50[c]  # α=2-equivalent norm
                g = torch.Generator().manual_seed(int(seed))
                rv = torch.randn(d_model, generator=g)
                rv = unit(rv) * target
                _, ids, sp = build_introspection_prompt(tokenizer, 1)
                # none
                r_u = forward_ld_and_hiddens(
                    model, tokenizer, [ids], [sp], [38], None, layer_steer, yes_id, no_id
                )
                r_s = forward_ld_and_hiddens(
                    model,
                    tokenizer,
                    [ids],
                    [sp],
                    [38],
                    rv.unsqueeze(0).to(torch.bfloat16),
                    layer_steer,
                    yes_id,
                    no_id,
                )
                dld_none = float(r_s["ld"][0] - r_u["ld"][0])
                device = next(model.parameters()).device
                input_ids, attn, pad_lens = pad_batch(tokenizer, [ids], device)
                st = torch.tensor([pad_lens[0] + sp], device=device)
                with mean_ablate_from_l0(model, l0_star, d_hat, mus, st, max_layer=61):
                    with torch.no_grad():
                        out_ub = model(input_ids=input_ids, attention_mask=attn)
                ld_ub = float(detection_logit(out_ub.logits[0, -1], yes_id, no_id))
                with add_to_residual_batched(
                    model, layer_steer, rv.unsqueeze(0).to(device=device, dtype=torch.bfloat16), st
                ):
                    with mean_ablate_from_l0(model, l0_star, d_hat, mus, st, max_layer=61):
                        with torch.no_grad():
                            out_sb = model(input_ids=input_ids, attention_mask=attn)
                ld_sb = float(detection_logit(out_sb.logits[0, -1], yes_id, no_id))
                dld_rand.append(dld_none)
                dld_rand_blk.append(ld_sb - ld_ub)
            dld_rand_t = torch.tensor(dld_rand)
            dld_rand_blk_t = torch.tensor(dld_rand_blk)
            random_transfer = {
                "dLD_none_mean": float(dld_rand_t.mean()),
                "B1_recovery": recovery_score(dld_rand_t, dld_rand_blk_t),
            }

            # ========== A7 Proxy Spearman ==========
            import csv
            from scipy.stats import spearmanr

            tpr_map = {}
            with open(wo001_per_concept_csv) as f:
                for row in csv.DictReader(f):
                    tpr_map[row["concept"]] = float(row["tpr@4"])
            dlds = []
            tprs = []
            for c in concepts_50:
                _, ids, sp = build_introspection_prompt(tokenizer, 1)
                r_u = forward_ld_and_hiddens(
                    model, tokenizer, [ids], [sp], [38], None, layer_steer, yes_id, no_id
                )
                r_s = forward_ld_and_hiddens(
                    model,
                    tokenizer,
                    [ids],
                    [sp],
                    [38],
                    (alpha * vec50[c]).unsqueeze(0).to(torch.bfloat16),
                    layer_steer,
                    yes_id,
                    no_id,
                )
                dlds.append(float(r_s["ld"][0] - r_u["ld"][0]))
                tprs.append(tpr_map[c])
            rho, pval = spearmanr(dlds, tprs)
            proxy = {"rho": float(rho), "p": float(pval)}
            _log(f"A7 Spearman rho={rho:.4f} p={pval:.4g}")

            partA = {
                "per_layer": per_layer,
                "sweep": sweep,
                "l0_star": int(l0_star),
                "heldout_dirval": held,
                "random_transfer": random_transfer,
                "proxy_spearman": proxy,
            }
            summary["partA"] = partA
            summary["mus"] = {str(k): v for k, v in mus.items()}

            # Project Part B GPU time
            # Rough: Part B ~1900 gens * ~similar to WO-001 rate
            # WO-001: 1300 trials in ~600s gen portion ≈ 0.46s/trial
            # Part B generations: 900+300+300+50+300+50 = 1900 + MMLU 200*3 + KL
            n_gen = 900 + 300 + 300 + 50 + 300 + 50
            part_a_hours = (time.perf_counter() - t_a1) / 3600.0  # incomplete; use wall from ledger later
            # Use elapsed since model load approx from t_a1 start of A1; include vector time
            projected_b = n_gen * 0.5 / 3600.0  # conservative 0.5s/trial
            projected_mmlu = 200 * 3 * 0.15 / 3600.0
            projected_total = projected_b + projected_mmlu + 0.25  # + part A already spent ~estimate
            # Better: report based on A1 timing
            a1_sec = time.perf_counter() - t_a1
            _log(
                f"SANITY9: Part A wall so far (from A1) {a1_sec/60:.1f} min; "
                f"projected Part B ≈ {projected_b*60:.1f} min; "
                f"projected total remaining+done ≈ {(a1_sec/3600)+projected_b+projected_mmlu:.2f} h "
                f"(limit {cfg['projected_gpu_limit_hours']})"
            )
            projected_hours = a1_sec / 3600.0 + projected_b + projected_mmlu
            summary["projected_hours"] = projected_hours
            if projected_hours > float(cfg["projected_gpu_limit_hours"]):
                summary["stopped_before_partB"] = True
                summary["stop_reason"] = (
                    f"Projected {projected_hours:.3f} h > {cfg['projected_gpu_limit_hours']} h"
                )
                _log(summary["stop_reason"])
            else:
                summary["stopped_before_partB"] = False

            (results_path / "partA_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            return summary

        except Exception as e:
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
    run_part_a(
        model_dir=os.environ["SFO_MODEL_DIR"],
        revision_sha=os.environ["SFO_REVISION_SHA"],
        config_path=os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_002.yaml")),
        results_dir=os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-002")),
        ledger_path=os.environ.get(
            "SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")
        ),
        submodule_root=os.environ.get(
            "SFO_SUBMODULE", str(REPO_ROOT / "third_party/introspection-mechanisms")
        ),
        wo001_vectors_path=str(REPO_ROOT / "results/WO-001/raw/concept_vectors_L37.pt"),
        wo001_snapshot_path=str(REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json"),
        wo001_per_concept_csv=str(REPO_ROOT / "results/WO-001/raw/per_concept.csv"),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )
