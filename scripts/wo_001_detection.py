"""WO-001 GPU: sanity checks, concept vectors, pilot, main generation."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.hooks import add_to_residual_batched  # noqa: E402
from sfo.inputs_ast import load_reference_inputs  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import (  # noqa: E402
    format_chat,
    get_d_model,
    get_layers,
    load_model,
    load_model_revision,
    load_tokenizer,
)
from sfo.prompts import (  # noqa: E402
    YESBIAS_QUESTIONS,
    build_introspection_prompt,
    build_yesbias_prompt,
    decode_new_tokens,
)
from sfo.provenance import collect_env, set_seeds  # noqa: E402
from sfo.vectors import (  # noqa: E402
    compute_concept_vectors,
    pad_batch,
    residual_norm_diagnostics,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _sampling_config_dict(model) -> Dict[str, Any]:
    gc = getattr(model, "generation_config", None)
    if gc is None:
        return {}
    keys = [
        "temperature",
        "top_k",
        "top_p",
        "typical_p",
        "repetition_penalty",
        "do_sample",
        "num_beams",
        "eos_token_id",
        "pad_token_id",
        "max_length",
    ]
    out = {}
    for k in keys:
        if hasattr(gc, k):
            v = getattr(gc, k)
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)
    return out


def _make_random_vector(
    d_model: int, seed: int, target_norm: float, device
) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    v = torch.randn(d_model, generator=g, dtype=torch.float32)
    v = v / v.norm() * target_norm
    return v.to(device=device, dtype=torch.bfloat16)


def _generate_batch(
    model,
    tokenizer,
    id_list: List[torch.Tensor],
    vectors: Optional[torch.Tensor],
    start_positions_unpadded: List[int],
    layer_idx: int,
    max_new_tokens: int,
    temperature: float,
    seed: int,
) -> Tuple[List[str], List[int], Dict[str, Any]]:
    """
    Left-pad batch, optionally steer from padded start positions, sample.
    Returns responses, pad_lens, meta.
    """
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
    starts_padded = torch.tensor(
        [pl + sp for pl, sp in zip(pad_lens, start_positions_unpadded)],
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
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )

    with torch.no_grad():
        if vectors is None:
            out = model.generate(**gen_kwargs)
        else:
            vecs = vectors.to(device=device, dtype=torch.bfloat16)
            with add_to_residual_batched(model, layer_idx, vecs, starts_padded):
                out = model.generate(**gen_kwargs)

    responses = []
    for i in range(out.shape[0]):
        # full sequence including left pad; content starts at pad_lens[i]
        row = out[i, pad_lens[i] :]
        responses.append(decode_new_tokens(tokenizer, row, prompt_lens[i]))
    return responses, pad_lens, {"starts_padded": starts_padded.tolist(), "seed": seed}


def run_detection(
    *,
    model_dir: str,
    revision_sha: str,
    config_path: str,
    results_dir: str,
    ledger_path: str,
    submodule_root: str,
    git_hash_str: str,
    gpu_type: str = "A100-80GB",
    n_cores: float = 8.0,
    mem_gib: float = 64.0,
    volume_commit=None,
    skip_main: bool = False,
) -> Dict[str, Any]:
    results_path = Path(results_dir)
    raw_dir = results_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    layer = int(cfg["layer"])
    alphas = [float(a) for a in cfg["alphas"]]
    batch_size = int(cfg["batch_size"])
    max_new = int(cfg["max_new_tokens"])
    temperature = float(cfg["temperature"])
    seed_base = int(cfg["seed_base"])
    vec_bs = int(cfg.get("vector_batch_size", 16))

    summary: Dict[str, Any] = {
        "git_hash": git_hash_str,
        "model_revision": revision_sha,
        "sanity": {},
        "errors": [],
    }

    with gpu_ledger(
        ledger_path,
        wo_id="WO-001",
        function="wo_001_detection",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            # --- inputs ---
            inputs = load_reference_inputs(Path(submodule_root))
            concepts = inputs["concepts"]
            baselines = inputs["baselines"]
            snapshot = {
                "concepts": concepts,
                "baselines": baselines,
                "claims_detection_grading_prompt": inputs[
                    "claims_detection_grading_prompt"
                ],
                "correct_concept_identification_grading_prompt": inputs[
                    "correct_concept_identification_grading_prompt"
                ],
                "prompt_fill_doc": inputs["prompt_fill_doc"],
                "source_paths": inputs["source_paths"],
            }
            (raw_dir / "inputs_snapshot.json").write_text(
                json.dumps(snapshot, indent=2)
            )
            _log(
                f"Inputs: {len(concepts)} concepts, {len(baselines)} baselines; "
                f"prompts non-empty"
            )

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            model = load_model(model_dir, revision=revision_sha, device=0)
            d_model = get_d_model(model)
            _ = get_layers(model)
            sampling = _sampling_config_dict(model)
            sampling["wo_temperature"] = temperature
            sampling["wo_max_new_tokens"] = max_new
            sampling["wo_do_sample"] = True
            summary["sampling"] = sampling
            summary["env"] = collect_env()
            _log(f"Sampling / generation_config defaults: {json.dumps(sampling)}")

            # ========== Sanity 2: prompts / start_pos ==========
            text1, ids1, sp1 = build_introspection_prompt(tokenizer, 1)
            toks = ids1[0].tolist()
            window = toks[max(0, sp1 - 3) : sp1 + 4]
            decoded_window = [tokenizer.decode([t]) for t in window]
            token_after = tokenizer.decode([toks[sp1 + 1]])
            _log(f"SANITY2 introspection n=1 formatted:\n{text1}")
            _log(f"SANITY2 start_pos={sp1} window_ids={window} decoded={decoded_window}")
            _log(f"SANITY2 token at start_pos+1: {token_after!r}")
            assert "Trial" in token_after or token_after.strip().startswith("Trial") or "Tri" in token_after, (
                f"expected Trial piece at start_pos+1, got {token_after!r}"
            )
            # BOS
            assert int((ids1[0] == tokenizer.bos_token_id).sum()) == 1

            ytext, yids, ysp = build_yesbias_prompt(
                tokenizer, 1, YESBIAS_QUESTIONS[0]
            )
            yt = yids[0].tolist()
            ywin = yt[max(0, ysp - 3) : ysp + 4]
            ydec = [tokenizer.decode([t]) for t in ywin]
            yafter = tokenizer.decode([yt[ysp + 1]])
            _log(f"SANITY2 yesbias formatted:\n{ytext}")
            _log(f"SANITY2 yesbias start_pos={ysp} window={ydec} after={yafter!r}")
            assert "Trial" in yafter or "Tri" in yafter
            assert int((yids[0] == tokenizer.bos_token_id).sum()) == 1
            summary["sanity"]["start_pos"] = {
                "introspection": {"start_pos": sp1, "token_after": token_after},
                "yesbias": {"start_pos": ysp, "token_after": yafter},
            }

            # ========== Sanity 4: alpha=0 bit-identical first-step logits ==========
            device = next(model.parameters()).device
            for n_try in (1, 2):
                text, ids, sp = build_introspection_prompt(tokenizer, n_try)
                ids_d = ids.to(device)
                with torch.no_grad():
                    base_logits = model(input_ids=ids_d).logits
                zero = torch.zeros(1, d_model, dtype=torch.bfloat16, device=device)
                starts = torch.tensor([sp], device=device, dtype=torch.long)
                with add_to_residual_batched(model, layer, zero, starts):
                    with torch.no_grad():
                        z_logits = model(input_ids=ids_d).logits
                assert torch.equal(base_logits, z_logits), f"alpha0 mismatch trial {n_try}"
            _log("SANITY4 alpha=0 first-step logits bit-identical: PASS")
            summary["sanity"]["alpha0"] = True

            # ========== Vectors ==========
            _log("Computing concept vectors...")
            t0 = time.perf_counter()
            vec_pack = compute_concept_vectors(
                model,
                tokenizer,
                concepts,
                baselines,
                layer_idx=layer,
                batch_size=vec_bs,
            )
            vec_time = time.perf_counter() - t0
            _log(
                f"Vectors done in {vec_time:.1f}s; mean pairwise cos="
                f"{vec_pack['mean_pairwise_cosine']:.4f} max="
                f"{vec_pack['max_pairwise_cosine']:.4f}"
            )
            torch.save(
                {
                    "concepts": concepts,
                    "vectors": vec_pack["vectors"],
                    "norms": vec_pack["norms"],
                    "mean_baseline": vec_pack["mean_baseline"],
                    "layer": layer,
                },
                raw_dir / "concept_vectors_L37.pt",
            )
            summary["sanity"]["vectors"] = {
                "mean_pairwise_cosine": vec_pack["mean_pairwise_cosine"],
                "max_pairwise_cosine": vec_pack["max_pairwise_cosine"],
                "norms": [float(x) for x in vec_pack["norms"].tolist()],
            }

            # Norm diagnostics on introspection n=1 unsteered
            nd = residual_norm_diagnostics(model, tokenizer, ids1, layer_idx=layer)
            norms = vec_pack["norms"]
            med_ratio2 = float((2.0 * norms / nd["resid_median_excl_bos"]).median())
            med_ratio4 = float((4.0 * norms / nd["resid_median_excl_bos"]).median())
            summary["norms"] = {
                **nd,
                "ratio_alpha2": med_ratio2,
                "ratio_alpha4": med_ratio4,
            }
            _log(f"Norm diagnostics: {summary['norms']}")

            # ========== Sanity 5: batched vs single greedy ==========
            _log("SANITY5 batched hook GPU check...")
            trial_ns = [1, 2, 3, 4]
            id_list = []
            starts = []
            for tn in trial_ns:
                _, ids, sp = build_introspection_prompt(tokenizer, tn)
                id_list.append(ids)
                starts.append(sp)
            # use concept 0 vector * 4
            v0 = (4.0 * vec_pack["vectors"][0]).to(dtype=torch.bfloat16)
            vectors_b = v0.unsqueeze(0).expand(len(trial_ns), -1).contiguous()

            def _greedy_batch(ids_l, vecs, sps, n_new=10):
                torch.manual_seed(42)
                device = next(model.parameters()).device
                input_ids, attn, pad_lens = pad_batch(tokenizer, ids_l, device)
                starts_p = torch.tensor(
                    [pl + s for pl, s in zip(pad_lens, sps)],
                    device=device,
                    dtype=torch.long,
                )
                with add_to_residual_batched(
                    model, layer, vecs.to(device), starts_p
                ):
                    with torch.no_grad():
                        out = model.generate(
                            input_ids=input_ids,
                            attention_mask=attn,
                            max_new_tokens=n_new,
                            do_sample=False,
                            use_cache=True,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                gens = []
                for i, ids in enumerate(ids_l):
                    plen = ids.shape[-1]
                    row = out[i, pad_lens[i] :]
                    gens.append(row[plen:].detach().cpu())
                return gens

            batched_gens = _greedy_batch(id_list, vectors_b, starts, 10)
            single_gens = []
            for i in range(len(trial_ns)):
                g = _greedy_batch(
                    [id_list[i]], vectors_b[i : i + 1], [starts[i]], 10
                )[0]
                single_gens.append(g)
            for i in range(len(trial_ns)):
                assert torch.equal(batched_gens[i], single_gens[i]), (
                    f"SANITY5 mismatch row {i}"
                )
            _log("SANITY5 batched vs single: PASS")
            summary["sanity"]["batched_hook"] = True

            # ========== Pilot ==========
            pilot_path = raw_dir / "pilot_log.txt"
            pilot_lines = []
            _log("PILOT: 8 inject α=4 + 8 control")
            pilot_t0 = time.perf_counter()
            # 8 inject: concepts 0..7, n=1, alpha=4
            inj_ids, inj_starts, inj_vecs = [], [], []
            for ci in range(8):
                _, ids, sp = build_introspection_prompt(tokenizer, 1)
                inj_ids.append(ids)
                inj_starts.append(sp)
                inj_vecs.append(4.0 * vec_pack["vectors"][ci])
            inj_vecs_t = torch.stack(inj_vecs).to(dtype=torch.bfloat16)
            inj_resp, _, _ = _generate_batch(
                model,
                tokenizer,
                inj_ids,
                inj_vecs_t,
                inj_starts,
                layer,
                max_new,
                temperature,
                seed_base,
            )
            for ci, resp in enumerate(inj_resp):
                line = f"[PILOT inject α=4 concept={concepts[ci]!r} n=1]\n{resp}\n"
                pilot_lines.append(line)
                _log(line)

            ctrl_ids, ctrl_starts = [], []
            for i in range(8):
                _, ids, sp = build_introspection_prompt(tokenizer, (i % 6) + 1)
                ctrl_ids.append(ids)
                ctrl_starts.append(sp)
            ctrl_resp, _, _ = _generate_batch(
                model,
                tokenizer,
                ctrl_ids,
                None,
                ctrl_starts,
                layer,
                max_new,
                temperature,
                seed_base + 1,
            )
            for i, resp in enumerate(ctrl_resp):
                line = f"[PILOT control n={(i % 6) + 1}]\n{resp}\n"
                pilot_lines.append(line)
                _log(line)
            pilot_dt = time.perf_counter() - pilot_t0
            pilot_path.write_text("\n".join(pilot_lines))

            # Project runtime: 16 pilot gens in pilot_dt; main has 1300 trials
            n_main = 600 + 300 + 200 + 150 + 50
            # pilot used 2 batches of 8; scale by batches of batch_size
            n_batches_main = int(np.ceil(n_main / batch_size))
            # crude: time per trial from pilot
            time_per_trial = pilot_dt / 16.0
            projected = time_per_trial * n_main + vec_time
            # also batch-aware estimate
            projected_batch = (pilot_dt / 2.0) * n_batches_main + vec_time
            projected_hours = max(projected, projected_batch) / 3600.0
            _log(
                f"PILOT projected GPU time ≈ {projected_hours:.3f} h "
                f"(limit {cfg['projected_gpu_limit_hours']} h); "
                f"pilot_dt={pilot_dt:.1f}s vec_time={vec_time:.1f}s"
            )
            summary["pilot"] = {
                "seconds": pilot_dt,
                "projected_hours": projected_hours,
            }
            if projected_hours > float(cfg["projected_gpu_limit_hours"]):
                msg = (
                    f"Projected GPU time {projected_hours:.3f} h exceeds "
                    f"{cfg['projected_gpu_limit_hours']} h; stopping before main run."
                )
                _log(msg)
                summary["stopped_before_main"] = True
                summary["stop_reason"] = msg
                (results_path / "generation_summary.json").write_text(
                    json.dumps(summary, indent=2, default=str)
                )
                if volume_commit:
                    volume_commit()
                return summary

            if skip_main:
                (results_path / "generation_summary.json").write_text(
                    json.dumps(summary, indent=2, default=str)
                )
                return summary

            # ========== Main generation ==========
            trials_path = raw_dir / "trials.jsonl"
            if trials_path.exists():
                trials_path.unlink()

            concept_index = {c: i for i, c in enumerate(concepts)}
            vectors = vec_pack["vectors"]  # float32 CPU [50,d]
            norms = vec_pack["norms"]

            # Build all trial specs
            specs: List[Dict[str, Any]] = []

            # inject
            for ci, concept in enumerate(concepts):
                for alpha in alphas:
                    for n in range(1, int(cfg["inject_trials_per_concept"]) + 1):
                        specs.append(
                            {
                                "condition": "inject",
                                "concept": concept,
                                "concept_index": ci,
                                "alpha": alpha,
                                "n": n,
                                "question_idx": None,
                            }
                        )
            # control: n=1..6 repeated 50×
            for rep in range(int(cfg["control_repeats"])):
                for n in cfg["control_trial_ns"]:
                    specs.append(
                        {
                            "condition": "control",
                            "concept": None,
                            "concept_index": rep,  # for ordering only
                            "alpha": None,
                            "n": int(n),
                            "question_idx": None,
                        }
                    )
            # random
            for ci, concept in enumerate(concepts):
                for alpha in alphas:
                    for n in cfg["random_trial_ns"]:
                        specs.append(
                            {
                                "condition": "random",
                                "concept": concept,
                                "concept_index": ci,
                                "alpha": float(alpha),
                                "n": int(n),
                                "question_idx": None,
                            }
                        )
            # yesbias_steered α=4
            for ci, concept in enumerate(concepts):
                for n in cfg["yesbias_steered_trial_ns"]:
                    n = int(n)
                    qidx = (ci + n) % 10
                    specs.append(
                        {
                            "condition": "yesbias_steered",
                            "concept": concept,
                            "concept_index": ci,
                            "alpha": 4.0,
                            "n": n,
                            "question_idx": qidx,
                        }
                    )
            # yesbias_unsteered: 10 questions × 5
            for qidx in range(10):
                for rep in range(int(cfg["yesbias_unsteered_repeats"])):
                    specs.append(
                        {
                            "condition": "yesbias_unsteered",
                            "concept": None,
                            "concept_index": None,
                            "alpha": None,
                            "n": 1,
                            "question_idx": qidx,
                            "rep": rep,
                        }
                    )

            assert len([s for s in specs if s["condition"] == "inject"]) == 600
            assert len([s for s in specs if s["condition"] == "control"]) == 300
            assert len([s for s in specs if s["condition"] == "random"]) == 200
            assert len([s for s in specs if s["condition"] == "yesbias_steered"]) == 150
            assert len([s for s in specs if s["condition"] == "yesbias_unsteered"]) == 50
            _log(f"Main run: {len(specs)} trials, batch_size={batch_size}")

            def _vector_for_spec(spec: Dict[str, Any]) -> Optional[torch.Tensor]:
                cond = spec["condition"]
                if cond == "inject":
                    ci = int(spec["concept_index"])
                    return float(spec["alpha"]) * vectors[ci]
                if cond == "random":
                    ci = int(spec["concept_index"])
                    alpha = float(spec["alpha"])
                    target = float(alpha * norms[ci])
                    return _make_random_vector(
                        d_model, seed=ci, target_norm=target, device="cpu"
                    ).float()
                if cond == "yesbias_steered":
                    ci = int(spec["concept_index"])
                    return 4.0 * vectors[ci]
                return None

            def _run_condition_batches(cond_specs: List[Dict[str, Any]], steered: bool):
                nonlocal batch_index, oom_batch
                idx = 0
                while idx < len(cond_specs):
                    chunk = cond_specs[idx : idx + oom_batch]
                    try:
                        id_list, starts, metas = [], [], []
                        for spec in chunk:
                            n = int(spec["n"])
                            if spec["condition"] in (
                                "yesbias_steered",
                                "yesbias_unsteered",
                            ):
                                q = YESBIAS_QUESTIONS[int(spec["question_idx"])]
                                text, ids, sp = build_yesbias_prompt(tokenizer, n, q)
                            else:
                                text, ids, sp = build_introspection_prompt(
                                    tokenizer, n
                                )
                            id_list.append(ids)
                            starts.append(sp)
                            metas.append(
                                {"formatted_prompt": text, "start_pos": sp, **spec}
                            )

                        seed = seed_base + batch_index
                        if steered:
                            stacked = [
                                _vector_for_spec(s).float() for s in chunk  # type: ignore
                            ]
                            vectors_t = torch.stack(stacked).to(dtype=torch.bfloat16)
                        else:
                            vectors_t = None

                        resps, _, _ = _generate_batch(
                            model,
                            tokenizer,
                            id_list,
                            vectors_t,
                            starts,
                            layer,
                            max_new,
                            temperature,
                            seed,
                        )
                        with trials_path.open("a") as f:
                            for j, spec in enumerate(chunk):
                                ci = spec.get("concept_index")
                                vnorm = None
                                if spec["condition"] in (
                                    "inject",
                                    "yesbias_steered",
                                    "random",
                                ):
                                    vnorm = float(
                                        float(spec["alpha"]) * norms[int(ci)]
                                    )
                                row = {
                                    "condition": spec["condition"],
                                    "concept": spec.get("concept"),
                                    "alpha": spec.get("alpha"),
                                    "n": spec["n"],
                                    "question_idx": spec.get("question_idx"),
                                    "batch_index": batch_index,
                                    "seed": seed,
                                    "formatted_prompt": metas[j]["formatted_prompt"],
                                    "start_pos": metas[j]["start_pos"],
                                    "vector_norm": vnorm,
                                    "response": resps[j],
                                    "judge_detection_raw": None,
                                    "judge_detection": None,
                                    "judge_ident_raw": None,
                                    "judge_ident": None,
                                    "yesbias_label": None,
                                }
                                f.write(json.dumps(row) + "\n")
                        _log(
                            f"batch {batch_index}: {len(chunk)} "
                            f"{chunk[0]['condition']} trials"
                        )
                        batch_index += 1
                        idx += len(chunk)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        if oom_batch <= 16:
                            raise
                        oom_batch = 16
                        _log(f"OOM: lowering batch_size to {oom_batch}")

            batch_index = 0
            oom_batch = batch_size
            by_cond = {
                c: [s for s in specs if s["condition"] == c]
                for c in (
                    "inject",
                    "control",
                    "random",
                    "yesbias_steered",
                    "yesbias_unsteered",
                )
            }
            _run_condition_batches(by_cond["inject"], steered=True)
            _run_condition_batches(by_cond["control"], steered=False)
            _run_condition_batches(by_cond["random"], steered=True)
            _run_condition_batches(by_cond["yesbias_steered"], steered=True)
            _run_condition_batches(by_cond["yesbias_unsteered"], steered=False)

            # Verify counts
            n_lines = sum(1 for _ in trials_path.open())
            _log(f"Wrote {n_lines} trials to {trials_path}")
            assert n_lines == 1300, f"expected 1300 trials, got {n_lines}"

            summary["n_trials"] = n_lines
            summary["batch_size_used"] = oom_batch
            summary["n_batches"] = batch_index
            (results_path / "generation_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            return summary

        except Exception as e:
            _log(f"ERROR: {e}")
            _log(traceback.format_exc())
            summary["errors"].append(str(e))
            (results_path / "generation_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            raise


if __name__ == "__main__":
    run_detection(
        model_dir=os.environ["SFO_MODEL_DIR"],
        revision_sha=os.environ["SFO_REVISION_SHA"],
        config_path=os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_001.yaml")),
        results_dir=os.environ.get(
            "SFO_RESULTS", str(REPO_ROOT / "results/WO-001")
        ),
        ledger_path=os.environ.get(
            "SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")
        ),
        submodule_root=os.environ.get(
            "SFO_SUBMODULE",
            str(REPO_ROOT / "third_party/introspection-mechanisms"),
        ),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )
