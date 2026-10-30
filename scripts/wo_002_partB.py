"""WO-002 Part B: validate B1 at l0* with generation + MMLU/KL."""

from __future__ import annotations

import json
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

from sfo.blockers import mean_ablate_from_l0  # noqa: E402
from sfo.data import (  # noqa: E402
    format_mmlu_user,
    load_mmlu_validation,
    load_wikitext_chunks,
    save_jsonl,
)
from sfo.hooks import add_to_residual_batched  # noqa: E402
from sfo.ld import concept_match_string, yes_no_token_ids  # noqa: E402
from sfo.ledger_ctx import gpu_ledger  # noqa: E402
from sfo.model import format_chat, get_d_model, load_model, load_tokenizer  # noqa: E402
from sfo.prompts import (  # noqa: E402
    build_introspection_prompt,
    decode_new_tokens,
    format_chat_text,
    introspection_messages,
)
from sfo.provenance import collect_env  # noqa: E402
from sfo.vectors import pad_batch  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / (v.norm() + 1e-12)


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
    from sfo.vectors import compute_start_pos

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


def _generate(
    model,
    tokenizer,
    id_list: List[torch.Tensor],
    starts: List[int],
    steer_vecs: Optional[torch.Tensor],
    steer_layer: int,
    blocker_dirs: Optional[Dict[int, torch.Tensor]],
    blocker_mus: Optional[Dict[int, float]],
    l0: Optional[int],
    max_new: int,
    temperature: float,
    seed: int,
    do_sample: bool,
) -> List[str]:
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
    st = torch.tensor([pl + s for pl, s in zip(pad_lens, starts)], device=device)
    prompt_lens = [t.shape[-1] if t.ndim == 2 else t.numel() for t in id_list]

    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attn,
        max_new_tokens=max_new,
        do_sample=do_sample,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    if do_sample:
        kwargs["temperature"] = temperature

    def _run():
        with torch.no_grad():
            return model.generate(**kwargs)

    # nested context managers
    from contextlib import nullcontext

    steer_ctx = (
        add_to_residual_batched(
            model, steer_layer, steer_vecs.to(device=device, dtype=torch.bfloat16), st
        )
        if steer_vecs is not None
        else nullcontext()
    )
    blk_ctx = (
        mean_ablate_from_l0(model, l0, blocker_dirs, blocker_mus, st, max_layer=61)
        if blocker_dirs is not None and l0 is not None
        else nullcontext()
    )
    with steer_ctx:
        with blk_ctx:
            out = _run()

    resps = []
    for i in range(out.shape[0]):
        row = out[i, pad_lens[i] :]
        resps.append(decode_new_tokens(tokenizer, row, prompt_lens[i]))
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
        "errors": [],
    }

    with gpu_ledger(
        ledger_path,
        wo_id="WO-002",
        function="wo_002_partB",
        gpu_type=gpu_type,
        n_cores=n_cores,
        mem_gib=mem_gib,
        git_hash=git_hash_str,
        volume_commit=volume_commit,
    ):
        try:
            partA = json.loads((results_path / "partA_summary.json").read_text())
            l0_star = int(partA["partA"]["l0_star"])
            directions = torch.load(raw / "directions.pt", map_location="cpu", weights_only=False)
            d_hat = directions["d_hat"]
            mus = directions["mus"]
            layers = directions["layers"]

            wo001 = json.loads(Path(wo001_snapshot_path).read_text())
            concepts_50 = list(wo001["concepts"])
            pack50 = torch.load(wo001_vectors_path, map_location="cpu", weights_only=False)
            vec50 = {c: pack50["vectors"][i] for i, c in enumerate(pack50["concepts"])}

            tokenizer = load_tokenizer(model_dir, revision=revision_sha)
            model = load_model(model_dir, revision=revision_sha, device=0)
            d_model = get_d_model(model)
            yes_id, no_id = yes_no_token_ids(tokenizer)
            summary["env"] = collect_env()
            summary["l0_star"] = l0_star

            alpha = float(cfg["alpha"])
            layer_steer = int(cfg["layer_steer"])
            batch_size = int(cfg["batch_size"])
            temperature = float(cfg["temperature"])
            max_new = int(cfg["max_new_tokens"])
            seed_base = int(cfg["seed_base"])
            prefill = cfg["fid_prefill"]

            # Sanity 7: print one B-fid prompt
            ftext, fids, fsp = build_fid_prompt(tokenizer, 1, prefill)
            _log(f"SANITY7 B-fid prompt:\n{ftext}")
            assert prefill in ftext
            assert "<end_of_turn>" not in ftext[ftext.find(prefill) :]
            _log("SANITY7: prefill inside model turn, no end_of_turn after — PASS")

            # B1-null: one fixed random direction seed 0
            g = torch.Generator().manual_seed(int(cfg["b1_null_seed"]))
            null_dirs = {}
            null_mus = {}
            # μ from partA unsteered if available; else 0
            dirdev = torch.load(raw / "partA_dirdev.pt", map_location="cpu", weights_only=False)
            unsteered = dirdev["unsteered"]
            for l in layers:
                v = torch.randn(d_model, generator=g)
                null_dirs[l] = unit(v)
                dots = []
                for n in unsteered:
                    h = unsteered[n]["full_from_start"][l]
                    dots.append((h * null_dirs[l].unsqueeze(0)).sum(-1))
                null_mus[l] = float(torch.cat(dots).mean().item())

            arms_b1 = {
                "none": (None, None, None),
                "B1": (d_hat, mus, l0_star),
                "B1-null": (null_dirs, null_mus, l0_star),
            }

            trials_path = raw / "partB_trials.jsonl"
            expected_trials = 900 + 300 + 300 + 50 + 300 + 50
            skip_gen = False
            if trials_path.exists():
                n_existing = sum(1 for _ in trials_path.open())
                if n_existing == expected_trials:
                    _log(
                        f"Resuming: found {n_existing} trials on disk; "
                        "skipping generation, continuing with MMLU/KL"
                    )
                    skip_gen = True
                else:
                    _log(f"Removing incomplete trials.jsonl ({n_existing} lines)")
                    trials_path.unlink()
            batch_index = 0

            def write_rows(rows: List[dict]):
                with trials_path.open("a") as f:
                    for r in rows:
                        f.write(json.dumps(r) + "\n")

            def run_batches(specs, steered: bool, sample: bool, max_tok: int, start_mode: str):
                """start_mode: 'trial' uses start_pos from introspection; 'one' uses 1; 'fid' uses fid builder."""
                nonlocal batch_index
                i = 0
                while i < len(specs):
                    chunk = specs[i : i + batch_size]
                    id_list, starts, metas = [], [], []
                    vecs = []
                    for spec in chunk:
                        arm = spec["arm"]
                        if spec["condition"] == "B-fid" or spec["condition"] == "B-fid0":
                            text, ids, sp = build_fid_prompt(
                                tokenizer, int(spec["n"]), prefill
                            )
                        elif spec["condition"] in ("B-exp", "B-exp0"):
                            text, ids = build_exp_prompt(tokenizer, cfg["exp_user"])
                            sp = 1
                        else:
                            text, ids, sp = build_introspection_prompt(
                                tokenizer, int(spec["n"])
                            )
                        if start_mode == "one":
                            sp = 1
                        id_list.append(ids)
                        starts.append(sp)
                        metas.append({"formatted_prompt": text, "start_pos": sp, **spec})
                        if steered and spec.get("concept"):
                            vecs.append(alpha * vec50[spec["concept"]])
                        else:
                            vecs.append(None)

                    seed = seed_base + batch_index
                    # arm is same within chunk ideally — group by arm
                    # We build homogeneous arm batches outside
                    arm = chunk[0]["arm"]
                    bdirs, bmus, bl0 = arms_b1[arm]
                    if steered:
                        vec_t = torch.stack([v.float() for v in vecs]).to(torch.bfloat16)
                    else:
                        vec_t = None

                    resps = _generate(
                        model,
                        tokenizer,
                        id_list,
                        starts,
                        vec_t,
                        layer_steer,
                        bdirs,
                        bmus,
                        bl0,
                        max_tok,
                        temperature,
                        seed,
                        do_sample=sample,
                    )
                    rows = []
                    for j, spec in enumerate(chunk):
                        match = None
                        if spec["condition"] in ("B-fid", "B-fid0", "B-exp", "B-exp0"):
                            if spec.get("concept"):
                                match = concept_match_string(spec["concept"], resps[j])
                        rows.append(
                            {
                                "condition": spec["condition"],
                                "arm": arm,
                                "concept": spec.get("concept"),
                                "n": spec.get("n"),
                                "seed": seed,
                                "batch_index": batch_index,
                                "start_pos": metas[j]["start_pos"],
                                "prompt": metas[j]["formatted_prompt"],
                                "response": resps[j],
                                "match": match,
                                "judge_detection_raw": None,
                                "judge_detection": None,
                                "judge_ident_raw": None,
                                "judge_ident": None,
                            }
                        )
                    write_rows(rows)
                    _log(f"batch {batch_index}: {len(chunk)} {chunk[0]['condition']}/{arm}")
                    batch_index += 1
                    i += len(chunk)

            # ---- Build specs / generate ----
            if not skip_gen:
                # B-inj: 50×6×3
                for arm in ("none", "B1", "B1-null"):
                    specs = []
                    for c in concepts_50:
                        for n in range(1, 7):
                            specs.append(
                                {
                                    "condition": "B-inj",
                                    "arm": arm,
                                    "concept": c,
                                    "n": n,
                                }
                            )
                    run_batches(
                        specs, steered=True, sample=True, max_tok=max_new, start_mode="trial"
                    )

                # B-ctl: 25 repeats × n=1..6 × {none,B1} = 300
                for arm in ("none", "B1"):
                    specs = []
                    for rep in range(25):
                        for n in range(1, 7):
                            specs.append(
                                {
                                    "condition": "B-ctl",
                                    "arm": arm,
                                    "concept": None,
                                    "n": n,
                                }
                            )
                    run_batches(
                        specs, steered=False, sample=True, max_tok=max_new, start_mode="trial"
                    )

                # B-fid: 50×2×3 greedy 6 tokens
                for arm in ("none", "B1", "B1-null"):
                    specs = []
                    for c in concepts_50:
                        for n in (1, 2):
                            specs.append(
                                {
                                    "condition": "B-fid",
                                    "arm": arm,
                                    "concept": c,
                                    "n": n,
                                }
                            )
                    run_batches(
                        specs,
                        steered=True,
                        sample=False,
                        max_tok=int(cfg["fid_new_tokens"]),
                        start_mode="fid",
                    )

                # B-fid0: 50 unsteered
                specs = [
                    {"condition": "B-fid0", "arm": "none", "concept": c, "n": 1}
                    for c in concepts_50
                ]
                run_batches(
                    specs,
                    steered=False,
                    sample=False,
                    max_tok=int(cfg["fid_new_tokens"]),
                    start_mode="fid",
                )

                # B-exp: 50×2 samples × 3 arms
                for arm in ("none", "B1", "B1-null"):
                    specs = []
                    for c in concepts_50:
                        for s in (0, 1):
                            specs.append(
                                {
                                    "condition": "B-exp",
                                    "arm": arm,
                                    "concept": c,
                                    "n": s,
                                }
                            )
                    run_batches(
                        specs,
                        steered=True,
                        sample=True,
                        max_tok=int(cfg["exp_new_tokens"]),
                        start_mode="one",
                    )

                # B-exp0
                specs = [
                    {"condition": "B-exp0", "arm": "none", "concept": c, "n": 0}
                    for c in concepts_50
                ]
                run_batches(
                    specs,
                    steered=False,
                    sample=True,
                    max_tok=int(cfg["exp_new_tokens"]),
                    start_mode="one",
                )

            n_lines = sum(1 for _ in trials_path.open())
            expected = expected_trials
            _log(f"Wrote {n_lines} trials (expected {expected})")
            assert n_lines == expected, f"got {n_lines} != {expected}"

            # ---- MMLU ----
            # Cap batch: full logits [B,T,V] with V≈262k OOMs at batch=32 on long prompts.
            from contextlib import nullcontext

            cap_batch = min(4, batch_size)
            torch.cuda.empty_cache()
            _log("Loading MMLU...")
            mmlu_items, mmlu_rev = load_mmlu_validation(n=int(cfg["mmlu_n"]), seed=0)
            cfg_out = {
                "mmlu_revision": mmlu_rev,
                "wikitext_revision": None,
            }
            # Sanity 8
            letters = ["A", "B", "C", "D"]
            letter_ids = []
            for L in letters:
                ids = tokenizer.encode(L, add_special_tokens=False)
                assert len(ids) == 1, f"{L} not single token: {ids}"
                letter_ids.append(ids[0])
            for item in mmlu_items[:2]:
                msg = format_mmlu_user(item)
                _log(f"SANITY8 MMLU:\n{msg}\ngold={letters[item['answer']]}")
            _log(f"SANITY8 letter ids={letter_ids}")

            save_jsonl(raw / "mmlu_items.jsonl", mmlu_items)

            mmlu_acc = {}
            for arm_name, (bdirs, bmus, bl0) in arms_b1.items():
                correct = 0
                for start in range(0, len(mmlu_items), cap_batch):
                    chunk = mmlu_items[start : start + cap_batch]
                    id_list = []
                    for item in chunk:
                        messages = [{"role": "user", "content": format_mmlu_user(item)}]
                        ids = format_chat(tokenizer, messages)
                        id_list.append(ids)
                    device = next(model.parameters()).device
                    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
                    st = torch.tensor(
                        [pl + 1 for pl in pad_lens], device=device, dtype=torch.long
                    )
                    blk = (
                        mean_ablate_from_l0(model, bl0, bdirs, bmus, st, max_layer=61)
                        if bdirs is not None
                        else nullcontext()
                    )
                    with blk:
                        with torch.no_grad():
                            try:
                                out = model(
                                    input_ids=input_ids,
                                    attention_mask=attn,
                                    logits_to_keep=1,
                                )
                            except TypeError:
                                out = model(
                                    input_ids=input_ids, attention_mask=attn
                                )
                    logits = out.logits[:, -1, :]
                    for j, item in enumerate(chunk):
                        scores = [float(logits[j, tid]) for tid in letter_ids]
                        pred = int(max(range(4), key=lambda k: scores[k]))
                        if pred == int(item["answer"]):
                            correct += 1
                    del out, logits
                mmlu_acc[arm_name] = correct / len(mmlu_items)
                _log(f"MMLU {arm_name}: {mmlu_acc[arm_name]:.4f}")
                torch.cuda.empty_cache()

            # ---- Neutral KL ----
            _log("Loading WikiText chunks...")
            chunks, wt_rev = load_wikitext_chunks(
                n=int(cfg["wikitext_n"]),
                chunk_tokens=int(cfg["wikitext_chunk_tokens"]),
                tokenizer=tokenizer,
            )
            cfg_out["wikitext_revision"] = wt_rev
            save_jsonl(raw / "wikitext_chunks.jsonl", chunks)
            (raw / "dataset_revisions.json").write_text(json.dumps(cfg_out, indent=2))

            summary["dataset_revisions"] = cfg_out

            kl_means = {}
            bos = tokenizer.bos_token_id
            kl_batch = 1  # KL needs full [B,T,V] twice; keep tiny
            for arm_name, (bdirs, bmus, bl0) in [
                ("B1", arms_b1["B1"]),
                ("B1-null", arms_b1["B1-null"]),
            ]:
                kls = []
                for start in range(0, len(chunks), kl_batch):
                    chunk = chunks[start : start + kl_batch]
                    id_list = []
                    for ch in chunk:
                        ids = tokenizer(
                            ch["text"], add_special_tokens=False, return_tensors="pt"
                        )["input_ids"]
                        content = ids[0, : int(cfg["wikitext_chunk_tokens"])]
                        row = torch.cat([torch.tensor([bos]), content], dim=0)
                        id_list.append(row.unsqueeze(0))
                    device = next(model.parameters()).device
                    input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
                    st = torch.tensor(
                        [pl + 1 for pl in pad_lens], device=device, dtype=torch.long
                    )
                    with torch.no_grad():
                        logits_none = model(
                            input_ids=input_ids, attention_mask=attn
                        ).logits
                    with mean_ablate_from_l0(model, bl0, bdirs, bmus, st, max_layer=61):
                        with torch.no_grad():
                            logits_arm = model(
                                input_ids=input_ids, attention_mask=attn
                            ).logits
                    log_p = torch.log_softmax(logits_none.float(), dim=-1)
                    log_q = torch.log_softmax(logits_arm.float(), dim=-1)
                    p = log_p.exp()
                    kl_tok = (p * (log_p - log_q)).sum(dim=-1)  # [B,T]
                    for b in range(kl_tok.shape[0]):
                        pl = pad_lens[b]
                        vals = kl_tok[b, pl + 1 :]
                        mask = attn[b, pl + 1 :].bool()
                        if mask.any():
                            kls.append(float(vals[mask].mean().cpu()))
                    del logits_none, logits_arm, log_p, log_q, p, kl_tok
                kl_means[arm_name] = float(sum(kls) / len(kls)) if kls else float("nan")
                _log(f"KL {arm_name}: {kl_means[arm_name]:.6f}")
                torch.cuda.empty_cache()

            summary["mmlu_acc"] = mmlu_acc
            summary["kl_means"] = kl_means
            (results_path / "partB_summary.json").write_text(
                json.dumps(summary, indent=2, default=str)
            )
            if volume_commit:
                volume_commit()
            return summary

        except Exception as e:
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
        config_path=os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs/wo_002.yaml")),
        results_dir=os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results/WO-002")),
        ledger_path=os.environ.get(
            "SFO_LEDGER", str(REPO_ROOT / "results/budget_ledger.csv")
        ),
        wo001_vectors_path=str(REPO_ROOT / "results/WO-001/raw/concept_vectors_L37.pt"),
        wo001_snapshot_path=str(REPO_ROOT / "results/WO-001/raw/inputs_snapshot.json"),
        git_hash_str=os.environ.get("SFO_GIT_HASH", "unknown"),
    )
