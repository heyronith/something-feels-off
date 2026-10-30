"""WO-000 GPU smoke test. Invoked by Modal; can also be imported."""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# Allow importing sfo when run from repo root or Modal mount
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sfo.budget import append_ledger_row  # noqa: E402
from sfo.hooks import add_to_residual, capture_residual  # noqa: E402
from sfo.model import (  # noqa: E402
    EXPECTED_D_MODEL,
    EXPECTED_N_LAYERS,
    assert_hidden_width,
    first_param_dtype,
    format_chat,
    get_d_model,
    get_layers,
    load_model,
    load_model_revision,
    load_tokenizer,
)
from sfo.provenance import collect_env, git_hash, set_seeds  # noqa: E402


def _log(msg: str) -> None:
    print(msg, flush=True)


def _pass_fail(name: str, ok: bool, details: Dict[str, Any]) -> Dict[str, Any]:
    status = "PASS" if ok else "FAIL"
    _log(f"[{status}] {name}: {json.dumps(details, default=str)}")
    return {"pass": bool(ok), "details": details}


def _peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)


def _user_messages(text: str) -> List[dict]:
    return [{"role": "user", "content": text}]


def _decode_top5(tokenizer, logits_1d: torch.Tensor) -> List[Dict[str, Any]]:
    vals, idxs = torch.topk(logits_1d.float(), k=5)
    out = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        out.append(
            {
                "token_id": int(i),
                "token": tokenizer.decode([i]),
                "logit": float(v),
            }
        )
    return out


def _paris_in_top5(top5: List[Dict[str, Any]]) -> bool:
    for item in top5:
        tok = item["token"].strip().lower()
        if tok == "paris" or tok.startswith("paris"):
            return True
    return False


def _make_unit_vector(d_model: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    v = torch.randn(d_model, generator=g, dtype=torch.float32)
    v = v / v.norm()
    return v.to(dtype=torch.bfloat16)


def run_smoke(
    *,
    model_dir: str,
    revision_sha: str,
    config_path: str,
    results_dir: str,
    ledger_path: str,
    git_hash_str: str,
    gpu_type: str = "A100-80GB",
    n_cores: float = 8.0,
    mem_gib: float = 64.0,
    start_utc: Optional[datetime] = None,
) -> Dict[str, Any]:
    start_utc = start_utc or datetime.now(timezone.utc)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    raw_dir = results_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg.get("seed", 0))
    set_seeds(seed)

    checks: Dict[str, Any] = {}
    metrics: Dict[str, Any] = {
        "checks": checks,
        "model": {},
        "throughput": {},
        "env": {},
        "git_hash": git_hash_str,
    }

    model = None
    tokenizer = None
    baseline_logits = None
    nonzero_vector = None
    r_norm = None
    prompt1_ids = None

    try:
        # ---- 1. Load ----
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        tokenizer = load_tokenizer(model_dir, revision=revision_sha)
        model = load_model(model_dir, revision=revision_sha, device=0)
        layers = get_layers(model)
        d_model = get_d_model(model)
        dtype = first_param_dtype(model)
        peak = _peak_mem_gb()

        load_ok = (
            len(layers) == EXPECTED_N_LAYERS
            and d_model == EXPECTED_D_MODEL
            and dtype == torch.bfloat16
        )
        checks["load"] = _pass_fail(
            "load",
            load_ok,
            {
                "n_layers": len(layers),
                "d_model": d_model,
                "dtype": str(dtype).replace("torch.", ""),
                "peak_mem_gb_after_load": peak,
            },
        )
        metrics["model"] = {
            "id": cfg.get("model_id", "google/gemma-3-27b-it"),
            "revision_sha": revision_sha,
            "n_layers": len(layers),
            "d_model": d_model,
            "dtype": str(dtype).replace("torch.", ""),
            "peak_mem_gb_after_load": peak,
        }

        # Assert hidden width matches config d_model (after load check recorded)
        probe_ids = format_chat(tokenizer, _user_messages("hi")).to(model.device)
        with torch.no_grad():
            probe_out = model(input_ids=probe_ids, output_hidden_states=True)
        hs = probe_out.hidden_states[-1] if probe_out.hidden_states is not None else None
        if hs is not None:
            assert_hidden_width(model, hs)
        else:
            # Fallback: capture residual at layer 0
            with capture_residual(model, [0]) as caps:
                with torch.no_grad():
                    model(input_ids=probe_ids)
            assert_hidden_width(model, caps[0])

        # ---- 2. Tokenization ----
        tok_details = {"prompts": []}
        tok_ok = True
        for p in cfg["prompts"]:
            messages = _user_messages(p)
            # formatted string
            formatted = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            ids = format_chat(tokenizer, messages)
            token_ids = ids[0].tolist()
            decoded_tokens = [tokenizer.decode([t]) for t in token_ids]
            bos_id = tokenizer.bos_token_id
            bos_count = sum(1 for t in token_ids if t == bos_id)
            single_bos = bos_count == 1 and token_ids[0] == bos_id
            if not single_bos:
                tok_ok = False
            entry = {
                "prompt": p,
                "formatted": formatted,
                "token_ids": token_ids,
                "decoded_tokens": decoded_tokens,
                "bos_count": bos_count,
                "single_bos": single_bos,
            }
            tok_details["prompts"].append(entry)
            _log(f"TOKENIZATION prompt={p!r}")
            _log(f"  formatted={formatted!r}")
            _log(f"  token_ids={token_ids}")
            _log(f"  decoded_tokens={decoded_tokens}")
            _log(f"  bos_count={bos_count}")
        checks["tokenization"] = _pass_fail("tokenization", tok_ok, tok_details)
        prompt1_ids = format_chat(tokenizer, _user_messages(cfg["prompts"][0])).to(
            model.device
        )

        # ---- 3. Sanity forward ----
        sanity_ids = format_chat(
            tokenizer, _user_messages(cfg["sanity_prompt"])
        ).to(model.device)
        with torch.no_grad():
            logits = model(input_ids=sanity_ids).logits
        last = logits[0, -1]
        top5 = _decode_top5(tokenizer, last)
        sanity_ok = _paris_in_top5(top5)
        checks["sanity_forward"] = _pass_fail(
            "sanity_forward",
            sanity_ok,
            {"top5": top5, "prompt": cfg["sanity_prompt"]},
        )
        _log(f"SANITY top-5: {top5}")

        # ---- 4. Determinism ----
        with torch.no_grad():
            a = model(input_ids=sanity_ids).logits
            b = model(input_ids=sanity_ids).logits
        det_ok = torch.equal(a, b)
        checks["determinism"] = _pass_fail(
            "determinism",
            det_ok,
            {"bit_identical": det_ok},
        )
        baseline_logits = a.detach()

        # ---- 5. Zero hook ----
        zero = torch.zeros(d_model, dtype=torch.bfloat16, device=model.device)
        with add_to_residual(model, cfg["hook_layer"], zero, positions="all"):
            with torch.no_grad():
                z_logits = model(input_ids=sanity_ids).logits
        zero_ok = torch.equal(baseline_logits, z_logits)
        checks["zero_hook"] = _pass_fail(
            "zero_hook",
            zero_ok,
            {"bit_identical": zero_ok, "layer": cfg["hook_layer"]},
        )

        # ---- 6. Non-zero hook ----
        # mean L2 norm of layer-37 residual over prompt 1 positions
        with capture_residual(model, [cfg["hook_layer"]]) as caps:
            with torch.no_grad():
                model(input_ids=prompt1_ids)
        resid = caps[cfg["hook_layer"]]  # [1, seq, d] float32 CPU
        # L2 over last dim, mean over positions
        norms = resid[0].float().norm(dim=-1)  # [seq]
        r_norm = float(norms.mean().item())
        unit = _make_unit_vector(d_model, seed=0)
        scale = float(cfg["scale_factor"]) * r_norm
        nonzero_vector = (unit.float() * scale).to(dtype=torch.bfloat16, device=model.device)

        with torch.no_grad():
            before_logits = model(input_ids=sanity_ids).logits
        before_top5 = _decode_top5(tokenizer, before_logits[0, -1])

        with add_to_residual(model, cfg["hook_layer"], nonzero_vector, positions="all"):
            with torch.no_grad():
                after_logits = model(input_ids=sanity_ids).logits
        after_top5 = _decode_top5(tokenizer, after_logits[0, -1])
        delta = (after_logits[0, -1].float() - before_logits[0, -1].float()).abs()
        max_delta = float(delta.max().item())
        nonzero_ok = max_delta > 1.0
        checks["nonzero_hook"] = _pass_fail(
            "nonzero_hook",
            nonzero_ok,
            {
                "r": r_norm,
                "scale": scale,
                "max_abs_delta_logit": max_delta,
                "top5_before": before_top5,
                "top5_after": after_top5,
            },
        )
        _log(f"NONZERO r={r_norm} scale={scale} max|Δlogit|={max_delta}")
        _log(f"  top5_before={before_top5}")
        _log(f"  top5_after={after_top5}")

        # ---- 7. Hook removal ----
        with torch.no_grad():
            after_ctx = model(input_ids=sanity_ids).logits
        rem_ok = torch.equal(baseline_logits, after_ctx)
        checks["hook_removal"] = _pass_fail(
            "hook_removal",
            rem_ok,
            {"bit_identical_to_baseline": rem_ok},
        )

        # ---- 8. Capture ----
        with capture_residual(model, cfg["capture_layers"]) as caps2:
            with torch.no_grad():
                model(input_ids=prompt1_ids)
        cap_ok = True
        cap_details: Dict[str, Any] = {}
        seq_len = prompt1_ids.shape[1]
        for li in cfg["capture_layers"]:
            t = caps2[li]
            shape_ok = list(t.shape) == [1, seq_len, d_model]
            finite = bool(torch.isfinite(t).all().item())
            if not (shape_ok and finite):
                cap_ok = False
            cap_details[str(li)] = {
                "shape": list(t.shape),
                "finite": finite,
            }
        checks["capture"] = _pass_fail("capture", cap_ok, cap_details)

        # ---- 9. Generation ----
        prompt2 = cfg["prompts"][1]
        gen_ids = format_chat(tokenizer, _user_messages(prompt2)).to(model.device)
        n_new = int(cfg.get("generation_new_tokens", 30))
        with torch.no_grad():
            unhooked = model.generate(
                input_ids=gen_ids,
                max_new_tokens=n_new,
                do_sample=False,
                use_cache=True,
            )
        unhooked_text = tokenizer.decode(unhooked[0], skip_special_tokens=False)
        with add_to_residual(model, cfg["hook_layer"], nonzero_vector, positions="all"):
            with torch.no_grad():
                hooked = model.generate(
                    input_ids=gen_ids,
                    max_new_tokens=n_new,
                    do_sample=False,
                    use_cache=True,
                )
        hooked_text = tokenizer.decode(hooked[0], skip_special_tokens=False)
        _log("GENERATION unhooked:")
        _log(unhooked_text)
        _log("GENERATION hooked:")
        _log(hooked_text)
        checks["generation"] = _pass_fail(
            "generation",
            True,
            {
                "unhooked_text": unhooked_text,
                "hooked_text": hooked_text,
                "new_tokens": n_new,
            },
        )

        # ---- 10. Throughput ----
        assert tokenizer.padding_side == "left", "padding_side must be left for generation"
        thr = cfg["throughput"]
        # (a) forward batch 8 x 256
        b, tlen = int(thr["fwd_batch"]), int(thr["fwd_seq"])
        # left-pad random-ish token ids using pad token
        pad_id = tokenizer.pad_token_id
        # Use repeated prompt1 content truncated/padded
        base = prompt1_ids[0].tolist()
        rows = []
        for i in range(b):
            seq = (base * ((tlen // len(base)) + 1))[:tlen]
            rows.append(seq)
        fwd_ids = torch.tensor(rows, device=model.device)
        attn = (fwd_ids != pad_id).long()
        # warm-up
        with torch.no_grad():
            model(input_ids=fwd_ids, attention_mask=attn)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times = []
        for _ in range(int(thr["fwd_runs"])):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                model(input_ids=fwd_ids, attention_mask=attn)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        fwd_mean = float(np.mean(times))

        # (b) greedy gen batch 8, prompt ~64, 48 new tokens
        gb = int(thr["gen_batch"])
        gp = int(thr["gen_prompt_tokens"])
        gn = int(thr["gen_new_tokens"])
        # Build left-padded batch from chat-formatted prompts
        texts = [cfg["prompts"][i % len(cfg["prompts"])] for i in range(gb)]
        encoded = []
        for text in texts:
            ids = format_chat(tokenizer, _user_messages(text))[0].tolist()
            if len(ids) > gp:
                ids = ids[-gp:]
            encoded.append(ids)
        max_len = max(len(x) for x in encoded)
        # left pad
        pad_rows = []
        attn_rows = []
        for ids in encoded:
            pad_len = max_len - len(ids)
            pad_rows.append([pad_id] * pad_len + ids)
            attn_rows.append([0] * pad_len + [1] * len(ids))
        gen_batch_ids = torch.tensor(pad_rows, device=model.device)
        gen_attn = torch.tensor(attn_rows, device=model.device)

        def _timed_gen(hook_vec=None):
            def _one():
                if hook_vec is None:
                    with torch.no_grad():
                        model.generate(
                            input_ids=gen_batch_ids,
                            attention_mask=gen_attn,
                            max_new_tokens=gn,
                            do_sample=False,
                            use_cache=True,
                        )
                else:
                    with add_to_residual(
                        model, cfg["hook_layer"], hook_vec, positions="all"
                    ):
                        with torch.no_grad():
                            model.generate(
                                input_ids=gen_batch_ids,
                                attention_mask=gen_attn,
                                max_new_tokens=gn,
                                do_sample=False,
                                use_cache=True,
                            )

            # warm-up
            _one()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            run_times = []
            for _ in range(int(thr["gen_runs"])):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _one()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                run_times.append(time.perf_counter() - t0)
            mean_s = float(np.mean(run_times))
            # tokens/sec = (batch * new_tokens) / seconds
            tok_per_sec = (gb * gn) / mean_s if mean_s > 0 else 0.0
            return mean_s, tok_per_sec

        _, gen_tps = _timed_gen(None)
        _, gen_hooked_tps = _timed_gen(nonzero_vector)

        thr_details = {
            "fwd_b8_t256_sec": fwd_mean,
            "gen_b8_t48_tok_per_sec": gen_tps,
            "gen_b8_t48_hooked_tok_per_sec": gen_hooked_tps,
            "padding_side": tokenizer.padding_side,
        }
        metrics["throughput"] = {
            "fwd_b8_t256_sec": fwd_mean,
            "gen_b8_t48_tok_per_sec": gen_tps,
            "gen_b8_t48_hooked_tok_per_sec": gen_hooked_tps,
        }
        checks["throughput"] = _pass_fail("throughput", True, thr_details)
        _log(f"THROUGHPUT {thr_details}")

    except Exception as e:
        _log(f"ERROR during smoke: {e}")
        _log(traceback.format_exc())
        # Mark incomplete checks as failed if missing; do not re-raise so
        # metrics/ledger/artifacts are always returned to the caller.
        for name in [
            "load",
            "tokenization",
            "sanity_forward",
            "determinism",
            "zero_hook",
            "nonzero_hook",
            "hook_removal",
            "capture",
            "generation",
            "throughput",
        ]:
            if name not in checks:
                checks[name] = _pass_fail(name, False, {"error": str(e)})
    finally:
        metrics["env"] = collect_env()
        end_utc = datetime.now(timezone.utc)
        append_ledger_row(
            ledger_path,
            wo_id="WO-000",
            function="smoke_test",
            gpu_type=gpu_type,
            start_utc=start_utc,
            end_utc=end_utc,
            n_cores=n_cores,
            mem_gib=mem_gib,
            git_hash=git_hash_str,
        )
        metrics_path = results_path / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
        _log(f"Wrote {metrics_path}")

    return metrics


if __name__ == "__main__":
    model_dir = os.environ["SFO_MODEL_DIR"]
    revision = os.environ["SFO_REVISION_SHA"]
    config = os.environ.get("SFO_CONFIG", str(REPO_ROOT / "configs" / "wo_000.yaml"))
    results = os.environ.get("SFO_RESULTS", str(REPO_ROOT / "results" / "WO-000"))
    ledger = os.environ.get(
        "SFO_LEDGER", str(REPO_ROOT / "results" / "budget_ledger.csv")
    )
    gh = os.environ.get("SFO_GIT_HASH") or git_hash(str(REPO_ROOT))
    run_smoke(
        model_dir=model_dir,
        revision_sha=revision,
        config_path=config,
        results_dir=results,
        ledger_path=ledger,
        git_hash_str=gh,
    )
