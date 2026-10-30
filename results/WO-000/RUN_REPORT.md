# WO-000 RUN_REPORT

## Summary
Repo skeleton, Modal image, Gemma-3-27B-it download, residual-stream hooks, and GPU smoke test completed. All 10 smoke checks PASS. `pytest tests/` passes (5/5).

## Commands run
```bash
modal secret list
modal volume create sfo-models
git submodule add https://github.com/safety-research/introspection-mechanisms.git third_party/introspection-mechanisms
git -C third_party/introspection-mechanisms checkout 5d5d9b485e293dc766c3a056cfe50283a7e435b6
modal run modal_app.py --action freeze
modal run modal_app.py --action download
modal run modal_app.py --action manifest
pytest tests/
modal run modal_app.py --action smoke   # successful run after two failed attempts (see Errors)
git add … && git commit && git push origin main
```

## Runtime
| Step | Approx wall time |
|------|------------------|
| Image build + `pip freeze` | ~4 min (first build ~4 min; later reuse) |
| Model download (CPU) | ~3.9 min |
| GemmaScope manifest (CPU) | ~22 s |
| GPU smoke (successful) | ~104 s billable GPU |
| Earlier failed GPU attempts | ~67 s + ~90 s (format_chat / UnboundLocalError; see Errors) |

Successful smoke window (ledger): `2026-10-30T13:38:51Z` → `2026-10-30T13:40:35Z` (**103.993 s ≈ 1.73 GPU-minutes**).

## Git hash
- Smoke-time / ledger `git_hash`: `15ada9730a9a41966fc137018a98d3950f53b2ac`
- Submodule `third_party/introspection-mechanisms`: `5d5d9b485e293dc766c3a056cfe50283a7e435b6`

## Environment
- Python (Modal): 3.11.12
- torch: 2.14.1+cu126 (CUDA 12.6 build from `https://download.pytorch.org/whl/cu126`)
- transformers: **5.22.1** — latest stable on PyPI at WO time; Gemma 3 CausalLM via `AutoModelForCausalLM` has been supported since 4.53.0; 5.22.1 is the newest release that satisfies the pin requirement
- cuda: 12.6
- gpu_name: NVIDIA A100-SXM4-80GB
- No bitsandbytes / quantization libraries installed (`requirements.lock` has none)

## Modal resources
- Volume: `sfo-models` (created)
- HF secret **name**: `hf-token` (only HF-token candidate from `modal secret list`; env key `HF_TOKEN`)
- GPU function: `gpu="A100-80GB"`, `cpu=8`, `memory=65536` MiB (64 GiB), timeout 45 min
- Model path on volume: `/models/gemma-3-27b-it`
- Model revision SHA: `005ad3404e59d6023443cb575daa05336842228a` → `configs/model_revision.yaml`

## Sanity-check outputs
All checks PASS (see `results/WO-000/metrics.json` and `results/WO-000/raw/smoke_log.txt`).

| Check | Result | Notes |
|-------|--------|-------|
| load | PASS | n_layers=62, d_model=5376, dtype=bfloat16, peak_mem≈51.10 GiB |
| tokenization | PASS | 5 prompts; single BOS at pos 0 each |
| sanity_forward | PASS | "Paris" in top-5 |
| determinism | PASS | bit-identical logits |
| zero_hook | PASS | bit-identical vs no hook @ layer 37 |
| nonzero_hook | PASS | max \|Δlogit\| ≈ 63.22 (> 1.0); r≈95229.7 |
| hook_removal | PASS | bit-identical to baseline after exit |
| capture | PASS | layers 37 & 45 → [1, 16, 5376], finite |
| generation | PASS | unhooked + hooked texts in raw log |
| throughput | PASS | fwd≈0.628 s; gen≈71.34 tok/s; hooked gen≈71.72 tok/s |

### Generation texts (verbatim excerpt from smoke log)
- Unhooked: haiku ending `Earth drinks, sighs content.`
- Hooked: whitespace-dominated continuation (strong steering)

## Budget ledger
`results/budget_ledger.csv` — one row for successful `smoke_test`:
- seconds=103.993, est_usd=0.149691 (formula with n_cores=8, mem_gib=64)
- Total recorded successful GPU time ≪ 0.5 h
- Failed attempts wrote ledger only on remote `/tmp` (not pulled); estimated extra GPU ≪ 5 min combined

## Reference submodule license
`safety-research/introspection-mechanisms` at the pinned commit has **no LICENSE file** (confirmed via GitHub contents API and local checkout). Status: **LICENSE absent**.

## Deviations and why
1. **numpy==2.4.1** (not latest 2.5.x): numpy 2.5+ requires Python ≥3.12; WO requires Python 3.11.
2. **scipy==1.17.0** (not 1.18.1): 1.18.1 requires Python ≥3.12.
3. **pandas==2.3.3**: latest line with broad py3.11 wheels; 3.x also has cp311 but 2.3.3 is the conservative pin used.
4. **torch==2.14.1+cu126** (not 2.15 from PyPI): WO requires a **CUDA 12.x** build; PyPI torch 2.15 pulls CUDA 13 toolkit deps.
5. **`format_chat` return handling**: `apply_chat_template(..., return_tensors="pt")` under transformers 5.22.1 returned a `tokenizers.Encoding`; code still calls that API and normalizes to a 2-D `torch.LongTensor` before the BOS assert.
6. **Local Modal entrypoint**: avoids importing PyYAML locally (parses `revision_sha` with a line scan) because the Modal CLI env lacks PyYAML.
7. **Smoke return payload**: GPU function returns text artifacts + pass/fail summary only (not the live metrics object) so the local machine need not have torch installed to deserialize the Modal response.
8. **`get_layers` assert 62** kept for production loads; hooks resolve layers via `_resolve_layers` so CPU unit tests can use tiny configs.

## Errors (encountered and fixed)
1. Modal image build: `__pycache__` modified during `add_local_dir(copy=True)` → switched to runtime mounts + ignore `**/__pycache__/**`.
2. First smoke: `RuntimeError: Could not infer dtype of tokenizers.Encoding` in `format_chat` → normalize Encoding/BatchEncoding to tensor.
3. Second smoke: `UnboundLocalError: capture_residual` caused by a shadowed local import in the load probe fallback → removed local import.
4. Local deserialize of Modal return failed without torch → return text-only artifacts.

## HF secret
- **Name used:** `hf-token`
- Value never logged or committed.

## GPU minutes consumed
- Successful smoke (ledger): **≈ 1.73 minutes**
- Plus two failed smoke attempts (load progressed): **≈ 2–3 minutes** estimated
- **Total GPU ≪ 0.5 h** (well under budget)
