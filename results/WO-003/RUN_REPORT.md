# WO-003 RUN_REPORT

## Commands run
```bash
# Unit tests (CPU)
pytest tests/test_transcoder.py -q

# Transcoder download (Modal CPU → volume sfo-models)
modal run modal_app.py --action wo003_download

# Main GPU run (after importlib dataclass fix + short-prompt fix)
modal run modal_app.py --action wo003_run

# Metrics + figures (local CPU)
python scripts/wo_003_metrics.py results/WO-003
```

## Runtime
| Step | Ledger seconds | Notes |
|------|----------------|-------|
| wo_003_download (CPU) | 129.0 | GemmaScope-2 L45+L50 affine params (~22.8 GB) |
| wo_003_run (failed import) | 15.221 | commit `de9b0fa`; `@dataclass` + importlib |
| wo_003_run (success) | 822.757 | commit `b467e6b`; COMPLETE 38038/38038 |
| wo_003_metrics (CPU) | ~21 | bootstrap + figures |

**Total WO-003 GPU billable:** 15.221 + 822.757 = **837.978 s ≈ 14.0 GPU-min ≈ 0.233 h** (limit 1.0 h).  
Successful main wall clock on GPU: `total_seconds=822.6` (`main_seconds=720.4`).

## Git hash
- GPU success ledger hash: `b467e6bf19bf53cc7cadc964ce725257a7f1cc29`
- Model revision: `005ad3404e59d6023443cb575daa05336842228a`
- Transcoder revision (`google/gemma-scope-2-27b-it`): `5c58dd4cddd52cef653059d85e12a86bf6222a28`
- Depends on WO-002b tip lineage: `7a79c16` (directions `results/WO-002b/raw/directions_B1p.pt`, `d_hat[44]` / `d_hat[49]`)

## Environment
From `run_summary.json` → `env`:
- python: 3.11.12
- torch: 2.14.1+cu126
- transformers: 5.22.1
- cuda: 12.6
- gpu_name: NVIDIA A100 80GB PCIe (Modal), 8 cores, 64 GiB host RAM
- batch_size: **24**

## Sanity-check outputs
1. **Provenance:** PASS (clean pushed HEAD at launch).
2. **Transcoder config:** PASS. Both layers: `hf_hook_point_in=model.layers.{ℓ}.pre_feedforward_layernorm.output`, `width=262144`, `affine_connection=true`, folders `*_affine`. Keys present: `w_enc,b_enc,threshold,w_dec,b_dec,affine_skip_connection`.
3. **Capture point:** PASS (LN output bit-identical to MLP input at L45 and L50).
4. **Early halt:** PASS (L45/L50 captures identical to full forward on 2 prompts).
5. **Identity / noise / BOS:** PASS.
   - Within-batch α=0 rows bit-identical.
   - Cross-batch max |Δpre_g| (α=0): see `run_summary.json` → `sanity.s5_batch_and_steer.cross_batch_noise` (e.g. B1 vs B24 L45 pre_g maxabs ≈ 24.78; B23 vs B24 all 0).
   - α=4 changes L45 capture; nothing added at BOS.
6. **Transcoder validity:** PASS.
   - L45: mean L0=149.8; best VE=`W+A`=0.7976 (W+A.T=-1.10; W+noskip=-1.63).
   - L50: mean L0=112.2; best VE=`W+A`=0.8046 (W+A.T=-12.82; W+noskip=0.309).
7. **Gate-column consistency:** PASS (fp32 vs bf16 within 2%).
8. **Readout position:** last prompt token; top-5 next logged (e.g. S1[0] top1=`No` 64.5). Full table in `raw/sanity_log.txt`.
9. **Tokenization / steering start:** logged with `⟦STEER→⟧` (S1/S3 Macar rule; S2/S2v/S4–S7 pos 1).
10. **Gate presence (S1):** unsteered + α=4 for Dust/Satellites/Trumpets — see sanity_log (e.g. baseline L45_F9959 pre≈2860; Dust α=4 dpre≈−1524).
11. **Projected time:** after 5.3% prompts, projected main=0.33 h / total=0.35 h (< 1.0 h) — continued.

## Counts
- Prompts: 266 (S1:6, S2:6, S2v:4, S3:10, S4:60, S5:60, S6:60, S7:60)
- Conditions: 143 → **38,038** gate/residual/sparse rows per layer; failed=0
- Logits LD rows: **12,298** (= 86 full-forward prompts × 143)
- JBB held-out indices recorded in config/summary; **never forwarded**

## Batch size
**24** (6 chunks: 24×5 + 23; baseline in first batch; noise-floor in middle + last).

## Deviations and why
1. **Short prompts (T < 20):** 24 JBB Goals tokenize to fewer than 20 chat tokens. Spec asks for a last-20 mean; implementation uses last `min(20, T)` (already in `forward_chunk`). Prompts were not dropped. Listed in `run_summary.json` → `short_prompts_lt_tail`.
2. **importlib + `@dataclass`:** first GPU launch failed because Modal loads the script via `exec_module` without registering `sys.modules`. Fixed by `sys.modules[spec.name] = mod` before exec.
3. **Failed short GPU attempt** (15 s): aborted at prompt build before model load; ledger row retained.
4. **Sparse `.pt` files** (~104 MB / ~90 MB) live under `results/WO-003/raw/` and are gitignored via `*.pt`; present on Modal volume and local pull.
5. **Metrics CI field names** emitted as `{m,lo95,hi95,lo9944,hi9944}` per WO schema; extra diagnostic keys (e.g. `direct_frac_ci`, `corr_dpre_dLD_ci`, `log_abs_ratio_pctl`) retained alongside.

## Errors
- Launch `de9b0fa`: `AttributeError: 'NoneType' object has no attribute '__dict__'` on `@dataclass` under importlib — fixed; re-run succeeded.
- Main run: **0** failed rows; `failed_rows.jsonl` empty.

## GPU minutes
Successful run: **822.757 s ≈ 13.71 GPU-min**.  
Including failed launch: **837.978 s ≈ 13.97 GPU-min ≈ 0.233 A100-h**.
