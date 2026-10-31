# WO-004 RUN_REPORT

## Commands run
```bash
# Unit tests (CPU)
PYTHONPATH=src python -m pytest tests/test_feature_clamp.py -q

# Main GPU run (Modal A100-80GB → volume sfo-models)
modal run modal_app.py --action wo004

# Metrics + figures (local CPU)
python scripts/wo_004_metrics.py results/WO-004
```

## Runtime
| Step | Ledger seconds | Notes |
|------|----------------|-------|
| wo_004_run (SANITY2 fail) | 127.232 | commit `d3cf817`; cross-batch identity compare |
| wo_004_run (SANITY4 stop) | 210.500 | commit `3fcc740`; replication applied to non-S1 |
| wo_004_run (success) | 943.206 | commit `ef38de86`; COMPLETE 39646/39646 |
| wo_004_metrics (CPU) | ~12 | bootstrap + figures |

**Successful main wall clock:** `total_seconds=943.213` ≈ **15.7 GPU-min ≈ 0.262 h** (limit 1.0 h).  
**Total WO-004 GPU billable (all attempts):** 127.232 + 210.500 + 943.206 = **1280.938 s ≈ 21.3 GPU-min ≈ 0.356 h**.

## Git hash
- GPU success ledger hash: `ef38de86266060b07293d7f90271904efe66b3f7`
- Model revision: `005ad3404e59d6023443cb575daa05336842228a`
- Transcoder revision (`google/gemma-scope-2-27b-it`): `5c58dd4cddd52cef653059d85e12a86bf6222a28`
- Depends on: WO-003 `e5cd479`, WO-003a `7ab3446`

## Environment
From `run_summary.json` → `env`:
- python: 3.11.12
- torch: 2.14.1+cu126
- transformers: 5.22.1
- cuda: 12.6
- gpu_name: NVIDIA A100-SXM4-80GB (Modal), 8 cores, 64 GiB host RAM
- batch_size: **25**

## Sanity-check outputs
1. **Provenance:** PASS (clean pushed HEAD at launch).
2. **Identity (SANITY2):** PASS on 2 S1 prompts (within-batch compares for clampG / clampMLP / sham↔base).
3. **Delta check (SANITY3):** PASS (L45 relative error within 1e-2).
4. **Replication (SANITY4):** S1 base and concept steer LD vs WO-003 `logits_ld.jsonl`; **max |diff| = 0.0**.
5. **Group integrity (SANITY5):** asserted every batch (base first; groups unsplit).
6. **Readout (SANITY6):** top-5 logged for every S8 base and 2 prompts/other set; also steer/clampG/clampMLP/shamG for Dust on S1[0]. See `raw/sanity_log.txt`.
7. **S8 default-Yes base LD (SANITY7):**
   - S8[0] = -18.500000
   - S8[1] = -14.750000
   - S8[2] = -14.750000
   - S8[3] = -15.500000
   - S8[4] = -15.250000
   - S8[5] = -15.000000
8. **Controls (SANITY8):** M_dla / M_norm ids, a₀, dla, K, frac for first 3 S1 prompts — printed in `raw/sanity_log.txt` (e.g. S1[0] L45 K_dla=50 frac≈0.18; L50 K_dla=44 frac≈0.53).
9. **Projected time (SANITY9):** after 5/86 prompts, projected **0.46 h** (limit 1.00 h) — continued.

Gate DLA cache assert (normed): L45_g9959=2.603483, L45_g74631=2.564099, L50_g167=3.094100 (tol 0.01).

## Batch size
**25** (= 1 base + ≤3 concept groups × 8 conds, or 1 base + ≤8 random groups × 3 conds).  
Concept batches per prompt: 17; random batches: 3.

## Counts
- Prompts: **86** (S1:6, S2v:4, S3:10, S4:60, S8:6)
- Rows logged: **39646** = 86 × (1 + 50×8 + 20×3); matches `expected_n_rows`
- Decomposition rows: **12212** = 86 × (1 + 50 + 20) × 2 layers
- Controls: **86** lines in `controls.jsonl`

## Deviations and why
1. **SANITY2 identity compare:** first launch compared unsteered clampG/clampMLP/sham logits to a separate B=1 base forward; bit-identity failed (batch-size numerics). Spec requires bit-identical to the reference; fixed by comparing against the base row **in the same batch**. Thresholds unchanged.
2. **SANITY4 scope:** first successful-sanity launch stopped on S4 concept LD vs WO-003 (|diff|=1.25 > 1.0). Spec §sanity 4 is **S1 only**; check restricted to S1. Thresholds unchanged. S1 max |diff| = 0.
3. **Aborted GPU spends** retained in `results/budget_ledger.csv` (127 s + 211 s) before the successful 943 s run.
4. **`dla_cache.pt`** (~18 MB) under `results/WO-004/raw/` is gitignored via `*.pt`; present on Modal volume and local pull.
5. **Decomposition** taken from the same concept/random forwards as the main row log (base + steer slots), not extra dedicated forwards — same formulas; saves GPU time under the 1.0 h cap.
6. **Encode on GPU / decode on CPU** for full-width W+A decomposition (headroom next to 27B); feature clamp still uses fp32 sliced columns via `get_slice` on GPU.

## Errors
- Launch `d3cf817`: `SanityAbort: SANITY2 clampG unsteered vs base` — fixed (within-batch compare).
- Launch `3fcc740`: `RunStop: SANITY4: |LD-LD003|=1.25 > 1.0 for ('S4', 0, 'concept__Happiness__a4')` — fixed (S1-only replication).
- Successful run `ef38de86`: **0** errors; `status=COMPLETE`.

## GPU minutes
- Successful run: **≈15.7 min** (0.262 h).
- All WO-004 attempts combined: **≈21.3 min** (0.356 h).
