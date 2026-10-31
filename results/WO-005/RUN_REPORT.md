# WO-005 RUN_REPORT

## Commands run
```bash
# Unit tests (CPU)
python -m pytest tests/test_wo005.py -q

# Code push (SANITY1)
git push origin main   # ad36520 scaffolding; 8d87319 resume support

# GPU logit+gen (Modal A100-80GB) — first pass cancelled mid-gen by client disconnect
modal run modal_app.py --action wo005

# Resume remaining generations (detach; skip complete logit + 8 complete gen prompts)
modal run --detach modal_app.py --action wo005_resume

# Judge (Modal CPU + openai-key)
modal run --detach modal_app.py --action wo005_judge

# Metrics + figures (local CPU)
python scripts/wo_005_metrics.py results/WO-005
```

## Runtime
| Step | Ledger seconds | Est USD | Notes |
|------|----------------|---------|-------|
| wo_005_run | 3525.042 | 5.074087 | git `ad36520`; logit COMPLETE; gen partial (1221/1800); client cancel |
| wo_005_run_resume | 1866.511 | 2.686730 | git `8d87319`; skip logit; finish 4 remaining gen prompts → 1800 |
| wo_005_judge | 124.712 | 0.188352 | 1824 calls (incl. spot-check); parse_fail=0 |

**Successful final status:** `COMPLETE` with `n_logit_rows=23556`, `n_gens=1800`.  
**Resume wall:** `total_seconds=1867.5` ≈ 0.52 h (model reload + 4 gen prompts).  
**Billable GPU (both runs):** 3525.042 + 1866.511 = **5391.553 s ≈ 89.9 GPU-min ≈ 1.498 h**.  
**Judge:** ≈ **$0.19**.

## Git hash
- First GPU launch: `ad36520a89fd63aa73a1573453c99d13d29ed4bc`
- Resume + judge: `8d87319d52408c9ea5de8522271ff1c15c4c8028`
- Model revision: `005ad3404e59d6023443cb575daa05336842228a`
- BoolQ revision: `35b264d03638db9f4ce671b711558bf7ff0f80d5`
- Depends on: WO-004a `cd8cff8`

## Environment
From `run_summary.json` → `env`:
- python: 3.11.12
- torch: 2.14.1+cu126
- transformers: 5.22.1
- cuda: 12.6
- gpu_name: NVIDIA A100-SXM4-80GB (Modal), 8 cores, 64 GiB host RAM
- batch_size (logit): **25**
- gen_batch_size: **8**
- sampling: temperature=1.0, max_new_tokens=100, top_k=64, top_p=0.95, seed_base=1000

## Sanity-check outputs
1. **Provenance (SANITY1):** PASS (clean pushed HEAD at each launch).
2. **Replication (SANITY2):** P0F/P0I n=1..3 token ids match WO-004 S1/S8 prompts 0–2. Max |LD − WO-004| for base + concept α=4 = **0.0** (≤ 1.0).
3. **Anchors (SANITY3):** PASS; unique Trial anchor; late start = T−1 for each wording n=1. Logged in `raw/sanity_log.txt`.
4. **Top-5 (SANITY4):** base top-5 printed for every F/I (n=1), S3T, and C prompt (154 lines). Examples:
   - F[0] P1: No 60.75, no 46.75, Yes 39.25, …
   - I[0] P1: Yes 69.5, No 64.0, yes 53.75, …
   - S3T[0]: Yes 82.5, yes 62.5, …
   - Full list in `raw/sanity_log.txt`.
5. **Identity (SANITY5):** PASS — α=0 bit-identical to base in-batch; α=4 changed LD (base_LD=22.0000 → a4_LD=10.7500).
6. **Judge spot-check (SANITY6):** 12 base + 12 concept α=4 printed in `raw/judge_log.txt` before full judge. Spot claims mostly NO; one concept α=4 YES. Full judge: parse_fail=0.
7. **Time projection (SANITY7):**
   - Logit 5%: projected **~0.83 h** (limit 1.50 h) — continued.
   - Gen 5% (first pass): projected **~1.42 h** (limit 1.50 h) — continued.
   - Resume gen 5%: projected segment **~0.53 h** — continued.

## Counts
- Steered logit rows: **23,424** (= 122 × 192)
- Set C unsteered: **132**
- Total logit: **23,556**
- Generations: **1,800** (= 12 × 150), all judged
- Judge audit rows: **50** (`raw/judge_audit.jsonl`)
- Valid pairs (primary filter): **0** / 6 (see `metrics.json` → `validity`)

## Metrics / figures
- `metrics.json` sections: validity, logit, compression, gen, delta_claim, D, contingency, lopo, sanity, judge
- Figures: `fig_claims.png`, `fig_ld.png`, `fig_residual.png`, `fig_comprehension.png`
- D (concept α=4, pooled_all): m=0.1667, lo95=-0.0244, hi95=0.3934
- D (concept α=4, pooled_valid): null (no valid pairs)

## Deviations and why
1. **Client disconnect mid-gen:** First `modal run` (no `--detach`) was cancelled after logit COMPLETE and 1221/1800 gens (8 prompts × 150 + partial I-P3). Modal reported local client disconnect. Thresholds unchanged. Added `resume=True` path (`wo005_resume`) that keeps complete logit + complete gen prompts and regenerates incomplete/missing prompts only. Partial I-P3 (21 base rows) was dropped and that prompt regenerated fully.
2. **Combined GPU bill ≈ 1.498 h:** First-pass SANITY7 projected 1.42 h under the 1.5 h single-run cap; cancel + model reload on resume caused total billable GPU to approach 1.5 h. No threshold was loosened; gens were finished via resume rather than a full redo.
3. **Batched generation seeding:** Concept/random gens are batched (`gen_batch_size=8`) for time. Torch RNG is seeded once per batch (`seed_base+off+first_index`); per-row logged `seed` / `gen_seed` still follow the WO seed formulas. Responses are therefore not independent across a batch under distinct seeds.
4. **Sanity log duplication:** Modal Tee + `Logger` both append the same lines on resume (cosmetic).
5. **`*.pt` vectors** remain under WO-001 on volume/local; not re-copied into WO-005.

## Errors
- First launch: `RemoteError: Function call was cancelled by user or a failure` after logit done / gen in progress — recovered via resume.
- No SanityAbort / RunStop on thresholds.

## GPU minutes and judge $
- GPU billable: **≈ 89.9 min (1.498 h)** across two A100 runs.
- Judge: **1824 calls**, **≈ $0.188**, parse_failures=0.
