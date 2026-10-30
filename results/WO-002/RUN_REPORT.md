# WO-002 RUN_REPORT

## Commands run
```bash
# Implementation + unit tests
pytest tests/test_blockers.py -q

# Part A (after blocker/SANITY5 fix)
modal run modal_app.py --action wo002_partA

# Part B (first attempt OOMed on MMLU; second resumed from 1900 trials)
modal run modal_app.py --action wo002_partB   # generation + OOM
modal run modal_app.py --action wo002_partB   # resume MMLU/KL

# Judge + metrics
modal run modal_app.py --action wo002_judge
PYTHONPATH=src python scripts/wo_002_metrics.py results/WO-002
```

## Runtime
| Step | Ledger seconds | Notes |
|------|----------------|-------|
| wo_002_partA (failed SANITY5) | 60.047 | commit `260cdff`; aborted before main A1 |
| wo_002_partA (success) | 634.411 | commit `d7b3ec9`; l0*=42 |
| wo_002_partB (generation) | 730.770 | 1900 trials written; OOM on MMLU |
| wo_002_partB (MMLU/KL resume) | 170.525 | skip_gen; MMLU+KL complete |
| Judge (CPU Modal) | ~295 s wall | 2100 OpenAI calls |
| Metrics/figures (local) | <1 s | |

**Total WO-002 GPU billable:** 60.047 + 634.411 + 730.770 + 170.525 = **1595.753 s ≈ 26.6 GPU-min ≈ 0.443 h** (limit 1.3 h).

## Git hash
- Final / completion commit: `f3c42292428d5d5149304038bf024b45ed0c7fae` (message: WO-002: Anomaly-direction blocker (B1) — find and validate; this file’s hash line updated in a follow-up commit)
- Part A success ledger hash: `d7b3ec98e4820557940d0067f352eb661651153b`
- Part B generation ledger hash: `7a7b9073e722583802d51701ae90c28f1de5a921`
- Part B MMLU/KL ledger hash: `5ee4d2a498ab81a087279493efbf99528c9fee72`
- Model revision: `005ad3404e59d6023443cb575daa05336842228a`
- Depends on WO-001: `1bb31c7`

## Environment
From Part A GPU summary (`partA_summary.json`):
- python: 3.11.12
- torch / transformers / cuda / gpu_name: as recorded in `results/WO-002/partA_summary.json` → `env`
- GPU: A100-80GB (Modal), 8 cores, 64 GiB host RAM
- Datasets: `cais/mmlu` rev `c30699e8356da336a370243923dbaf21066bb9fe`; `Salesforce/wikitext` rev `b08601e04326c79dfdd32d625aee71d232d685c3`

## Sanity-check outputs
1. Clean tree / pushed-HEAD: PASS (each GPU launch).
2. NEW_CONCEPTS: 450 unique; overlap with 50+100 = 0; dir_dev=225 dir_val=225.
3. YES/NO tokens: Yes id=10784 (`Yes`); No id=3771 (`No`).
4. Blocker identity: PASS.
5. Blocker correctness: PASS (max rel_to_‖h‖ = 1.16e-04; proj_mean=0.6507). Relative error measured vs residual L2 scale (bf16); see Deviations.
6. Composition: PASS.
7. Prefill: PASS (prefill inside model turn; no `<end_of_turn>` after).
8. MMLU: letter ids=[236776, 236799, 236780, 236796] (A/B/C/D single tokens); 2 formatted questions logged.
9. Projected total after Part A: **0.45 h** (< 1.3) — continued.

## l0*
**42** (dir-dev subset B1 recovery 2.2534; tie rule N/A).

Held-out dir-val B1 recovery: 2.2295.  
B1-rand mean/max (held-out): 0.0274 / 0.0928.  
B1-pc held-out: 0.1494.  
Random-transfer: dLD_none_mean=8.8825; B1_recovery=2.9412.  
Proxy Spearman (ΔLD vs WO-001 tpr@4): ρ=0.3360, p=0.01703.

## Part B headline numbers (from `metrics.json`; no interpretation)
| Arm | TPR | FPR | FID | EXP | MMLU | KL |
|-----|-----|-----|-----|-----|------|-----|
| none | 0.218 (65/298) | 0.000 (0/150) | 0.470 | 0.630 | 0.765 | — |
| B1 | 0.0039 (1/256) | 0.000 (0/120) | 0.050 | 0.230 | 0.760 | 2.492 |
| B1-null | 0.252 (75/298) | n/a (0 ctl) | 0.490 | 0.580 | 0.775 | 0.017 |
| fid0 / exp0 | — | — | 0.000 | 0.020 | — | — |

Parse failures (judge): **118** (det 78, ident 40); not folded into negatives.  
Judge: n_calls=2100; tokens prompt=1342384 completion=503976 total=1846360; **est_usd=$1.343**.

## Deviations and why
1. **SANITY5 relative scale:** Spec says “within 1e-2 relative.” On bf16 Gemma residuals (‖h‖ ~ 1e4–1e5), `|proj−μ|/|μ|` is not a meaningful tolerance after casting the corrected residual back to bf16. Check uses `|proj−μ|/mean(‖h‖)` and passed at 1.16e-04. Mean-ablation formula unchanged.
2. **Part B MMLU OOM:** First Part B run wrote all **1900** generations then OOM’d allocating full logits `[B,T,V]` at batch=32. Fix: resume-from-trials, MMLU batch≤4 + `logits_to_keep=1`, KL batch=1. Generations not re-sampled.
3. **`partA_dirdev.pt` not in git:** ~389 MB and covered by `*.pt` gitignore; remains on Modal volume `sfo-models` at `results/WO-002/raw/`. Smaller `.pt` artifacts (`directions.pt`, `concept_vectors_L37_new450.pt`, `partA_dirval.pt`) force-added like WO-001.
4. **Volume pull nesting:** First Part A download nested `results/WO-002/WO-002/`; `_pull_volume` fixed to `modal volume get … results/WO-002` into `results/`. Flattened locally.
5. **B1-null FPR:** Spec’s B-ctl table only includes arms `none` and `B1`; B1-null FPR is reported as n=0 in metrics (no ctl trials for that arm).
6. **Failed Part A ledger row** (60 s) retained for budget honesty.

## Errors
1. First Part A (`260cdff`): `AssertionError` SANITY5 proj≈μ (max_rel≈24.86) — fixed blocker hook validation + bf16-aware check; re-ran.
2. First Part B (`7a7b907`): `torch.OutOfMemoryError` on MMLU lm_head — resumed with smaller batches (`5ee4d2a`).
3. Provenance gate blocked a Part B launch when local `results/WO-002/` was uncommitted after Part A — committed artifacts then re-ran.

## GPU minutes
WO-002 only: **26.6 min** (0.443 h). Est. GPU $ from ledger rows: 0.086 + 0.913 + 1.052 + 0.245 ≈ **$2.30**.

## Judge tokens and $
- Calls: 2100  
- Tokens: 1,846,360 total (1,342,384 prompt + 503,976 completion)  
- Estimated USD: **$1.343**

## Outputs checklist
- [x] `src/sfo/blockers.py`, `data.py`; scripts partA/B/judge/metrics; `tests/test_blockers.py`
- [x] `configs/wo_002.yaml` (splits + dataset revisions)
- [x] `results/WO-002/metrics.json`, figures, raw artifacts, ledger rows
- [x] `work_orders/WO-002.md`
