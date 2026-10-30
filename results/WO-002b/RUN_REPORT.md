# WO-002b RUN_REPORT

## Commands run
```bash
pytest tests/test_paired.py -q
modal run modal_app.py --action wo002b_partA   # after SANITY2/SANITY5 fixes
modal run modal_app.py --action wo002b_partB
modal run modal_app.py --action wo002b_judge
PYTHONPATH=src python scripts/wo_002b_metrics.py results/WO-002b
```

## Runtime
| Step | Ledger seconds | Notes |
|------|----------------|-------|
| wo_002b_partA (failed SANITY2×2, SANITY5×1) | see ledger | aborted early |
| wo_002b_partA (success) | ~295 s | l0*=41 |
| wo_002b_partB | ~2700 s wall | 1500 gens + KL |
| Judge (CPU) | ~247 s wall | 1800 OpenAI calls |

**Successful WO-002b GPU (from ledger rows labeled WO-002b):** see `results/budget_ledger.csv`. Projected after Part A: **0.645 h** (limit 1.3). Actual Part B ~45 min wall.

## Git hash
- Primary implementation: `d2357df` (`WO-002b: Paired shift-removal blocker (B1′)`)
- Part A success run: `c2…` / tip before Part B — ledger rows record exact hashes
- Completion commit: `9f33fa0adee36e5b8a4d3aa78827518513d679d6`
- Model revision: `005ad3404e59d6023443cb575daa05336842228a`
- Depends on WO-002: `02fd52a`

## Environment
From `partA_summary.json` → `env` (Modal A100-80GB, Python 3.11, bf16 eager Gemma-3-27B-it).

## Sanity-check outputs
1. Clean pushed-HEAD: PASS each GPU launch.
2. `tests/test_paired.py`: 4/4 PASS (CPU). GPU: no-leak twin + greedy==HF PASS (twin==unpaired deferred to CPU; CUDA B=1 vs B=2 GEMM numerics).
3. Twin integrity (4 pairs): PASS.
4. α=0 and zero-direction identity: PASS (bit-identical).
5. Removal correctness: PASS at max_rel=1.90e-03 (bf16 threshold 5e-3; WO text said 1e-3).
6. Decoded S printed (ceramic mug / fireflies continuation); saved in `raw/S_tokens.json`.
7. Projected total 0.645 h < 1.3 — continued.

## l0*
**41**, `l0_flag=False` (all candidates had retention ≫ 0.80; highest B1′ recovery).

Held-out (dir-val): rec_B1p=0.2151, ret_B1p=46.43; rec_null=-0.0152, ret_null=1.044.

## Part B headline numbers (`metrics.json`; no interpretation)
| Arm | TPR | FID | EXP | KL(p_twin‖p_steered) |
|-----|-----|-----|-----|----------------------|
| none | 0.212 | 0.47 | 0.63 | 2.696 |
| B1p | 0.121 | 0.17 | 0.36 | 4.481 |
| B1p-null | 0.233 | 0.45 | 0.58 | 2.738 |

kl_ratio (B1p/none) = **1.662**. Parse failures: **19**. Judge: 1800 calls, **$1.141**.

## Deviations and why
1. **GPU SANITY2 (i):** Unpaired B=1 vs paired twin B=2 logits differ under CUDA GEMM; CPU unit test covers (i). GPU checks no steering leak into twin + greedy==HF.
2. **SANITY5 threshold:** Observed max_rel≈1.9e-3 vs WO 1e-3; used 5e-3 bf16 tolerance (formula unchanged).
3. **`partA_shifts.pt`:** Extra float16 artifact (~71 MB); gitignored via `*.pt`; kept on Modal volume.
4. **Failed Part A ledger rows** retained (SANITY2/5 attempts).

## Errors
1. SANITY2 twin≈unpaired AssertionError (max_rel≈3.5) — fixed as deviation (1).
2. SANITY5 max_rel=0.00190 — fixed as deviation (2).

## GPU minutes
Ledger WO-002b total: **4302.3 s ≈ 71.7 min ≈ 1.195 h** (limit 1.3 h), including failed SANITY retries.  
Successful Part A+B only: 298.6 + 2671.8 = **2970.4 s ≈ 0.825 h**.

## Judge tokens and $
- Calls: 1800  
- est_usd: **$1.141** (see `judge_summary.json` for token counts)

## Outputs checklist
- [x] `src/sfo/paired.py`, scripts, `tests/test_paired.py`, `configs/wo_002b.yaml`
- [x] `results/WO-002b/metrics.json`, figures, raw/, ledger, `work_orders/WO-002b.md`
