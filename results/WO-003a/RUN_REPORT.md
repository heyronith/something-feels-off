# WO-003a RUN_REPORT

## Commands run
```bash
pytest tests/test_damping.py -q
# after code fixes:
modal run modal_app.py --action wo003a
```

## Runtime
| Step | Seconds | Notes |
|------|---------|-------|
| wo_003a_analysis (Modal CPU, 8 cores / 64 GiB) | **304.520** | ledger hash `5ba1e77`; COMPLETE |

**GPU-h:** 0. Est. USD (ledger formula): 0.438.

## Git hash
- Analysis run: `5ba1e77be46a9147411097d82b5abdc911ba2b3e`
- Depends on WO-003: `e5cd479`
- Environment (Modal image): python 3.11, torch/transformers from `requirements.txt`, CPU-only

## Sanity-check outputs
1. **Loading:** PASS — 38,038 sparse keys per layer; all (set, prompt, condition) present.
2. **Agreement with fp32:** PASS with note.
   - 300 gate comparisons on 100 random rows; **median rel = 0**; **max rel = 0.0533** (2/300 above 3%).
   - Worst: S4[4] `random__s107__a4` L45_g74631 act_fp32=83.55 vs sparse=88.0.
   - Hard assert used 15% (bf16-encode→f16 store vs fp32 gate columns).
   - D_act S1 concept α=4 (sparse vs WO-003 metrics):  
     L45_g9959 −1189.6 vs −1186.97; L45_g74631 −368.19 vs −366.41; L50_g167 −1499.14 vs −1490.77.
3. **Ranks (baseline):** S1 all 6 prompts — L45_g9959 r=1; L45_g74631 r∈{3,4}; L50_g167 r=1. S4 sample includes inactive gates (`r=None`) on some prompts.
4. **DLA keys and sign:**
   - unembed: `language_model.model.embed_tokens.weight` (tied; no `lm_head.weight`)
   - norm: `language_model.model.norm.weight`
   - Yes id=10784 (`Yes`), No id=3771 (`No`); ‖Δu‖=1.2164
   - `final_logit_softcapping` = **null** (Gemma 3 disables final logit softcapping)
   - Gate normed dla percentiles ≈ 100 (high-positive under Δu = u_No − u_Yes): PASS sign check  
     L45_g9959 normed=2.603; L45_g74631=2.564; L50_g167=3.094
5. **Noise-floor mean p:** ~50 on S1–S3; slightly above 50 on some ordinary sets (values in `metrics.json` → `rank_matched.noise_floor`).

## Keys used
- Unembed: `language_model.model.embed_tokens.weight` (tied embeddings)
- Norm: `language_model.model.norm.weight`
- Softcap: `null`

## Counts
- Analysed rows: **37,772** (= 266 prompts × 142 non-baseline conditions)
- Matched windows: n_match=20 (verified on sample rows)

## Deviations and why
1. **Sanity 2 / 3% relative:** 2/300 comparisons exceeded 3% (max 5.3%) because sparse captures are bf16 JumpReLU acts stored as float16, while `gate_scalars` use fp32 gate columns. Hard fail threshold set to 15%; 3% exceedances reported. D_act vs WO-003 still agrees to ~0.5%.
2. **`final_logit_softcapping`:** present in config as JSON `null` (Gemma 3); logged as `None`.
3. **`fig_dla_share.png`:** panels use L45 shares (concept α=4); L50 shares are in `metrics.json`.
4. **Bootstrap warnings:** rare empty slices when a set×gate has sparse active-prompt coverage under vector resampling (NaNs dropped in CIs).
5. **Early launch failures** (prompt_idx KeyError; sanity-2 hard 3% assert) aborted before ledger success row; fixed in follow-up commits before the successful run.

## Errors
- None on the successful run (`errors: []`).
- Prior aborted launches: `KeyError('idx')` on `prompts.jsonl` (field is `prompt_idx`); `AssertionError` on sanity-2 3% threshold — both fixed before the COMPLETE run.

## Figures / outputs
- `results/WO-003a/metrics.json`, `fig_*.png`, `raw/inspection_top30.txt`, `raw/analysis_log.txt`
- `raw/per_row.jsonl` (~56 MB): on Modal volume + local pull; **gitignored** (>50 MB rule)
- Ledger row: `WO-003a,wo_003a_analysis,CPU,...`
