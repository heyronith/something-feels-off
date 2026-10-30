# something-feels-off

Activation-steering research on Gemma-3-27B-it.

## Setup

- Python 3.11
- Modal (GPU + volume for weights)
- Hugging Face token as Modal secret `hf-token` (Gemma 3 license accepted)

```bash
pip install -r requirements.txt
# Install torch CUDA 12.6 build via:
#   pip install torch==2.14.1+cu126 --extra-index-url https://download.pytorch.org/whl/cu126
```

Weights live on the Modal volume `sfo-models`, never in this repo.

## Work orders

See `work_orders/`. WO-000 sets up the Modal environment, downloads the model, and runs the smoke test.

## Package

`src/sfo` — model loading, residual-stream hooks, budget ledger, provenance.
