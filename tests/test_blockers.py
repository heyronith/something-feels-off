"""CPU unit tests for mean-ablation blockers."""

from __future__ import annotations

import torch
import pytest

from sfo.blockers import mean_ablate_batched
from sfo.hooks import add_to_residual_batched, capture_residual, _resolve_layers


def _tiny():
    from transformers import AutoModelForCausalLM

    for name in ("Gemma3TextConfig", "Gemma2Config", "GemmaConfig"):
        try:
            cls = getattr(__import__("transformers", fromlist=[name]), name)
            cfg = cls(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=6,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=256,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            m = AutoModelForCausalLM.from_config(cfg)
            m.eval()
            torch.set_grad_enabled(False)
            return m
        except Exception:
            continue
    pytest.skip("no gemma config")


@pytest.fixture(scope="module")
def model():
    return _tiny()


def _d(model):
    cfg = model.config
    return int(getattr(getattr(cfg, "text_config", cfg), "hidden_size", 64))


def test_blocker_identity_zero_direction(model):
    d = _d(model)
    ids = torch.randint(3, 50, (2, 10))
    with torch.no_grad():
        base = model(input_ids=ids).logits
    directions = {3: torch.zeros(d)}
    mus = {3: 0.0}
    starts = torch.tensor([0, 2])
    with mean_ablate_batched(model, [3], directions, mus, starts):
        with torch.no_grad():
            out = model(input_ids=ids).logits
    assert torch.equal(base, out)


def test_blocker_identity_empty_layers(model):
    ids = torch.randint(3, 50, (1, 8))
    with torch.no_grad():
        base = model(input_ids=ids).logits
    with mean_ablate_batched(model, [], {}, {}, torch.tensor([0])):
        with torch.no_grad():
            out = model(input_ids=ids).logits
    assert torch.equal(base, out)


def test_projection_equals_mu(model):
    d = _d(model)
    g = torch.Generator().manual_seed(0)
    direction = torch.randn(d, generator=g)
    direction = direction / direction.norm()
    mu = 1.5
    ids = torch.randint(3, 50, (1, 12))
    start = 4
    starts = torch.tensor([start])
    with mean_ablate_batched(
        model, [2], {2: direction}, {2: mu}, starts
    ):
        with capture_residual(model, [2]) as caps:
            with torch.no_grad():
                model(input_ids=ids)
    h = caps[2][0]  # [T,d] float32
    proj = (h * direction.float()).sum(-1)
    # positions >= start ≈ mu
    rel = (proj[start:] - mu).abs() / (abs(mu) + 1e-6)
    assert float(rel.max()) < 1e-2
    # before start: compare to unblocked
    with capture_residual(model, [2]) as caps2:
        with torch.no_grad():
            model(input_ids=ids)
    h0 = caps2[2][0]
    assert torch.allclose(h[:start], h0[:start], atol=1e-5)


def test_composition_batched_vs_single(model):
    d = _d(model)
    pad_id = 0
    seqs = [
        torch.tensor([1, 5, 6, 7, 8, 9]),
        torch.tensor([1, 10, 11, 12]),
    ]
    max_len = max(s.numel() for s in seqs)
    B = len(seqs)
    padded = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    pad_lens = []
    unpadded_starts = [2, 1]
    starts_p = []
    for i, s in enumerate(seqs):
        pl = max_len - s.numel()
        pad_lens.append(pl)
        padded[i, pl:] = s
        attn[i, pl:] = 1
        starts_p.append(pl + unpadded_starts[i])
    starts = torch.tensor(starts_p)
    g = torch.Generator().manual_seed(1)
    steer = (torch.randn(B, d, generator=g) * 3).to(torch.bfloat16)
    direction = torch.randn(d, generator=g)
    direction = direction / direction.norm()
    mus = {3: 0.0, 4: 0.0, 5: 0.0}
    dirs = {3: direction, 4: direction, 5: direction}

    def gen(ids, mask, st, vec, n_new=10):
        with add_to_residual_batched(model, 1, vec, st):
            with mean_ablate_batched(model, [3, 4, 5], dirs, mus, st):
                return model.generate(
                    input_ids=ids,
                    attention_mask=mask,
                    max_new_tokens=n_new,
                    do_sample=False,
                    use_cache=True,
                    pad_token_id=pad_id,
                )

    batched = gen(padded, attn, starts, steer, 10)
    for i in range(B):
        ids_i = seqs[i].unsqueeze(0)
        mask_i = torch.ones_like(ids_i)
        st_i = torch.tensor([unpadded_starts[i]])
        single = gen(ids_i, mask_i, st_i, steer[i : i + 1], 10)
        gb = batched[i, pad_lens[i] + seqs[i].numel() :][:10]
        gs = single[0, seqs[i].numel() :][:10]
        assert torch.equal(gb, gs), f"row {i}"
