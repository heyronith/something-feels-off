"""CPU unit tests for paired-twin B1′."""

from __future__ import annotations

import torch
import pytest

from sfo.paired import (
    b1prime_from_l0,
    paired_forward_logits,
    paired_generate,
    unit,
)


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


def test_twin_logits_match_unpaired(model):
    """(i) twin row logits equal unpaired unsteered run on same tokens."""
    d = _d(model)
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 50, (2, 10), generator=g)
    attn = torch.ones_like(ids)
    starts = torch.tensor([1, 1])
    vec = torch.randn(2, d, generator=g) * 0.5

    with torch.no_grad():
        base = model(input_ids=ids, attention_mask=attn).logits
        logits_2b = paired_forward_logits(
            model,
            ids,
            attn,
            starts,
            steer_vecs=vec,
            steer_layer=2,
            blocker_dirs=None,
            l0=None,
        )
    twin = logits_2b[2:]
    assert torch.allclose(twin.float(), base.float(), atol=1e-4, rtol=1e-4)


def test_alpha0_b1prime_identity(model):
    """(ii) α=0 with B1′ active → bit-identical to no blocker."""
    d = _d(model)
    ids = torch.randint(3, 50, (2, 8))
    attn = torch.ones_like(ids)
    starts = torch.tensor([1, 2])
    zeros = torch.zeros(2, d)
    direction = unit(torch.randn(d))
    dirs = {3: direction, 4: direction, 5: direction}

    with torch.no_grad():
        none = paired_forward_logits(
            model, ids, attn, starts, zeros, 2, None, None
        )
        blk = paired_forward_logits(
            model, ids, attn, starts, zeros, 2, dirs, l0=3
        )
    assert torch.equal(none, blk)


def test_zero_direction_identity(model):
    """(iii) zero direction B1′ → bit-identical to no blocker."""
    d = _d(model)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, 50, (2, 8), generator=g)
    attn = torch.ones_like(ids)
    starts = torch.tensor([0, 1])
    vec = torch.randn(2, d, generator=g)
    dirs = {3: torch.zeros(d), 4: torch.zeros(d)}

    with torch.no_grad():
        none = paired_forward_logits(
            model, ids, attn, starts, vec, 2, None, None
        )
        blk = paired_forward_logits(
            model, ids, attn, starts, vec, 2, dirs, l0=3
        )
    assert torch.equal(none, blk)


def test_paired_greedy_matches_hf(model):
    """(iv) greedy paired_generate with no hooks matches HF greedy generate."""
    g = torch.Generator().manual_seed(2)
    ids = torch.randint(3, 50, (1, 6), generator=g)
    attn = torch.ones_like(ids)
    starts = torch.tensor([0])
    d = _d(model)
    zeros = torch.zeros(1, d)

    with torch.no_grad():
        hf = model.generate(
            input_ids=ids,
            attention_mask=attn,
            max_new_tokens=10,
            do_sample=False,
            use_cache=True,
        )
        steered, twin = paired_generate(
            model,
            ids,
            attn,
            starts,
            steer_vecs=zeros,
            steer_layer=1,
            blocker_dirs=None,
            l0=None,
            max_new_tokens=10,
            do_sample=False,
            eos_ids=(999,),  # never stop early
        )
    assert torch.equal(steered[0], hf[0].cpu())
    assert torch.equal(steered[0], twin[0])
