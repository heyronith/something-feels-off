"""CPU unit tests for add_to_residual_batched."""

from __future__ import annotations

import torch
import pytest

from sfo.hooks import add_to_residual_batched, _resolve_layers


def _tiny_model():
    from transformers import AutoModelForCausalLM

    for cfg_name in ("Gemma3TextConfig", "Gemma2Config", "GemmaConfig"):
        try:
            mod = __import__("transformers", fromlist=[cfg_name])
            cfg_cls = getattr(mod, cfg_name)
        except AttributeError:
            continue
        try:
            config = cfg_cls(
                vocab_size=128,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=16,
                max_position_embeddings=256,
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
            )
            model = AutoModelForCausalLM.from_config(config)
            model.eval()
            torch.set_grad_enabled(False)
            return model
        except TypeError:
            continue
    pytest.skip("No Gemma config available")


@pytest.fixture(scope="module")
def tiny():
    return _tiny_model()


def _d_model(model):
    cfg = model.config
    return int(getattr(getattr(cfg, "text_config", cfg), "hidden_size", 64))


def test_batched_zero_vectors_identical(tiny):
    model = tiny
    d = _d_model(model)
    B, T = 3, 8
    input_ids = torch.randint(3, 50, (B, T))
    with torch.no_grad():
        base = model(input_ids=input_ids).logits
    zeros = torch.zeros(B, d, dtype=torch.bfloat16)
    starts = torch.zeros(B, dtype=torch.long)
    with add_to_residual_batched(model, 1, zeros, starts):
        with torch.no_grad():
            hooked = model(input_ids=input_ids).logits
    assert torch.equal(base, hooked)


def test_batched_equals_single_greedy(tiny):
    """Batched result equals running each row alone (greedy 10 tokens, different pads)."""
    model = tiny
    d = _d_model(model)
    pad_id = 0
    # Different lengths → different left-pad when batched
    seqs = [
        torch.tensor([1, 5, 6, 7, 8]),
        torch.tensor([1, 9, 10]),
        torch.tensor([1, 11, 12, 13, 14, 15, 16]),
    ]
    max_len = max(s.numel() for s in seqs)
    B = len(seqs)
    padded = torch.full((B, max_len), pad_id, dtype=torch.long)
    attn = torch.zeros(B, max_len, dtype=torch.long)
    pad_lens = []
    unpadded_starts = [1, 0, 2]  # vary start within content
    starts_padded = []
    for i, s in enumerate(seqs):
        pl = max_len - s.numel()
        pad_lens.append(pl)
        padded[i, pl:] = s
        attn[i, pl:] = 1
        starts_padded.append(pl + unpadded_starts[i])
    starts = torch.tensor(starts_padded, dtype=torch.long)
    g = torch.Generator().manual_seed(0)
    vectors = torch.randn(B, d, generator=g, dtype=torch.float32)
    vectors = (vectors / vectors.norm(dim=-1, keepdim=True) * 5.0).to(torch.bfloat16)

    def greedy_gen(ids, mask, vecs, st, n_new=10):
        with add_to_residual_batched(model, 1, vecs, st):
            out = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_new_tokens=n_new,
                do_sample=False,
                use_cache=True,
                pad_token_id=pad_id,
            )
        return out

    batched = greedy_gen(padded, attn, vectors, starts, 10)

    singles = []
    for i in range(B):
        ids_i = seqs[i].unsqueeze(0)
        mask_i = torch.ones_like(ids_i)
        st_i = torch.tensor([unpadded_starts[i]], dtype=torch.long)
        vec_i = vectors[i : i + 1]
        singles.append(greedy_gen(ids_i, mask_i, vec_i, st_i, 10)[0])

    for i in range(B):
        # Compare newly generated tokens (after original unpadded length).
        # Batched generate may right-pad shorter sequences with pad_id.
        gen_b = batched[i, pad_lens[i] + seqs[i].numel() :][:10]
        gen_s = singles[i][seqs[i].numel() :][:10]
        assert torch.equal(gen_b, gen_s), f"row {i} mismatch {gen_b} vs {gen_s}"


def test_no_change_before_start_pos(tiny):
    model = tiny
    d = _d_model(model)
    T = 12
    input_ids = torch.randint(3, 50, (1, T))
    with torch.no_grad():
        base = model(input_ids=input_ids, output_hidden_states=False).logits
    start = 7
    vec = torch.randn(1, d, dtype=torch.bfloat16) * 8.0
    starts = torch.tensor([start], dtype=torch.long)
    with add_to_residual_batched(model, 2, vec, starts):
        with torch.no_grad():
            hooked = model(input_ids=input_ids).logits
    # Logits at positions predicting tokens before start should match
    # (logits[t] predicts token t+1; residual at layer affects from start onward)
    assert torch.equal(base[0, :start], hooked[0, :start])
    assert not torch.equal(base[0, start:], hooked[0, start:])
