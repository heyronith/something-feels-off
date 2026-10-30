"""CPU-only unit tests for residual-stream hooks on a tiny Gemma3 text model."""

from __future__ import annotations

import torch
import pytest

from sfo.hooks import add_to_residual, capture_residual, _resolve_layers


def _tiny_gemma3():
    """Randomly initialized tiny Gemma/Gemma3 text model (no download)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    # Prefer Gemma3TextConfig; fall back to Gemma2Config / GemmaConfig.
    config = None
    for cfg_name in ("Gemma3TextConfig", "Gemma2Config", "GemmaConfig"):
        try:
            mod = __import__("transformers", fromlist=[cfg_name])
            cfg_cls = getattr(mod, cfg_name)
        except AttributeError:
            continue
        kwargs = dict(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            rms_norm_eps=1e-6,
            rope_theta=10000.0,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        if cfg_name == "Gemma3TextConfig":
            kwargs.update(
                query_pre_attn_scalar=16,
                sliding_window=64,
                use_bidirectional_attention=False,
            )
        try:
            config = cfg_cls(**kwargs)
            break
        except TypeError:
            # Drop unknown kwargs and retry with minimal set
            try:
                config = cfg_cls(
                    vocab_size=128,
                    hidden_size=64,
                    intermediate_size=128,
                    num_hidden_layers=4,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=16,
                )
                break
            except TypeError:
                continue

    if config is None:
        pytest.skip("No Gemma/Gemma2/Gemma3TextConfig available in transformers")

    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    torch.set_grad_enabled(False)
    return model


def _forward_logits(model, input_ids):
    with torch.no_grad():
        out = model(input_ids=input_ids)
    return out.logits


@pytest.fixture(scope="module")
def tiny_model():
    return _tiny_gemma3()


def test_zero_vector_identical_outputs(tiny_model):
    model = tiny_model
    layers = _resolve_layers(model)
    d_model = layers[0].self_attn.q_proj.in_features if hasattr(layers[0], "self_attn") else 64
    # Prefer config hidden size
    cfg = model.config
    d_model = getattr(getattr(cfg, "text_config", cfg), "hidden_size", d_model)

    input_ids = torch.randint(0, 50, (1, 8))
    baseline = _forward_logits(model, input_ids)

    zero = torch.zeros(d_model, dtype=torch.bfloat16)
    with add_to_residual(model, layer_idx=1, vector=zero, positions="all"):
        hooked = _forward_logits(model, input_ids)

    assert torch.equal(baseline, hooked), "zero vector must leave logits unchanged"


def test_nonzero_vector_changes_outputs(tiny_model):
    model = tiny_model
    cfg = model.config
    d_model = getattr(getattr(cfg, "text_config", cfg), "hidden_size", 64)

    input_ids = torch.randint(0, 50, (1, 8))
    baseline = _forward_logits(model, input_ids)

    vec = torch.randn(d_model, dtype=torch.bfloat16) * 10.0
    with add_to_residual(model, layer_idx=1, vector=vec, positions="all"):
        hooked = _forward_logits(model, input_ids)

    assert not torch.equal(baseline, hooked), "non-zero vector must change logits"


def test_hooks_removed_after_context(tiny_model):
    model = tiny_model
    cfg = model.config
    d_model = getattr(getattr(cfg, "text_config", cfg), "hidden_size", 64)
    layer = _resolve_layers(model)[1]

    input_ids = torch.randint(0, 50, (1, 8))
    n_before = len(layer._forward_hooks)
    vec = torch.randn(d_model, dtype=torch.bfloat16)
    with add_to_residual(model, layer_idx=1, vector=vec):
        assert len(layer._forward_hooks) == n_before + 1
        _forward_logits(model, input_ids)
    assert len(layer._forward_hooks) == n_before

    baseline = _forward_logits(model, input_ids)
    after = _forward_logits(model, input_ids)
    assert torch.equal(baseline, after)


def test_hooks_removed_after_exception(tiny_model):
    model = tiny_model
    cfg = model.config
    d_model = getattr(getattr(cfg, "text_config", cfg), "hidden_size", 64)
    layer = _resolve_layers(model)[2]
    n_before = len(layer._forward_hooks)
    vec = torch.ones(d_model, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="boom"):
        with add_to_residual(model, layer_idx=2, vector=vec):
            assert len(layer._forward_hooks) == n_before + 1
            raise RuntimeError("boom")

    assert len(layer._forward_hooks) == n_before


def test_capture_residual_shapes(tiny_model):
    model = tiny_model
    cfg = model.config
    d_model = getattr(getattr(cfg, "text_config", cfg), "hidden_size", 64)
    input_ids = torch.randint(0, 50, (1, 7))
    with capture_residual(model, layer_idxs=[0, 2]) as caps:
        _forward_logits(model, input_ids)
    assert set(caps.keys()) == {0, 2}
    for t in caps.values():
        assert t.shape == (1, 7, d_model)
        assert t.dtype == torch.float32
        assert t.device.type == "cpu"
        assert torch.isfinite(t).all()
