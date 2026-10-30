"""Model and tokenizer loading, layer access, chat formatting."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence, Union

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "google/gemma-3-27b-it"
EXPECTED_N_LAYERS = 62
EXPECTED_D_MODEL = 5376


def load_model_revision(config_path: Union[str, Path]) -> str:
    """Load the pinned model revision SHA from configs/model_revision.yaml."""
    path = Path(config_path)
    with path.open() as f:
        data = yaml.safe_load(f)
    sha = data["revision_sha"]
    if not isinstance(sha, str) or len(sha) < 7:
        raise ValueError(f"Invalid revision_sha in {path}: {sha!r}")
    return sha


def load_tokenizer(
    model_path: Union[str, Path],
    revision: str,
) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        revision=revision,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_path: Union[str, Path],
    revision: str,
    device: int = 0,
) -> AutoModelForCausalLM:
    """Load Gemma-3-27B-it in bf16 with eager attention. Never quantized."""
    torch.set_grad_enabled(False)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        revision=revision,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        device_map={"": device},
        local_files_only=True,
    )
    model.eval()
    torch.set_grad_enabled(False)
    return model


def get_layers(model: Any) -> torch.nn.ModuleList:
    """Return decoder layer list. Prefer language_model.layers (Gemma 3 multimodal)."""
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            layers = lm.layers
            assert len(layers) == EXPECTED_N_LAYERS, (
                f"Expected {EXPECTED_N_LAYERS} layers, got {len(layers)}"
            )
            return layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
        assert len(layers) == EXPECTED_N_LAYERS, (
            f"Expected {EXPECTED_N_LAYERS} layers, got {len(layers)}"
        )
        return layers
    raise AttributeError("Could not locate decoder layers on model")


def get_d_model(model: Any) -> int:
    """Record d_model from config (expected 5376)."""
    cfg = model.config
    d = None
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        d = int(cfg.text_config.hidden_size)
    elif hasattr(cfg, "hidden_size"):
        d = int(cfg.hidden_size)
    else:
        raise AttributeError("Could not read hidden_size from model config")
    assert d == EXPECTED_D_MODEL, f"Expected d_model={EXPECTED_D_MODEL}, got {d}"
    return d


def assert_hidden_width(model: Any, hidden: torch.Tensor) -> None:
    """Assert config d_model matches observed hidden-state width."""
    d_model = get_d_model(model)
    assert hidden.shape[-1] == d_model, (
        f"Hidden width {hidden.shape[-1]} != config d_model {d_model}"
    )


def format_chat(
    tokenizer: AutoTokenizer,
    messages: Sequence[dict],
) -> torch.Tensor:
    """
    Apply chat template with generation prompt.
    Asserts exactly one BOS token at position 0 (no double BOS).
    """
    result = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
    )
    # transformers may return Tensor, BatchEncoding, or tokenizers.Encoding
    if isinstance(result, torch.Tensor):
        input_ids = result
    elif hasattr(result, "input_ids"):
        input_ids = result.input_ids
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids)
    elif hasattr(result, "ids"):
        input_ids = torch.tensor([list(result.ids)], dtype=torch.long)
    else:
        input_ids = torch.as_tensor(result, dtype=torch.long)

    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)

    bos_id = tokenizer.bos_token_id
    assert bos_id is not None, "tokenizer.bos_token_id is None"
    assert input_ids.ndim == 2 and input_ids.shape[0] == 1
    assert int(input_ids[0, 0].item()) == int(bos_id), (
        f"Expected BOS ({bos_id}) at position 0, got {int(input_ids[0, 0].item())}"
    )
    bos_count = int((input_ids[0] == bos_id).sum().item())
    assert bos_count == 1, f"Expected exactly one BOS token, found {bos_count}"
    return input_ids


def first_param_dtype(model: Any) -> torch.dtype:
    """Dtype of the first parameter in the first decoder layer."""
    layers = get_layers(model)
    for p in layers[0].parameters():
        return p.dtype
    raise RuntimeError("Layer 0 has no parameters")
