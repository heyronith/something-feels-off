"""Concept-vector extraction at a decoder layer (last-token residual)."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import torch

from sfo.hooks import capture_residual
from sfo.model import format_chat, get_d_model, get_layers


def format_chat_text(tokenizer: Any, messages: Sequence[dict]) -> str:
    """Chat-formatted string with generation prompt (Gemma: no system role)."""
    text = tokenizer.apply_chat_template(
        list(messages),
        add_generation_prompt=True,
        tokenize=False,
    )
    assert isinstance(text, str)
    return text


def compute_start_pos(tokenizer: Any, formatted: str, trial_n: int) -> int:
    """
    Token just before "Trial {n}" in the formatted prompt.
    start_pos = len(tokenize(s[:s.find(Trial n)], add_special_tokens=False)) - 1
    """
    marker = f"Trial {trial_n}"
    idx = formatted.find(marker)
    if idx < 0:
        raise ValueError(f"{marker!r} not found in formatted prompt")
    prefix = formatted[:idx]
    ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return int(ids.numel()) - 1


def pad_batch(
    tokenizer: Any,
    input_id_list: List[torch.Tensor],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """Left-pad a list of [1, L] or [L] id tensors. Returns ids, attn, pad_lengths."""
    assert tokenizer.padding_side == "left"
    pad_id = tokenizer.pad_token_id
    rows = []
    for t in input_id_list:
        if t.ndim == 2:
            t = t[0]
        rows.append(t.tolist())
    max_len = max(len(r) for r in rows)
    padded = []
    attn = []
    pad_lens = []
    for r in rows:
        pl = max_len - len(r)
        pad_lens.append(pl)
        padded.append([pad_id] * pl + r)
        attn.append([0] * pl + [1] * len(r))
    ids = torch.tensor(padded, device=device, dtype=torch.long)
    mask = torch.tensor(attn, device=device, dtype=torch.long)
    return ids, mask, pad_lens


@torch.no_grad()
def last_token_layer_acts(
    model: Any,
    tokenizer: Any,
    words: Sequence[str],
    layer_idx: int,
    batch_size: int = 16,
) -> torch.Tensor:
    """
    For each word w, forward "Tell me about {w}" and return layer output at last
    non-pad token. Shape [N, d_model] float32 CPU.
    """
    device = next(model.parameters()).device
    d_model = get_d_model(model)
    # Verify layer exists
    _ = get_layers(model)[layer_idx]

    all_acts: List[torch.Tensor] = []
    for start in range(0, len(words), batch_size):
        batch_words = list(words[start : start + batch_size])
        id_list = []
        for w in batch_words:
            messages = [{"role": "user", "content": f"Tell me about {w}"}]
            ids = format_chat(tokenizer, messages)
            id_list.append(ids)
        input_ids, attn, pad_lens = pad_batch(tokenizer, id_list, device)
        # Assert last token of each row is not pad
        for i, pl in enumerate(pad_lens):
            last = input_ids[i, -1]
            assert int(last.item()) != int(tokenizer.pad_token_id), (
                f"last token is pad for word {batch_words[i]!r}"
            )

        with capture_residual(model, [layer_idx]) as caps:
            model(input_ids=input_ids, attention_mask=attn)
        hidden = caps[layer_idx]  # [B, T, d] float32 CPU
        # Last token index = -1 (left-padded; real tokens at the end)
        acts = hidden[:, -1, :]
        assert acts.shape == (len(batch_words), d_model)
        all_acts.append(acts)

    return torch.cat(all_acts, dim=0)


def compute_concept_vectors(
    model: Any,
    tokenizer: Any,
    concepts: Sequence[str],
    baselines: Sequence[str],
    layer_idx: int = 37,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    v_c = act(c) - mean_b act(b) in float32, unnormalized.
    Returns dict with vectors [50, d], norms, mean_baseline, etc.
    """
    base_acts = last_token_layer_acts(
        model, tokenizer, baselines, layer_idx=layer_idx, batch_size=batch_size
    )
    mean_b = base_acts.mean(dim=0)  # [d]
    concept_acts = last_token_layer_acts(
        model, tokenizer, concepts, layer_idx=layer_idx, batch_size=batch_size
    )
    vectors = concept_acts - mean_b.unsqueeze(0)
    norms = vectors.norm(dim=-1)
    assert torch.isfinite(norms).all() and (norms > 0).all()

    # Pairwise cosines
    vn = vectors / norms.unsqueeze(1)
    cos = vn @ vn.T
    # Exclude diagonal
    n = cos.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool)
    off = cos[mask]
    max_off = float(off.max().item())
    mean_off = float(off.mean().item())
    assert max_off <= 0.99, f"concept cosine {max_off} > 0.99"

    return {
        "concepts": list(concepts),
        "vectors": vectors,  # float32 CPU
        "norms": norms,
        "mean_baseline": mean_b,
        "mean_pairwise_cosine": mean_off,
        "max_pairwise_cosine": max_off,
        "layer_idx": layer_idx,
    }


def residual_norm_diagnostics(
    model: Any,
    tokenizer: Any,
    formatted_ids: torch.Tensor,
    layer_idx: int = 37,
) -> Dict[str, float]:
    """
    Layer residual L2 norms on an unsteered forward.
    Excluding BOS: median and IQR; separately BOS-position norm.
    """
    device = next(model.parameters()).device
    ids = formatted_ids.to(device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    with capture_residual(model, [layer_idx]) as caps:
        model(input_ids=ids)
    h = caps[layer_idx][0]  # [T, d] float32
    norms = h.norm(dim=-1)  # [T]
    bos_norm = float(norms[0].item())
    excl = norms[1:]
    median = float(excl.median().item())
    q1 = float(excl.quantile(0.25).item())
    q3 = float(excl.quantile(0.75).item())
    return {
        "resid_median_excl_bos": median,
        "resid_iqr_excl_bos": q3 - q1,
        "bos_norm": bos_norm,
        "q1_excl_bos": q1,
        "q3_excl_bos": q3,
    }
