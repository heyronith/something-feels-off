"""Detection logit LD = logit[Yes] − logit[No] helpers."""

from __future__ import annotations

from typing import Any, Tuple

import torch


def yes_no_token_ids(tokenizer: Any) -> Tuple[int, int]:
    """Single-token ids for 'Yes' and 'No' (no leading space)."""
    yes_ids = tokenizer.encode("Yes", add_special_tokens=False)
    no_ids = tokenizer.encode("No", add_special_tokens=False)
    assert len(yes_ids) == 1, f"'Yes' not single token: {yes_ids}"
    assert len(no_ids) == 1, f"'No' not single token: {no_ids}"
    return int(yes_ids[0]), int(no_ids[0])


def detection_logit(
    logits_last: torch.Tensor,
    yes_id: int,
    no_id: int,
) -> torch.Tensor:
    """
    LD from last-position logits.
    logits_last: [B, vocab] or [vocab]
    returns [B] or scalar tensor
    """
    if logits_last.ndim == 1:
        return logits_last[yes_id].float() - logits_last[no_id].float()
    return logits_last[:, yes_id].float() - logits_last[:, no_id].float()


def concept_match_string(concept: str, text: str) -> str | None:
    """Case-insensitive substring match of concept or singular (strip s/es)."""
    t = (text or "").lower()
    c = concept.lower()
    variants = [c]
    if c.endswith("es") and len(c) > 3:
        variants.append(c[:-2])
    elif c.endswith("s") and len(c) > 2:
        variants.append(c[:-1])
    for v in variants:
        if v and v in t:
            return v
    return None
