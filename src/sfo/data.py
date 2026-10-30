"""Dataset loaders for MMLU and WikiText (CPU download; ids + revisions saved)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml


def resolve_dataset_revision(repo_id: str, config_name: str | None = None) -> str:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(repo_id)
    return info.sha


def load_mmlu_validation(
    n: int = 200,
    seed: int = 0,
    cache_dir: str | None = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    200 questions from cais/mmlu, config 'all', split validation, shuffle seed 0.
    Returns (items, revision_sha).
    """
    from datasets import load_dataset
    import random

    revision = resolve_dataset_revision("cais/mmlu")
    ds = load_dataset(
        "cais/mmlu",
        "all",
        split="validation",
        revision=revision,
        cache_dir=cache_dir,
    )
    idxs = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    idxs = idxs[:n]
    items = []
    for i in idxs:
        row = ds[int(i)]
        items.append(
            {
                "id": int(i),
                "question": row["question"],
                "choices": list(row["choices"]),
                "answer": int(row["answer"]),  # 0..3
                "subject": row.get("subject"),
            }
        )
    return items, revision


def format_mmlu_user(item: Dict[str, Any]) -> str:
    letters = ["A", "B", "C", "D"]
    lines = [item["question"], ""]
    for let, ch in zip(letters, item["choices"]):
        lines.append(f"{let}. {ch}")
    lines.append("")
    lines.append("Answer with the letter only.")
    return "\n".join(lines)


def load_wikitext_chunks(
    n: int = 100,
    chunk_tokens: int = 64,
    cache_dir: str | None = None,
    tokenizer=None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    First 100 text chunks with ≥ 64 tokens from Salesforce/wikitext
    wikitext-103-raw-v1 validation, as raw text (no chat template).
    """
    from datasets import load_dataset

    revision = resolve_dataset_revision("Salesforce/wikitext")
    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="validation",
        revision=revision,
        cache_dir=cache_dir,
    )
    chunks: List[Dict[str, Any]] = []
    for i, row in enumerate(ds):
        text = row["text"]
        if not text or not text.strip():
            continue
        if tokenizer is not None:
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < chunk_tokens:
                continue
            # take first 64 tokens worth of text via decode
            piece_ids = ids[:chunk_tokens]
            piece = tokenizer.decode(piece_ids)
        else:
            # whitespace approx; refined later with tokenizer on GPU side
            words = text.split()
            if len(words) < chunk_tokens // 2:
                continue
            piece = " ".join(words[:chunk_tokens])
        chunks.append({"id": int(i), "text": piece})
        if len(chunks) >= n:
            break
    if len(chunks) < n:
        raise RuntimeError(f"Only found {len(chunks)} wikitext chunks ≥ {chunk_tokens} tokens")
    return chunks, revision


def save_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
